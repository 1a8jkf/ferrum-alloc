#[cfg(windows)]
pub mod dllmain {
    use minhook::{MinHook, MH_STATUS};
    use crate::api::{ferrum_malloc, ferrum_free, ferrum_usable_size, ferrum_realloc};
    use core::ffi::c_void;
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    use windows_sys::Win32::System::SystemServices::DLL_PROCESS_ATTACH;

    // We will hook HeapAlloc and HeapFree from Kernel32
    // We will also try to hook malloc/free from msvcrt/ucrtbase if possible

    use core::sync::atomic::{AtomicPtr, Ordering};

    type FnHeapAlloc = unsafe extern "system" fn(isize, u32, usize) -> *mut c_void;
    type FnHeapFree = unsafe extern "system" fn(isize, u32, *mut c_void) -> i32;
    type FnHeapReAlloc = unsafe extern "system" fn(isize, u32, *mut c_void, usize) -> *mut c_void;
    type FnHeapSize = unsafe extern "system" fn(isize, u32, *mut c_void) -> usize;

    type FnMalloc = unsafe extern "C" fn(usize) -> *mut c_void;
    type FnFree = unsafe extern "C" fn(*mut c_void);

    // Globals for original functions
    static ORIGINAL_HEAP_ALLOC: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
    static ORIGINAL_HEAP_FREE: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
    static ORIGINAL_HEAP_REALLOC: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
    static ORIGINAL_HEAP_SIZE: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());

    static ORIGINAL_MALLOC: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
    static ORIGINAL_FREE: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());

    // ------------------------------------------------------------------------
    // Hooks
    // ------------------------------------------------------------------------

    unsafe extern "system" fn hooked_heap_alloc(hheap: isize, dwflags: u32, dwbytes: usize) -> *mut c_void {
        // We ignore the specific heap handle (hheap) and just allocate via Ferrum
        let ptr = ferrum_malloc(dwbytes);
        let orig_ptr = ORIGINAL_HEAP_ALLOC.load(Ordering::Relaxed);
        if ptr.is_null() && !orig_ptr.is_null() {
            let orig: FnHeapAlloc = std::mem::transmute(orig_ptr);
            return orig(hheap, dwflags, dwbytes);
        }
        ptr as *mut c_void
    }

    unsafe extern "system" fn hooked_heap_free(hheap: isize, dwflags: u32, lpmem: *mut c_void) -> i32 {
        if lpmem.is_null() {
            return 1; // TRUE
        }
        let usable = ferrum_usable_size(lpmem as *const u8);
        if usable > 0 {
            ferrum_free(lpmem as *mut u8);
            return 1;
        }

        let orig_ptr = ORIGINAL_HEAP_FREE.load(Ordering::Relaxed);
        if !orig_ptr.is_null() {
            let orig: FnHeapFree = std::mem::transmute(orig_ptr);
            return orig(hheap, dwflags, lpmem);
        }
        1
    }

    unsafe extern "system" fn hooked_heap_realloc(hheap: isize, dwflags: u32, lpmem: *mut c_void, dwbytes: usize) -> *mut c_void {
        if lpmem.is_null() {
            return hooked_heap_alloc(hheap, dwflags, dwbytes);
        }
        
        let usable = ferrum_usable_size(lpmem as *const u8);
        if usable > 0 {
            return ferrum_realloc(lpmem as *mut u8, dwbytes) as *mut c_void;
        }

        let orig_ptr = ORIGINAL_HEAP_REALLOC.load(Ordering::Relaxed);
        if !orig_ptr.is_null() {
            let orig: FnHeapReAlloc = std::mem::transmute(orig_ptr);
            return orig(hheap, dwflags, lpmem, dwbytes);
        }
        core::ptr::null_mut()
    }

    unsafe extern "system" fn hooked_heap_size(hheap: isize, dwflags: u32, lpmem: *mut c_void) -> usize {
        if lpmem.is_null() {
            return 0; // -1 technically but usize
        }
        let usable = ferrum_usable_size(lpmem as *const u8);
        if usable > 0 {
            return usable;
        }

        let orig_ptr = ORIGINAL_HEAP_SIZE.load(Ordering::Relaxed);
        if !orig_ptr.is_null() {
            let orig: FnHeapSize = std::mem::transmute(orig_ptr);
            return orig(hheap, dwflags, lpmem);
        }
        0
    }

    unsafe extern "C" fn hooked_malloc(size: usize) -> *mut c_void {
        ferrum_malloc(size) as *mut c_void
    }

    unsafe extern "C" fn hooked_free(ptr: *mut c_void) {
        if ptr.is_null() { return; }
        let usable = ferrum_usable_size(ptr as *const u8);
        if usable > 0 {
            ferrum_free(ptr as *mut u8);
        } else {
            let orig_ptr = ORIGINAL_FREE.load(Ordering::Relaxed);
            if !orig_ptr.is_null() {
                let orig: FnFree = std::mem::transmute(orig_ptr);
                orig(ptr);
            }
        }
    }

    // ------------------------------------------------------------------------
    // DllMain Entry Point
    // ------------------------------------------------------------------------
    
    #[unsafe(no_mangle)]
    #[allow(non_snake_case)]
    pub unsafe extern "system" fn DllMain(_hinst_dll: *mut c_void, fdw_reason: u32, _lpv_reserved: *mut c_void) -> i32 {
        if fdw_reason == DLL_PROCESS_ATTACH {
            // Hook Kernel32 Heap functions
                let kernel32 = GetModuleHandleA(c"kernel32.dll".as_ptr() as *const u8);
                if !kernel32.is_null() {
                    let heap_alloc = GetProcAddress(kernel32, c"HeapAlloc".as_ptr() as *const u8);
                    if let Some(addr) = heap_alloc {
                        ORIGINAL_HEAP_ALLOC.store(MinHook::create_hook(addr as *mut _, hooked_heap_alloc as *mut _).unwrap_or(core::ptr::null_mut()), Ordering::SeqCst);
                    }

                    let heap_free = GetProcAddress(kernel32, c"HeapFree".as_ptr() as *const u8);
                    if let Some(addr) = heap_free {
                        ORIGINAL_HEAP_FREE.store(MinHook::create_hook(addr as *mut _, hooked_heap_free as *mut _).unwrap_or(core::ptr::null_mut()), Ordering::SeqCst);
                    }

                    let heap_realloc = GetProcAddress(kernel32, c"HeapReAlloc".as_ptr() as *const u8);
                    if let Some(addr) = heap_realloc {
                        ORIGINAL_HEAP_REALLOC.store(MinHook::create_hook(addr as *mut _, hooked_heap_realloc as *mut _).unwrap_or(core::ptr::null_mut()), Ordering::SeqCst);
                    }

                    let heap_size = GetProcAddress(kernel32, c"HeapSize".as_ptr() as *const u8);
                    if let Some(addr) = heap_size {
                        ORIGINAL_HEAP_SIZE.store(MinHook::create_hook(addr as *mut _, hooked_heap_size as *mut _).unwrap_or(core::ptr::null_mut()), Ordering::SeqCst);
                    }
                }

                // Hook msvcrt malloc/free if present
                let msvcrt = GetModuleHandleA(c"msvcrt.dll".as_ptr() as *const u8);
                if !msvcrt.is_null() {
                    let m_alloc = GetProcAddress(msvcrt, c"malloc".as_ptr() as *const u8);
                    if let Some(addr) = m_alloc {
                        ORIGINAL_MALLOC.store(MinHook::create_hook(addr as *mut _, hooked_malloc as *mut _).unwrap_or(core::ptr::null_mut()), Ordering::SeqCst);
                    }

                    let m_free = GetProcAddress(msvcrt, c"free".as_ptr() as *const u8);
                    if let Some(addr) = m_free {
                        ORIGINAL_FREE.store(MinHook::create_hook(addr as *mut _, hooked_free as *mut _).unwrap_or(core::ptr::null_mut()), Ordering::SeqCst);
                    }
                }

                let _ = MinHook::enable_all_hooks();
        }
        1 // TRUE
    }
}
