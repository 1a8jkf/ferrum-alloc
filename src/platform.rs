//! Platform abstraction layer for OS memory operations.
//!
//! Provides a unified interface over `VirtualAlloc`/`VirtualFree` (Windows)
//! and `mmap`/`munmap` (Linux). All allocations are page-aligned and
//! requested in large chunks (arena-sized) to amortize syscall overhead.

use core::ptr;

/// Default arena size: 2 MB (aligned with huge pages on Linux / large pages on Windows)
pub const DEFAULT_ARENA_SIZE: usize = 2 * 1024 * 1024;

/// Minimum page size (4 KB on both platforms)
pub const PAGE_SIZE: usize = 4096;

/// Allocate `size` bytes of memory directly from the OS.
///
/// The returned pointer is page-aligned. `size` is rounded up to the nearest
/// page boundary. Returns null on failure.
///
/// # Safety
///
/// The caller must ensure `size > 0` and must eventually call `os_free` on the
/// returned pointer with the same `size` to avoid leaking OS resources.
pub unsafe fn os_alloc(size: usize) -> *mut u8 {
    let aligned_size = align_up(size, PAGE_SIZE);

    #[cfg(target_os = "windows")]
    {
        unsafe { windows::virtual_alloc(aligned_size) }
    }

    #[cfg(target_os = "linux")]
    {
        unsafe { linux::mmap_alloc(aligned_size) }
    }

    #[cfg(miri)]
    {
        use std::alloc::{alloc_zeroed, Layout};
        let layout = Layout::from_size_align(aligned_size, PAGE_SIZE).unwrap();
        unsafe { alloc_zeroed(layout) }
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", miri)))]
    {
        let _ = aligned_size;
        ptr::null_mut()
    }
}

/// Free `size` bytes of memory previously allocated with `os_alloc`.
///
/// # Safety
///
/// - `ptr` must have been returned by `os_alloc`.
/// - `size` must match the size passed to `os_alloc` (before internal alignment).
/// - `ptr` must not be used after this call.
pub unsafe fn os_free(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    let aligned_size = align_up(size, PAGE_SIZE);

    #[cfg(target_os = "windows")]
    {
        unsafe { windows::virtual_free(ptr, aligned_size); }
    }

    #[cfg(target_os = "linux")]
    {
        unsafe { linux::mmap_free(ptr, aligned_size); }
    }

    #[cfg(miri)]
    {
        use std::alloc::{dealloc, Layout};
        let layout = Layout::from_size_align(aligned_size, PAGE_SIZE).unwrap();
        unsafe { dealloc(ptr, layout) }
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", miri)))]
    {
        let _ = (ptr, aligned_size);
    }
}

// ============================================================================
// Thread-Local Storage (TLS)
// ============================================================================

#[cfg(miri)]
mod miri_tls {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU32, Ordering};

    static NEXT_TLS_KEY: AtomicU32 = AtomicU32::new(1);

    std::thread_local! {
        static MIRI_TLS: RefCell<std::collections::HashMap<u32, *mut u8>> = RefCell::new(std::collections::HashMap::new());
    }

    pub fn tls_create() -> u32 {
        NEXT_TLS_KEY.fetch_add(1, Ordering::SeqCst)
    }

    pub fn tls_get(key: u32) -> *mut u8 {
        MIRI_TLS.with(|tls| {
            tls.borrow().get(&key).copied().unwrap_or(core::ptr::null_mut())
        })
    }

    pub fn tls_set(key: u32, value: *mut u8) {
        MIRI_TLS.with(|tls| {
            tls.borrow_mut().insert(key, value);
        })
    }
}

/// Create a new TLS key.
pub fn tls_create() -> u32 {
    #[cfg(miri)]
    {
        miri_tls::tls_create()
    }
    
    #[cfg(all(target_os = "windows", not(miri)))]
    {
        windows::tls_create()
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    {
        linux::tls_create()
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", miri)))]
    {
        unimplemented!("TLS not supported on this platform")
    }
}

/// Get the thread-local value for the given key.
pub fn tls_get(key: u32) -> *mut u8 {
    #[cfg(miri)]
    {
        miri_tls::tls_get(key)
    }

    #[cfg(all(target_os = "windows", not(miri)))]
    {
        windows::tls_get(key)
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    {
        linux::tls_get(key)
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", miri)))]
    {
        let _ = key;
        ptr::null_mut()
    }
}

/// Set the thread-local value for the given key.
pub fn tls_set(key: u32, value: *mut u8) {
    #[cfg(miri)]
    {
        miri_tls::tls_set(key, value)
    }

    #[cfg(all(target_os = "windows", not(miri)))]
    {
        windows::tls_set(key, value)
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    {
        linux::tls_set(key, value)
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", miri)))]
    {
        let _ = (key, value);
    }
}

/// Align `value` up to the nearest multiple of `align`.
/// `align` must be a power of two.
#[inline(always)]
pub const fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// Align a pointer up to the nearest multiple of `align`.
/// `align` must be a power of two.
#[inline(always)]
pub fn align_ptr_up(ptr: *mut u8, align: usize) -> *mut u8 {
    align_up(ptr as usize, align) as *mut u8
}

// ============================================================================
// Windows implementation via VirtualAlloc / VirtualFree
// ============================================================================

#[cfg(target_os = "windows")]
mod windows {
    use core::ffi::c_void;
    use core::ptr;

    // Windows API constants
    const MEM_COMMIT: u32 = 0x00001000;
    const MEM_RESERVE: u32 = 0x00002000;
    const MEM_RELEASE: u32 = 0x00008000;
    const PAGE_READWRITE: u32 = 0x04;

    // FFI declarations — we declare these ourselves to avoid depending on
    // the `windows` crate, keeping the core `no_std`.
    unsafe extern "system" {
        fn VirtualAlloc(
            lp_address: *mut c_void,
            dw_size: usize,
            fl_allocation_type: u32,
            fl_protect: u32,
        ) -> *mut c_void;

        fn VirtualFree(
            lp_address: *mut c_void,
            dw_size: usize,
            dw_free_type: u32,
        ) -> i32;

        fn TlsAlloc() -> u32;
        fn TlsGetValue(dw_tls_index: u32) -> *mut c_void;
        fn TlsSetValue(dw_tls_index: u32, lp_tls_value: *mut c_void) -> i32;
    }

    /// Allocate `size` bytes via VirtualAlloc (MEM_COMMIT | MEM_RESERVE).
    pub unsafe fn virtual_alloc(size: usize) -> *mut u8 {
        let result = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        result as *mut u8
    }

    /// Free memory via VirtualFree (MEM_RELEASE).
    /// Note: when using MEM_RELEASE, dwSize must be 0 per Windows API spec.
    pub unsafe fn virtual_free(ptr: *mut u8, _size: usize) {
        unsafe {
            VirtualFree(ptr as *mut c_void, 0, MEM_RELEASE);
        }
    }

    pub fn tls_create() -> u32 {
        let key = unsafe { TlsAlloc() };
        assert!(key != 0xFFFFFFFF, "TlsAlloc failed");
        key
    }

    pub fn tls_get(key: u32) -> *mut u8 {
        unsafe { TlsGetValue(key) as *mut u8 }
    }

    pub fn tls_set(key: u32, value: *mut u8) {
        unsafe {
            TlsSetValue(key, value as *mut c_void);
        }
    }
}

// ============================================================================
// Linux implementation via mmap / munmap
// ============================================================================

#[cfg(target_os = "linux")]
mod linux {
    use core::ffi::c_void;
    use core::ptr;

    // mmap constants
    const PROT_READ: i32 = 0x1;
    const PROT_WRITE: i32 = 0x2;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;
    const MAP_FAILED: *mut c_void = !0 as *mut c_void;

    // FFI declarations for libc
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;

        fn munmap(addr: *mut c_void, length: usize) -> i32;
        
        // pthread TLS
        fn pthread_key_create(key: *mut u32, dtor: Option<extern "C" fn(*mut c_void)>) -> i32;
        fn pthread_getspecific(key: u32) -> *mut c_void;
        fn pthread_setspecific(key: u32, value: *mut c_void) -> i32;
    }

    /// Allocate `size` bytes via mmap (MAP_PRIVATE | MAP_ANONYMOUS).
    pub unsafe fn mmap_alloc(size: usize) -> *mut u8 {
        let result = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1, // no file descriptor
                0,  // no offset
            )
        };
        if result == MAP_FAILED {
            ptr::null_mut()
        } else {
            result as *mut u8
        }
    }

    /// Free `size` bytes via munmap.
    pub unsafe fn mmap_free(ptr: *mut u8, size: usize) {
        unsafe {
            munmap(ptr as *mut c_void, size);
        }
    }

    pub fn tls_create() -> u32 {
        let mut key: u32 = 0;
        let res = unsafe { pthread_key_create(&mut key, None) };
        assert_eq!(res, 0, "pthread_key_create failed");
        key
    }

    pub fn tls_get(key: u32) -> *mut u8 {
        unsafe { pthread_getspecific(key) as *mut u8 }
    }

    pub fn tls_set(key: u32, value: *mut u8) {
        unsafe {
            pthread_setspecific(key, value as *mut c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(100, 8), 104);
        assert_eq!(align_up(8, 8), 8);
    }

    #[test]
    fn test_os_alloc_free() {
        unsafe {
            let size = DEFAULT_ARENA_SIZE; // 2 MB
            let ptr = os_alloc(size);
            assert!(!ptr.is_null(), "os_alloc returned null");

            // Write to every page to ensure the memory is actually committed
            for i in (0..size).step_by(PAGE_SIZE) {
                ptr.add(i).write(0xAA);
            }

            // Read back to verify
            for i in (0..size).step_by(PAGE_SIZE) {
                assert_eq!(ptr.add(i).read(), 0xAA);
            }

            os_free(ptr, size);
        }
    }

    #[test]
    fn test_os_alloc_small() {
        unsafe {
            // Even a 1-byte request should work (gets rounded to PAGE_SIZE)
            let ptr = os_alloc(1);
            assert!(!ptr.is_null());
            ptr.write(42);
            assert_eq!(ptr.read(), 42);
            os_free(ptr, 1);
        }
    }

    #[test]
    fn test_os_alloc_multiple() {
        unsafe {
            let ptrs: [*mut u8; 4] = core::array::from_fn(|_| {
                let p = os_alloc(PAGE_SIZE * 4);
                assert!(!p.is_null());
                p
            });

            // All pointers should be distinct
            for i in 0..4 {
                for j in (i + 1)..4 {
                    assert_ne!(ptrs[i], ptrs[j]);
                }
            }

            for p in ptrs {
                os_free(p, PAGE_SIZE * 4);
            }
        }
    }
}
