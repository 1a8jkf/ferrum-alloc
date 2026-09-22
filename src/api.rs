//! C-compatible API surface.
//!
//! These functions are the public interface of ferrum-alloc, matching the
//! standard C allocator API (`malloc`, `free`, `calloc`, `realloc`).
//!
//! In Phase 1, this delegates to a simple bump allocator. The API is stable;
//! the implementation behind it will evolve through phases.
//!
//! ## Usage
//!
//! - **Linux**: `LD_PRELOAD=libferrum_alloc.so ./your_app`
//! - **Windows**: Link against `ferrum_alloc.dll` or use detours
//! - **Rust**: Use as a `GlobalAlloc` implementation

use core::ptr;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::arena::{Arena, MAX_ALIGN};
use crate::platform::PAGE_SIZE;
use crate::size_class::{size_to_class, class_to_size, SLAB_MAX_SIZE};
use crate::slab::SlabAllocator;
use crate::tlsf::TlsfAllocator;
use crate::thread_cache::{self, ThreadCache};

// ============================================================================
// Global state — Phase 1: single arena with spin lock
//
// This will be replaced with thread-local arenas in Phase 4. For now, a
// single global arena behind a spin lock is correct (if not performant).
// ============================================================================

/// Spin lock for global arena access (temporary, replaced in Phase 4)
static ARENA_LOCK: AtomicBool = AtomicBool::new(false);

/// Whether the global arena has been initialized
static ARENA_INIT: AtomicBool = AtomicBool::new(false);

/// Storage for the global arena
static mut GLOBAL_ARENA_STORAGE: core::mem::MaybeUninit<Arena> = core::mem::MaybeUninit::uninit();

/// Raw pointer to the global arena (initialized once)
static mut GLOBAL_ARENA: *mut Arena = ptr::null_mut();

/// Storage for the global slab allocator
static mut GLOBAL_SLAB_STORAGE: core::mem::MaybeUninit<SlabAllocator> = core::mem::MaybeUninit::uninit();

/// Raw pointer to the global slab allocator
static mut GLOBAL_SLAB: *mut SlabAllocator = ptr::null_mut();

/// Storage for the global TLSF allocator
static mut GLOBAL_TLSF_STORAGE: core::mem::MaybeUninit<TlsfAllocator> = core::mem::MaybeUninit::uninit();

/// Raw pointer to the global TLSF allocator
static mut GLOBAL_TLSF: *mut TlsfAllocator = ptr::null_mut();

#[inline]
fn spin_lock() {
    while ARENA_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
}

#[inline]
fn spin_unlock() {
    ARENA_LOCK.store(false, Ordering::Release);
}

/// Initialize the global arena if not already done.
///
/// Uses a simple double-checked locking pattern.
fn ensure_init() {
    if ARENA_INIT.load(Ordering::Acquire) {
        return;
    }
    spin_lock();
    // Double-check after acquiring lock
    if !ARENA_INIT.load(Ordering::Relaxed) {
        if let Some(arena) = Arena::new(0) {
            unsafe {
                thread_cache::init_tls(); // Initialize TLS key

                let storage_ptr = core::ptr::addr_of_mut!(GLOBAL_ARENA_STORAGE);
                (*storage_ptr).write(arena);
                GLOBAL_ARENA = (*storage_ptr).as_mut_ptr();

                let slab_ptr = core::ptr::addr_of_mut!(GLOBAL_SLAB_STORAGE);
                (*slab_ptr).write(SlabAllocator::new());
                GLOBAL_SLAB = (*slab_ptr).as_mut_ptr();

                // Allocate a 512 KB pool for the TLSF allocator
                let arena_ref = &mut *GLOBAL_ARENA;
                let tlsf_pool_size = 128 * PAGE_SIZE; // 512 KB
                let tlsf_pool = arena_ref.alloc(tlsf_pool_size, MAX_ALIGN);
                if !tlsf_pool.is_null() {
                    let mut tlsf = TlsfAllocator::new();
                    tlsf.add_pool(tlsf_pool, tlsf_pool_size);
                    let tp = core::ptr::addr_of_mut!(GLOBAL_TLSF_STORAGE);
                    (*tp).write(tlsf);
                    GLOBAL_TLSF = (*tp).as_mut_ptr();
                }
            }
            ARENA_INIT.store(true, Ordering::Release);
        }
    }
    spin_unlock();
}

// ============================================================================
// C API
// ============================================================================

/// Allocate `size` bytes of memory.
///
/// Returns a pointer to the allocated memory, aligned to at least 16 bytes.
/// Returns null if allocation fails.
///
/// Routing:
/// - size ≤ 1024: ThreadCache (Lock-Free fast path)
/// - size > 1024: Global TLSF (Spin lock slow path)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ferrum_malloc(size: usize) -> *mut u8 {
    if size == 0 {
        return ptr::null_mut();
    }

    ensure_init();

    if size >= crate::huge::HUGE_ALLOC_THRESHOLD {
        return crate::huge::huge_alloc(size);
    }

    if size <= SLAB_MAX_SIZE {
        if let Some(class) = size_to_class(size) {
            if let Some(tc) = ThreadCache::get_or_create() {
                // We have a ThreadCache! Fast path (no spin locks unless we need a new page)
                
                // First, process any pending remote frees (requires arena lock)
                // In a fully lock-free world, we'd have a local arena too, but for Phase 4
                // we still share the global arena for page allocations.
                let has_remote = !tc.remote_frees.load(Ordering::Relaxed).is_null();
                if has_remote {
                    spin_lock();
                    unsafe { tc.drain_remote_frees(&mut *GLOBAL_ARENA) };
                    spin_unlock();
                }

                // Try to allocate from the thread-local slab without any locks!
                // If it needs a new page, we temporarily lock.
                let ptr = {
                    // Peek if we need a new page
                    if tc.slab.alloc_needs_page(class) {
                        spin_lock();
                        let p = unsafe { tc.slab.alloc(class, &mut *GLOBAL_ARENA, tc.id) };
                        spin_unlock();
                        p
                    } else {
                        // Fully lock-free fast path!
                        tc.slab.alloc_no_arena(class)
                    }
                };

                if !ptr.is_null() {
                    return ptr;
                }
            }
        }
    }

    // Large allocation → TLSF (Slow path)
    spin_lock();
    let result = unsafe {
        let arena = &mut *GLOBAL_ARENA;

        if !GLOBAL_TLSF.is_null() {
            let tlsf = &mut *GLOBAL_TLSF;
            let ptr = tlsf.alloc(size);
            if !ptr.is_null() {
                spin_unlock();
                return ptr;
            }
        }

        // Ultimate fallback: arena bump allocator
        arena.alloc(size, MAX_ALIGN)
    };
    spin_unlock();

    result
}

/// Free memory previously allocated by `ferrum_malloc`.
///
/// Routes to ThreadCache (small) or TLSF (large). Cross-thread small frees
/// use a lock-free Treiber stack.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ferrum_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }

    ensure_init();

    // Fast path check: does it belong to our ThreadCache?
    if let Some(tc) = ThreadCache::get_or_create() {
        let arena = unsafe { &*GLOBAL_ARENA };
        if arena.contains(ptr) {
            let page_idx = arena.ptr_to_page_index(ptr);
            if page_idx > 0 {
                let meta = arena.meta_ptr(page_idx);
                let size_class = unsafe { (*meta).size_class };
                let owner_id = unsafe { (*meta).owner_id };

                if size_class != 0xFF {
                    // Small allocation
                    if owner_id == tc.id {
                        // FAST PATH: Local free, no locks!
                        // (We don't actually need the arena to push to the local free list,
                        // but the signature expects it in case the page goes empty. For Phase 4,
                        // we ignore returning empty pages to avoid locking here).
                        tc.slab.free_no_arena(ptr, page_idx, meta);
                        return;
                    } else if owner_id != 0xFFFF {
                        // CROSS-THREAD FREE: push to owner's Treiber stack (Lock-free!)
                        thread_cache::remote_free_push(owner_id, ptr);
                        return;
                    }
                }
            }
        }
    }

    // Slow path (Large allocation, bump fallback, or global slab fallback)
    spin_lock();
    unsafe {
        let arena = &mut *GLOBAL_ARENA;
        
        if arena.contains(ptr) {
            let page_idx = arena.ptr_to_page_index(ptr);
            if page_idx > 0 {
                let meta = arena.meta_ptr(page_idx);
                if (*meta).size_class != 0xFF {
                    // Fallback to global slab if not owned by a ThreadCache
                    let slab = &mut *GLOBAL_SLAB;
                    slab.free(ptr, arena);
                } else if !GLOBAL_TLSF.is_null() && (&*GLOBAL_TLSF).contains(ptr) {
                    (&mut *GLOBAL_TLSF).free(ptr);
                }
            }
        } else {
            // Pointer is completely outside the arena.
            // In Phase 6, this means it's a huge allocation mapping directly to OS memory.
            crate::huge::huge_free(ptr);
        }
    }
    spin_unlock();
}

// ============================================================================
// Standard POSIX C ABI
// ============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc(size: usize) -> *mut core::ffi::c_void {
    ferrum_malloc(size) as *mut core::ffi::c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free(ptr: *mut core::ffi::c_void) {
    ferrum_free(ptr as *mut u8)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn calloc(nmemb: usize, size: usize) -> *mut core::ffi::c_void {
    let total_size = nmemb.checked_mul(size).unwrap_or(0);
    if total_size == 0 {
        return core::ptr::null_mut();
    }
    let ptr = ferrum_malloc(total_size);
    if !ptr.is_null() {
        core::ptr::write_bytes(ptr, 0, total_size);
    }
    ptr as *mut core::ffi::c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn realloc(ptr: *mut core::ffi::c_void, size: usize) -> *mut core::ffi::c_void {
    ferrum_realloc(ptr as *mut u8, size) as *mut core::ffi::c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn posix_memalign(memptr: *mut *mut core::ffi::c_void, alignment: usize, size: usize) -> i32 {
    // We guarantee 16-byte alignment natively. Huge allocations are page-aligned (4096).
    // So if alignment is <= 16, we can just use normal malloc.
    // If alignment > 16, this requires a specific aligned-alloc strategy. 
    // For LD_PRELOAD simplicity right now, if alignment <= 16, we fulfill it directly.
    if alignment % core::mem::size_of::<*mut core::ffi::c_void>() != 0 || !alignment.is_power_of_two() {
        return 22; // EINVAL
    }
    
    let ptr = if alignment <= 16 {
        ferrum_malloc(size)
    } else if alignment <= 4096 {
        // Fallback to huge allocation since it guarantees 4096 alignment
        crate::huge::huge_alloc(size)
    } else {
        // For alignment > 4096, we don't officially support it in this minimal stub
        core::ptr::null_mut()
    };
    
    if ptr.is_null() {
        return 12; // ENOMEM
    }
    
    *memptr = ptr as *mut core::ffi::c_void;
    0
}

/// Allocate `count * size` bytes, initialized to zero.
///
/// Returns null if the multiplication overflows or allocation fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ferrum_calloc(count: usize, size: usize) -> *mut u8 {
    let total = match count.checked_mul(size) {
        Some(t) => t,
        None => return ptr::null_mut(), // overflow
    };

    let ptr = unsafe { ferrum_malloc(total) };
    if !ptr.is_null() {
        // Memory from OS (mmap/VirtualAlloc) is already zeroed on first use,
        // but we zero explicitly for correctness after arena recycling.
        unsafe { ptr::write_bytes(ptr, 0, total) };
    }
    ptr
}

/// Reallocate `ptr` to `new_size` bytes.
///
/// - If `ptr` is null, equivalent to `ferrum_malloc(new_size)`.
/// - If `new_size` is 0, equivalent to `ferrum_free(ptr)`.
///
/// Phase 3: properly determines old size via page metadata or TLSF header,
/// copies `min(old_size, new_size)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ferrum_realloc(ptr: *mut u8, new_size: usize) -> *mut u8 {
    if ptr.is_null() {
        return unsafe { ferrum_malloc(new_size) };
    }

    if new_size == 0 {
        unsafe { ferrum_free(ptr) };
        return ptr::null_mut();
    }

    // Determine old usable size (each call acquires/releases lock independently)
    let old_size = unsafe { ferrum_usable_size(ptr as *const u8) };

    let new_ptr = unsafe { ferrum_malloc(new_size) };
    if !new_ptr.is_null() {
        let copy_size = if old_size > 0 { old_size.min(new_size) } else { new_size };
        unsafe {
            ptr::copy_nonoverlapping(ptr, new_ptr, copy_size);
            ferrum_free(ptr);
        }
    }
    new_ptr
}

/// Returns the usable size of the allocation at `ptr`.
///
/// For slab allocations, returns the slot size of the size class.
/// For TLSF allocations, returns the block payload size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ferrum_usable_size(ptr: *const u8) -> usize {
    if ptr.is_null() {
        return 0;
    }

    ensure_init();

    spin_lock();
    let size = unsafe {
        let arena = &*GLOBAL_ARENA;
        if arena.contains(ptr) {
            let page_idx = arena.ptr_to_page_index(ptr);
            if page_idx > 0 {
                let meta = arena.meta_ptr(page_idx);
                if (*meta).size_class != 0xFF {
                    // Slab allocation
                    class_to_size((*meta).size_class)
                } else if !GLOBAL_TLSF.is_null() && (&*GLOBAL_TLSF).contains(ptr) {
                    // TLSF allocation
                    TlsfAllocator::usable_size(ptr)
                } else {
                    0 // bump-allocated, size unknown
                }
            } else {
                0
            }
        } else {
            // Huge allocation
            crate::huge::huge_usable_size(ptr.cast_mut())
        }
    };
    spin_unlock();
    size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malloc_basic() {
        unsafe {
            let p = ferrum_malloc(64);
            assert!(!p.is_null());
            // Write and read back
            p.write(0xAA);
            assert_eq!(p.read(), 0xAA);
            ferrum_free(p);
        }
    }

    #[test]
    fn test_malloc_alignment() {
        unsafe {
            for _ in 0..10 {
                let p = ferrum_malloc(17); // odd size
                assert!(!p.is_null());
                assert_eq!(p as usize % MAX_ALIGN, 0, "Not aligned to MAX_ALIGN");
                ferrum_free(p);
            }
        }
    }

    #[test]
    fn test_malloc_zero() {
        unsafe {
            let p = ferrum_malloc(0);
            assert!(p.is_null());
        }
    }

    #[test]
    fn test_calloc_zeroed() {
        unsafe {
            let p = ferrum_calloc(10, 100); // 1000 bytes
            assert!(!p.is_null());
            // All bytes should be zero
            for i in 0..1000 {
                assert_eq!(p.add(i).read(), 0, "Byte {} not zero", i);
            }
            ferrum_free(p);
        }
    }

    #[test]
    fn test_calloc_overflow() {
        unsafe {
            let p = ferrum_calloc(usize::MAX, 2);
            assert!(p.is_null(), "Overflow should return null");
        }
    }

    #[test]
    fn test_realloc_from_null() {
        unsafe {
            // realloc(null, size) == malloc(size)
            let p = ferrum_realloc(ptr::null_mut(), 128);
            assert!(!p.is_null());
            ferrum_free(p);
        }
    }

    #[test]
    fn test_realloc_to_zero() {
        unsafe {
            let p = ferrum_malloc(64);
            assert!(!p.is_null());
            // realloc(ptr, 0) == free(ptr)
            let q = ferrum_realloc(p, 0);
            assert!(q.is_null());
        }
    }

    #[test]
    fn test_multiple_allocs_distinct() {
        unsafe {
            let mut ptrs = [ptr::null_mut(); 100];
            for i in 0..100 {
                ptrs[i] = ferrum_malloc(32);
                assert!(!ptrs[i].is_null());
            }
            // All should be distinct
            for i in 0..100 {
                for j in (i + 1)..100 {
                    assert_ne!(ptrs[i], ptrs[j]);
                }
            }
            for p in ptrs {
                ferrum_free(p);
            }
        }
    }

    #[test]
    fn test_large_alloc() {
        unsafe {
            // Allocate close to arena capacity
            let p = ferrum_malloc(1024 * 1024); // 1 MB
            assert!(!p.is_null());
            // Write to first and last byte
            p.write(0x11);
            p.add(1024 * 1024 - 1).write(0x22);
            assert_eq!(p.read(), 0x11);
            assert_eq!(p.add(1024 * 1024 - 1).read(), 0x22);
            ferrum_free(p);
        }
    }
}
