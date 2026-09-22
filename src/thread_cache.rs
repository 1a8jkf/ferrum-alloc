use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicU16, AtomicUsize, Ordering};

use crate::arena::Arena;
use crate::platform;
use crate::slab::SlabAllocator;

/// Global registry of ThreadCaches.
pub const MAX_THREADS: usize = 128;
static NEXT_THREAD_ID: AtomicU16 = AtomicU16::new(0);

// We store pointers to the ThreadCache in a global array for cross-thread frees.
// Array of AtomicPtr<ThreadCache>
static mut CACHE_REGISTRY: [AtomicPtr<ThreadCache>; MAX_THREADS] = [
    const { AtomicPtr::new(core::ptr::null_mut()) };
    MAX_THREADS
];

static TLS_KEY: AtomicUsize = AtomicUsize::new(usize::MAX);

/// A freed block overlaid as a lock-free stack node for cross-thread frees.
#[repr(C)]
pub struct RemoteFreeNode {
    pub next: *mut RemoteFreeNode,
}

/// Per-thread allocation cache.
pub struct ThreadCache {
    /// Unique ID (0..MAX_THREADS-1)
    pub id: u16,
    /// Thread-local slab allocator (NO lock needed!)
    pub slab: SlabAllocator,
    /// Treiber stack for cross-thread frees
    pub remote_frees: AtomicPtr<RemoteFreeNode>,
}

impl ThreadCache {
    /// Get the current thread's cache, or create one if it doesn't exist.
    pub fn get_or_create() -> Option<&'static mut ThreadCache> {
        let key = TLS_KEY.load(Ordering::Acquire);
        if key == usize::MAX {
            // Very first time any thread calls this, init the TLS key.
            // Note: in a real robust impl we'd use an atomic init pattern like Once,
            // but for this phase a simple spin-compare-exchange is fine since we
            // control ensure_init() upstream.
            return None; // Should be initialized via init_tls() first
        }

        let tls_ptr = platform::tls_get(key as u32);
        if !tls_ptr.is_null() {
            return Some(unsafe { &mut *(tls_ptr as *mut ThreadCache) });
        }

        // Need to create one.
        let id = NEXT_THREAD_ID.fetch_add(1, Ordering::Relaxed);
        if (id as usize) >= MAX_THREADS {
            return None; // Too many threads
        }

        // Allocate a dedicated page for the ThreadCache itself directly from OS
        // to avoid circular dependencies (allocating the allocator).
        let cache_ptr = unsafe { platform::os_alloc(platform::PAGE_SIZE) as *mut ThreadCache };
        if cache_ptr.is_null() {
            return None;
        }

        unsafe {
            ptr::write(cache_ptr, ThreadCache {
                id,
                slab: SlabAllocator::new(),
                remote_frees: AtomicPtr::new(ptr::null_mut()),
            });

            // Store in global registry
            CACHE_REGISTRY[id as usize].store(cache_ptr, Ordering::Release);

            // Store in TLS
            platform::tls_set(key as u32, cache_ptr as *mut u8);

            Some(&mut *cache_ptr)
        }
    }

    /// Process any pending cross-thread frees.
    pub fn drain_remote_frees(&mut self, arena: &mut Arena) {
        // Grab the whole stack atomically, swapping with null
        let mut head = self.remote_frees.swap(ptr::null_mut(), Ordering::Acquire);
        
        // Free them all locally
        while !head.is_null() {
            let next = unsafe { (*head).next };
            self.slab.free(head as *mut u8, arena);
            head = next;
        }
    }
}

/// Initialize the global TLS key. Called once during global init.
pub fn init_tls() {
    let key = platform::tls_create();
    TLS_KEY.store(key as usize, Ordering::Release);
}

/// Push a freed block to another thread's remote_frees stack.
pub fn remote_free_push(target_id: u16, ptr: *mut u8) {
    if (target_id as usize) >= MAX_THREADS {
        return; // Invalid ID
    }

    let cache_ptr = unsafe { CACHE_REGISTRY[target_id as usize].load(Ordering::Acquire) };
    if cache_ptr.is_null() {
        // Thread cache was destroyed or invalid, fallback/leak for now.
        return;
    }

    let node = ptr as *mut RemoteFreeNode;
    let remote_stack = unsafe { &(*cache_ptr).remote_frees };

    // Lock-free Treiber stack push
    let mut head = remote_stack.load(Ordering::Relaxed);
    loop {
        unsafe { (*node).next = head };
        match remote_stack.compare_exchange_weak(
            head,
            node,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(new_head) => head = new_head, // Try again
        }
    }
}
