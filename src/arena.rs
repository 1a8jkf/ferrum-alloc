//! Arena: a contiguous region of memory obtained from the OS.
//!
//! An arena is the fundamental unit of memory in ferrum-alloc. Each arena
//! is a large, page-aligned block (default 2 MB) obtained via `os_alloc`.
//!
//! In Phase 1, arenas use a simple bump allocator — the fastest possible
//! allocation strategy (single pointer increment). Individual `free` is not
//! supported in this phase; that comes with slab allocation in Phase 2.
//!
//! ## Layout
//!
//! ```text
//! ┌─────────────────┬─────────────────┬───────────┬───────────┐
//! │  Page 0         │  Pages 1-3      │  Page 4   │  Page 5   │
//! │  ArenaMeta      │  PageMeta[512]  │  Data     │  Data     │
//! └─────────────────┴─────────────────┴───────────┴───────────┘
//! ^base                                ^data_start            ^limit
//! ```

use core::ptr;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::platform::{self, DEFAULT_ARENA_SIZE, align_up, PAGE_SIZE};
use crate::slab::PageMeta;

/// Unique arena identifier, used to determine which arena owns a block
/// (critical for cross-thread free routing in Phase 4).
static NEXT_ARENA_ID: AtomicU32 = AtomicU32::new(1);

fn next_arena_id() -> u32 {
    NEXT_ARENA_ID.fetch_add(1, Ordering::Relaxed)
}

/// Number of pages per 2MB arena
pub const PAGES_PER_ARENA: usize = DEFAULT_ARENA_SIZE / PAGE_SIZE;

/// Metadata stored at the beginning of each arena (Page 0).
#[repr(C)]
pub struct ArenaMeta {
    pub id: u32,
    pub total_size: u32,
    pub next_free_page: u16, // index of the next free page for bump allocation
    pub alloc_count: u32,
    _pad: [u32; 3],
}

/// Metadata array lives starting at this offset (Page 1).
/// Since PageMeta is 24 bytes on 64-bit/// Number of pages reserved for metadata at the start of the arena.
/// ArenaMeta header is 16 bytes. PageMeta array is 512 * 32 = 16384 bytes.
/// Total: 16400 bytes. We need ceil(16400 / 4096) = 5 pages.
pub const META_PAGES: usize = 5;
pub const DATA_START_OFFSET: usize = META_PAGES * PAGE_SIZE;

/// Maximum alignment guaranteed by the allocator (16 bytes, matching
/// the requirement of most SIMD types and `max_align_t` on most platforms).
pub const MAX_ALIGN: usize = 16;

/// Size of the arena metadata header, aligned up to MAX_ALIGN.
pub const ARENA_META_SIZE: usize = align_up_const(core::mem::size_of::<ArenaMeta>(), MAX_ALIGN);

/// Const version of align_up for use in const contexts.
const fn align_up_const(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

/// A bump-allocating arena.
///
/// Wraps an OS-allocated region and provides fast, sequential allocation.
/// In this phase, individual deallocation is not supported — only full
/// arena reset or destruction.
pub struct Arena {
    /// Base pointer of the OS allocation
    base: *mut u8,
    /// Total size of the OS allocation
    total_size: usize,
}

impl Arena {
    /// Create a new arena of the default size (2MB).
    /// The `_size` parameter is ignored in Phase 2+ as arenas must have fixed size.
    /// Returns `None` if the OS allocation fails.
    pub fn new(_size: usize) -> Option<Self> {
        let total_size = DEFAULT_ARENA_SIZE;

        let base = unsafe { platform::os_alloc(total_size) };
        if base.is_null() {
            return None;
        }

        // Initialize metadata at the base
        let meta = base as *mut ArenaMeta;

        unsafe {
            ptr::write(meta, ArenaMeta {
                id: next_arena_id(),
                total_size: total_size as u32,
                next_free_page: META_PAGES as u16,
                alloc_count: 0,
                _pad: [0; 3],
            });

            // Initialize PageMeta array in Page 1
            let page_meta_array = base.add(PAGE_SIZE) as *mut PageMeta;
            for i in 0..PAGES_PER_ARENA {
                ptr::write(page_meta_array.add(i), PageMeta::empty());
            }
        }

        Some(Arena { base, total_size })
    }

    /// Allocates a single page (4KB) from this arena.
    /// Returns a tuple of (pointer to PageMeta, pointer to data page).
    pub fn alloc_page(&mut self) -> Option<(*mut PageMeta, *mut u8)> {
        let meta = self.meta_mut();
        let page_idx = meta.next_free_page as usize;
        
        if page_idx >= PAGES_PER_ARENA {
            return None; // Arena full
        }

        meta.next_free_page += 1;
        meta.alloc_count += 1;

        let page_meta = self.meta_ptr(page_idx);
        let page_data = self.page_addr(page_idx);

        Some((page_meta, page_data))
    }

    /// Allocates `size` bytes. Fallback for sizes > 1024 or before Slab is fully integrated.
    /// This now allocates in page increments.
    pub fn alloc(&mut self, size: usize, _align: usize) -> *mut u8 {
        if size == 0 {
            return ptr::null_mut();
        }

        // Simplistic page-aligned fallback allocator for Phase 2 integration
        let pages_needed = (size + PAGE_SIZE - 1) / PAGE_SIZE;
        let meta = self.meta_mut();
        let start_page = meta.next_free_page as usize;
        let end_page = start_page + pages_needed;

        if end_page > PAGES_PER_ARENA {
            return ptr::null_mut();
        }

        meta.next_free_page = end_page as u16;
        meta.alloc_count += 1;

        // Mark them as large in PageMeta
        for i in start_page..end_page {
            unsafe {
                let pm = &mut *self.meta_ptr(i);
                pm.size_class = 0xFF; // Large/Direct allocation
            }
        }

        self.page_addr(start_page)
    }

    /// Reset the arena's bump pointer.
    pub unsafe fn reset(&mut self) {
        let meta = self.meta_mut();
        meta.next_free_page = META_PAGES as u16;
        meta.alloc_count = 0;
    }

    /// Convert a pointer to a page index in this arena.
    /// Returns 0 if invalid or outside data bounds.
    #[inline]
    pub fn ptr_to_page_index(&self, ptr: *const u8) -> usize {
        let addr = ptr as usize;
        let base = self.base as usize;
        if addr < base + DATA_START_OFFSET || addr >= base + self.total_size {
            return 0;
        }
        (addr - base) / PAGE_SIZE
    }

    /// Get the PageMeta pointer for a given page index.
    #[inline]
    pub fn meta_ptr(&self, page_index: usize) -> *mut PageMeta {
        debug_assert!(page_index < PAGES_PER_ARENA);
        unsafe {
            let meta_array = self.base.add(PAGE_SIZE) as *mut PageMeta;
            meta_array.add(page_index)
        }
    }

    /// Get the PageMeta index from a PageMeta pointer.
    #[inline]
    pub fn meta_index(&self, meta_ptr: *const PageMeta) -> usize {
        let meta_array = unsafe { self.base.add(PAGE_SIZE) } as usize;
        let ptr_val = meta_ptr as usize;
        (ptr_val - meta_array) / core::mem::size_of::<PageMeta>()
    }

    /// Get the data address for a given page index.
    #[inline]
    pub fn page_addr(&self, page_index: usize) -> *mut u8 {
        debug_assert!(page_index < PAGES_PER_ARENA);
        unsafe { self.base.add(page_index * PAGE_SIZE) }
    }

    pub fn bytes_used(&self) -> usize {
        let meta = self.meta();
        (meta.next_free_page as usize - META_PAGES) * PAGE_SIZE
    }

    pub fn bytes_remaining(&self) -> usize {
        let meta = self.meta();
        (PAGES_PER_ARENA - meta.next_free_page as usize) * PAGE_SIZE
    }

    pub fn capacity(&self) -> usize {
        (PAGES_PER_ARENA - META_PAGES) * PAGE_SIZE
    }

    /// Returns the arena's unique ID.
    pub fn id(&self) -> u32 {
        self.meta().id
    }

    /// Returns the base pointer of the arena (for address-range checks).
    pub fn base_ptr(&self) -> *mut u8 {
        self.base
    }

    /// Check if a pointer belongs to this arena.
    pub fn contains(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let base = self.base as usize;
        addr >= base && addr < base + self.total_size
    }

    /// Returns the number of live allocations.
    pub fn alloc_count(&self) -> u32 {
        self.meta().alloc_count
    }

    #[inline]
    fn meta(&self) -> &ArenaMeta {
        unsafe { &*(self.base as *const ArenaMeta) }
    }

    #[inline]
    fn meta_mut(&mut self) -> &mut ArenaMeta {
        unsafe { &mut *(self.base as *mut ArenaMeta) }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            platform::os_free(self.base, self.total_size);
        }
    }
}

// Arena is Send — it owns its memory and can be transferred between threads.
// It is NOT Sync because bump allocation requires &mut self.
unsafe impl Send for Arena {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = Arena::new(0).expect("Failed to create arena");
        assert!(arena.id() > 0);
        assert!(arena.capacity() > 0);
        assert_eq!(arena.bytes_used(), 0);
        assert!(arena.bytes_remaining() > 0);
    }

    #[test]
    fn test_arena_alloc_basic() {
        let mut arena = Arena::new(0).expect("Failed to create arena");

        let p1 = arena.alloc(64, 8);
        assert!(!p1.is_null());
        assert!(arena.contains(p1));
        assert_eq!(arena.alloc_count(), 1);

        let p2 = arena.alloc(128, 16);
        assert!(!p2.is_null());
        assert!(arena.contains(p2));
        assert_ne!(p1, p2);
        assert_eq!(arena.alloc_count(), 2);

        // Verify alignment is page aligned since it allocates pages now
        assert_eq!(p1 as usize % PAGE_SIZE, 0);
        assert_eq!(p2 as usize % PAGE_SIZE, 0);
    }

    #[test]
    fn test_arena_alloc_write_read() {
        let mut arena = Arena::new(0).expect("Failed to create arena");

        let p = arena.alloc(256, 8);
        assert!(!p.is_null());

        // Write pattern
        unsafe {
            for i in 0..=255u8 {
                p.add(i as usize).write(i);
            }
            // Read back
            for i in 0..=255u8 {
                assert_eq!(p.add(i as usize).read(), i);
            }
        }
    }

    #[test]
    fn test_arena_exhaustion() {
        // Create a tiny arena (one page)
        let mut arena = Arena::new(platform::PAGE_SIZE).expect("Failed to create arena");

        // Allocate until it fails
        let capacity = arena.capacity();
        let p = arena.alloc(capacity, 1);
        assert!(!p.is_null());

        // This should fail — arena is full
        let p2 = arena.alloc(1, 1);
        assert!(p2.is_null());
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new(0).expect("Failed to create arena");

        let _ = arena.alloc(1024, 8);
        let _ = arena.alloc(2048, 8);
        assert!(arena.bytes_used() > 0);
        assert_eq!(arena.alloc_count(), 2);

        unsafe { arena.reset() };
        assert_eq!(arena.bytes_used(), 0);
        assert_eq!(arena.alloc_count(), 0);

        // Should be able to allocate again after reset
        let p = arena.alloc(64, 8);
        assert!(!p.is_null());
    }

    #[test]
    fn test_arena_alignment_various() {
        let mut arena = Arena::new(0).expect("Failed to create arena");

        for &align in &[1, 2, 4, 8, 16] {
            let p = arena.alloc(32, align);
            assert!(!p.is_null());
            // Since alloc now rounds up to PAGE_SIZE, it is always PAGE_SIZE aligned
            assert_eq!(
                p as usize % platform::PAGE_SIZE, 0,
                "Pointer {:p} not aligned to {}", p, platform::PAGE_SIZE
            );
        }
    }

    #[test]
    fn test_arena_zero_size_alloc() {
        let mut arena = Arena::new(0).expect("Failed to create arena");
        let p = arena.alloc(0, 8);
        assert!(p.is_null(), "Zero-size alloc should return null");
    }

    #[test]
    fn test_arena_unique_ids() {
        let a1 = Arena::new(platform::PAGE_SIZE).expect("Failed");
        let a2 = Arena::new(platform::PAGE_SIZE).expect("Failed");
        let a3 = Arena::new(platform::PAGE_SIZE).expect("Failed");
        assert_ne!(a1.id(), a2.id());
        assert_ne!(a2.id(), a3.id());
    }

    #[test]
    fn test_arena_contains() {
        let mut arena = Arena::new(platform::PAGE_SIZE * 2).expect("Failed");
        let p = arena.alloc(64, 8);
        assert!(arena.contains(p));

        // A stack pointer should not be contained
        let stack_var: u8 = 0;
        assert!(!arena.contains(&stack_var as *const u8));
    }
}
