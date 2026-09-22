//! TLSF (Two-Level Segregated Fit) allocator for medium/large allocations.
//!
//! Manages allocations from ~32 bytes to ~2 MB with O(1) worst-case time
//! for malloc, free, and coalescing. Uses bitmap-indexed segregated free
//! lists with boundary tags.
//!
//! ## Algorithm
//!
//! Free blocks are organized in a two-level matrix:
//! - **First Level (FL)**: log₂ buckets (powers of 2)
//! - **Second Level (SL)**: 16 linear subdivisions within each FL
//!
//! Finding a suitable free block uses bitmap CTZ (Count Trailing Zeros)
//! for O(1) lookup. Coalescing uses boundary tags for O(1) merge.

// Allocator internals: many functions touch raw pointers extensively.
#[allow(unsafe_op_in_unsafe_fn)]

use core::ptr;

// ============================================================================
// Constants
// ============================================================================

/// Block alignment (matches MAX_ALIGN)
const ALIGN: usize = 16;

/// Size of BlockHeader (16 bytes on 64-bit: two `usize` fields)
const HEADER_SIZE: usize = core::mem::size_of::<BlockHeader>();

/// Minimum payload a free block must hold (two pointers for free list links)
const MIN_PAYLOAD: usize = 2 * core::mem::size_of::<usize>(); // 16 on 64-bit

/// FL index of the smallest managed bin: log₂(16) = 4
const FL_SHIFT: usize = 4;
/// Largest FL index: log₂(2 MB) = 21
const FL_MAX: usize = 21;
/// Number of First-Level bins: 18  (covers 2⁴ through 2²¹)
const FL_COUNT: usize = FL_MAX - FL_SHIFT + 1;
/// Number of Second-Level bins per FL
const SL_COUNT: usize = 16;
/// log₂(SL_COUNT)
const SL_LOG2: usize = 4;

/// Flag: block is free
const F_FREE: usize = 1;
/// Flag: previous physical block is free
const F_PREV_FREE: usize = 2;
/// Mask to extract payload size from `size_flags`
const SIZE_MASK: usize = !(F_FREE | F_PREV_FREE);

// ============================================================================
// Block Header (Boundary Tag)
// ============================================================================

/// Header placed before every block (free or allocated).
///
/// On 64-bit this is exactly 16 bytes, ensuring the payload that follows
/// is naturally 16-byte aligned when the header itself is aligned.
///
/// ```text
/// [ size_flags: usize | prev_phys_size: usize | ...payload... ]
///   bits 0-1 = flags    (size of previous block for backward coalescing)
///   bits 2+  = payload size
/// ```
/// Free-list pointers, overlaid on the payload area of free blocks.
#[repr(C)]
pub struct FreeLinks {
    pub prev: *mut BlockHeader,
    pub next: *mut BlockHeader,
}

#[repr(C, align(16))]
pub struct BlockHeader {
    pub prev_phys_size: usize,
    pub size_and_flags: usize,
    // Links are stored in the payload area when the block is free.
    // We don't use an Option to avoid discriminant overhead and UB.
}

impl BlockHeader {
    #[inline]
    fn size(&self) -> usize {
        self.size_and_flags & SIZE_MASK
    }

    #[inline]
    fn set_size(&mut self, s: usize) {
        self.size_and_flags = s | (self.size_and_flags & !SIZE_MASK);
    }

    #[inline]
    fn is_free(&self) -> bool {
        self.size_and_flags & F_FREE != 0
    }

    #[inline]
    fn prev_is_free(&self) -> bool {
        self.size_and_flags & F_PREV_FREE != 0
    }

    #[inline]
    fn set_free(&mut self) {
        self.size_and_flags |= F_FREE;
    }

    #[inline]
    fn mark_used(&mut self) {
        self.size_and_flags &= !F_FREE;
    }

    #[inline]
    fn mark_prev_free(&mut self) {
        self.size_and_flags |= F_PREV_FREE;
    }

    #[inline]
    fn mark_prev_used(&mut self) {
        self.size_and_flags &= !F_PREV_FREE;
    }

    /// Pointer to the payload area (user data or FreeLinks).
    #[inline]
    fn payload(&self) -> *mut u8 {
        unsafe { (self as *const Self as *mut u8).add(HEADER_SIZE) }
    }

    /// Recover a `BlockHeader` pointer from a user payload pointer.
    #[inline]
    fn from_payload(p: *mut u8) -> *mut Self {
        unsafe { p.sub(HEADER_SIZE) as *mut Self }
    }

    /// Next physically adjacent block.
    #[inline]
    fn next_phys(&self) -> *mut BlockHeader {
        unsafe { self.payload().add(self.size()) as *mut BlockHeader }
    }

    /// Previous physically adjacent block.
    #[inline]
    unsafe fn prev_phys(&self) -> *mut BlockHeader {
        (self as *const Self as *mut u8)
            .sub(self.prev_phys_size + HEADER_SIZE) as *mut BlockHeader
    }

    #[inline]
    unsafe fn links(&self) -> &FreeLinks {
        &*(self.payload() as *const FreeLinks)
    }

    #[inline]
    unsafe fn links_mut(&mut self) -> &mut FreeLinks {
        &mut *(self.payload() as *mut FreeLinks)
    }
}

// ============================================================================
// TLSF Mapping Functions
// ============================================================================

/// Map a payload size to `(fl_index, sl_index)`.
///
/// `size` must be >= `MIN_PAYLOAD` and a multiple of `ALIGN`.
#[inline]
fn mapping(size: usize) -> (usize, usize) {
    debug_assert!(size >= MIN_PAYLOAD);
    let fl = (usize::BITS - 1 - size.leading_zeros()) as usize;
    let sl = (size >> fl.saturating_sub(SL_LOG2)) ^ SL_COUNT;
    (fl - FL_SHIFT, sl)
}

/// Map a payload size to `(fl_index, sl_index)` for **searching** — rounds up
/// to the next SL boundary so every block in the returned bin is guaranteed
/// to be ≥ `size`.
#[inline]
fn mapping_search(size: usize) -> (usize, usize) {
    let fl = (usize::BITS - 1 - size.leading_zeros()) as usize;
    let round = (1usize << fl.saturating_sub(SL_LOG2)) - 1;
    let size2 = size.saturating_add(round);
    let fl2 = (usize::BITS - 1 - size2.leading_zeros()) as usize;
    if fl2 > FL_MAX {
        return (FL_COUNT, 0); // too large, will be caught by find_suitable
    }
    let sl = (size2 >> fl2.saturating_sub(SL_LOG2)) ^ SL_COUNT;
    (fl2 - FL_SHIFT, sl)
}

/// Align `size` up to `ALIGN`.
#[inline]
fn align_up_size(size: usize) -> usize {
    (size + ALIGN - 1) & !(ALIGN - 1)
}

// ============================================================================
// TLSF Allocator
// ============================================================================

/// Two-Level Segregated Fit allocator.
///
/// Manages a contiguous memory pool with O(1) worst-case alloc, free, and
/// coalescing.
pub struct TlsfAllocator {
    /// First-level bitmap: bit *i* set ⇒ FL bin *i* has ≥1 free block.
    fl_bitmap: u32,
    /// Second-level bitmaps (one per FL bin).
    sl_bitmaps: [u32; FL_COUNT],
    /// Heads of the 18 × 16 = 288 free lists.
    lists: [[*mut BlockHeader; SL_COUNT]; FL_COUNT],
    /// Pool boundaries (for `contains` checks).
    pool_start: *mut u8,
    pool_end: *mut u8,
}

impl TlsfAllocator {
    pub const fn new() -> Self {
        TlsfAllocator {
            fl_bitmap: 0,
            sl_bitmaps: [0; FL_COUNT],
            lists: [[ptr::null_mut(); SL_COUNT]; FL_COUNT],
            pool_start: ptr::null_mut(),
            pool_end: ptr::null_mut(),
        }
    }

    /// Check if a pointer falls within this TLSF pool.
    #[inline]
    pub fn contains(&self, ptr: *const u8) -> bool {
        let a = ptr as usize;
        a >= self.pool_start as usize && a < self.pool_end as usize
    }

    /// Add a contiguous memory pool for the TLSF to manage.
    ///
    /// `base` must be `ALIGN`-aligned. Creates one large free block spanning
    /// the pool plus a zero-size sentinel at the end.
    ///
    /// # Safety
    ///
    /// `base` must point to a valid, writable region of `size` bytes.
    pub unsafe fn add_pool(&mut self, base: *mut u8, size: usize) {
        let min_pool = 2 * HEADER_SIZE + MIN_PAYLOAD;
        if size < min_pool {
            return;
        }

        self.pool_start = base;
        self.pool_end = base.add(size);

        // Payload for the initial free block (aligned down)
        let payload_size = (size - 2 * HEADER_SIZE) & !(ALIGN - 1);

        // --- Main free block header ---
        let block = base as *mut BlockHeader;
        unsafe {
            ptr::write(block, BlockHeader {
                size_and_flags: payload_size | F_FREE,
                prev_phys_size: 0,
            });
            let links = (*block).links_mut();
            links.prev = ptr::null_mut();
            links.next = ptr::null_mut();
        }

        // --- Sentinel (zero-size, used, marks pool boundary) ---
        let sentinel = unsafe { (*block).next_phys() };
        unsafe {
            ptr::write(sentinel, BlockHeader {
                size_and_flags: F_PREV_FREE, // size = 0, prev is free
                prev_phys_size: payload_size,
            });
        }

        // Insert the free block into the appropriate list
        self.insert_free(block);
    }

    /// Allocate `size` bytes. Returns null on failure.
    pub fn alloc(&mut self, size: usize) -> *mut u8 {
        if size == 0 {
            return ptr::null_mut();
        }

        let adjusted = align_up_size(size).max(MIN_PAYLOAD);
        let block = self.find_suitable(adjusted);
        if block.is_null() {
            return ptr::null_mut();
        }

        unsafe { self.prepare_used(block, adjusted) }
    }

    /// Free a previously allocated block, coalescing with neighbors.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by `alloc` and not yet freed.
    pub unsafe fn free(&mut self, ptr: *mut u8) {
        if ptr.is_null() {
            return;
        }

        let mut block = BlockHeader::from_payload(ptr);
        debug_assert!(!(*block).is_free(), "double free detected");

        unsafe { (*block).set_free() };

        // Coalesce with physically adjacent blocks
        block = self.merge_next(block);
        block = self.merge_prev(block);

        // Tell the next physical block that *we* are free
        unsafe {
            let next = (*block).next_phys();
            (*next).mark_prev_free();
            (*next).prev_phys_size = (*block).size();
        }

        self.insert_free(block);
    }

    /// Returns the usable (payload) size of the block at `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a live allocation from this TLSF.
    pub unsafe fn usable_size(ptr: *const u8) -> usize {
        if ptr.is_null() {
            return 0;
        }
        let block = BlockHeader::from_payload(ptr as *mut u8);
        unsafe { (*block).size() }
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Find (and remove from free list) a block ≥ `size`.
    fn find_suitable(&mut self, size: usize) -> *mut BlockHeader {
        let (fl, sl) = mapping_search(size);
        if fl >= FL_COUNT {
            return ptr::null_mut();
        }

        // Try the current FL level, starting from the computed SL
        let sl_map = self.sl_bitmaps[fl] & (!0u32 << sl);

        let (fl, sl) = if sl_map != 0 {
            // Found a non-empty list in the same FL
            (fl, sl_map.trailing_zeros() as usize)
        } else {
            // Move to the next higher FL with any free blocks
            let fl_map = self.fl_bitmap & (!0u32 << (fl as u32 + 1));
            if fl_map == 0 {
                return ptr::null_mut(); // OOM
            }
            let fl = fl_map.trailing_zeros() as usize;
            let sl = self.sl_bitmaps[fl].trailing_zeros() as usize;
            (fl, sl)
        };

        let block = self.lists[fl][sl];
        debug_assert!(!block.is_null());

        // Remove from free list before returning
        // SAFETY: block is a valid free-list node we just found via bitmaps.
        unsafe { self.remove_free(block); }
        block
    }

    /// Mark a block as used, splitting off any excess into a new free block.
    unsafe fn prepare_used(
        &mut self,
        block: *mut BlockHeader,
        size: usize,
    ) -> *mut u8 {
        let block_size = (*block).size();
        debug_assert!(block_size >= size);

        let remaining = block_size - size;

        if remaining >= HEADER_SIZE + MIN_PAYLOAD {
            // Split: shrink this block, create a new free remainder
            (*block).set_size(size);

            let split = (*block).next_phys();
            let split_payload = remaining - HEADER_SIZE;

            unsafe {
                ptr::write(split, BlockHeader {
                    size_and_flags: split_payload | F_FREE,
                    prev_phys_size: size,
                });
                (*split).links_mut().prev = ptr::null_mut();
                (*split).links_mut().next = ptr::null_mut();

                // Update the block *after* the split
                let after_split = (*split).next_phys();
                (*after_split).prev_phys_size = split_payload;
                (*after_split).mark_prev_free();
            }

            self.insert_free(split);
        }

        // Mark this block as used
        (*block).mark_used();

        // Tell the next physical block that we are NOT free
        let next = (*block).next_phys();
        (*next).mark_prev_used();

        (*block).payload()
    }

    /// Insert a free block into the appropriate FL/SL list.
    unsafe fn insert_free(&mut self, block: *mut BlockHeader) {
        let size = (*block).size();
        let (fl, sl) = mapping(size);
        debug_assert!(fl < FL_COUNT, "FL {} out of range", fl);
        debug_assert!(sl < SL_COUNT, "SL {} out of range", sl);

        let head = self.lists[fl][sl];

        (*block).links_mut().prev = ptr::null_mut();
        (*block).links_mut().next = head;

        if !head.is_null() {
            (*head).links_mut().prev = block;
        }

        self.lists[fl][sl] = block;

        // Light up the bitmap bits
        self.fl_bitmap |= 1u32 << fl;
        self.sl_bitmaps[fl] |= 1u32 << sl;
    }

    /// Remove a free block from its FL/SL list.
    unsafe fn remove_free(&mut self, block: *mut BlockHeader) {
        let size = (*block).size();
        let (fl, sl) = mapping(size);

        let prev = (*block).links().prev;
        let next = (*block).links().next;

        if !prev.is_null() {
            (*prev).links_mut().next = next;
        } else {
            self.lists[fl][sl] = next; // was head
        }
        if !next.is_null() {
            (*next).links_mut().prev = prev;
        }

        // If list is now empty, clear bitmap bits
        if self.lists[fl][sl].is_null() {
            self.sl_bitmaps[fl] &= !(1u32 << sl);
            if self.sl_bitmaps[fl] == 0 {
                self.fl_bitmap &= !(1u32 << fl);
            }
        }
    }

    /// Merge `block` with its next physical neighbor if free.
    unsafe fn merge_next(&mut self, block: *mut BlockHeader) -> *mut BlockHeader {
        let next = (*block).next_phys();

        if (*next).is_free() {
            self.remove_free(next);
            let new_size = (*block).size() + HEADER_SIZE + (*next).size();
            (*block).set_size(new_size);

            // Update the block *after* next
            let after = (*block).next_phys();
            (*after).prev_phys_size = new_size;
        }

        block
    }

    /// Merge `block` with its previous physical neighbor if free.
    unsafe fn merge_prev(&mut self, block: *mut BlockHeader) -> *mut BlockHeader {
        if !(*block).prev_is_free() {
            return block;
        }

        let prev = (*block).prev_phys();
        debug_assert!((*prev).is_free());

        self.remove_free(prev);
        let new_size = (*prev).size() + HEADER_SIZE + (*block).size();
        (*prev).set_size(new_size);

        let after = (*prev).next_phys();
        (*after).prev_phys_size = new_size;

        prev
    }
}

// TlsfAllocator can be sent between threads (guarded by external lock)
unsafe impl Send for TlsfAllocator {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform;

    /// Helper: allocate a raw pool via the OS backend.
    fn make_pool(size: usize) -> *mut u8 {
        let p = unsafe { platform::os_alloc(size) };
        assert!(!p.is_null(), "os_alloc failed for {} bytes", size);
        p
    }

    // --- mapping tests ---

    #[test]
    fn test_mapping_1024() {
        // 1024 = 2^10  →  FL = 10 - 4 = 6, SL = 0
        let (fl, sl) = mapping(1024);
        assert_eq!(fl, 6);
        assert_eq!(sl, 0);
    }

    #[test]
    fn test_mapping_2048() {
        let (fl, sl) = mapping(2048);
        assert_eq!(fl, 7); // 11 - 4
        assert_eq!(sl, 0);
    }

    #[test]
    fn test_mapping_min_payload() {
        // 16 = 2^4  →  FL = 4 - 4 = 0, SL = 0
        let (fl, sl) = mapping(16);
        assert_eq!(fl, 0);
        assert_eq!(sl, 0);
    }

    #[test]
    fn test_mapping_1536() {
        // 1536 is halfway through [1024, 2048)
        // fl = 10, sl = (1536 >> 6) ^ 16 = 24 ^ 16 = 8
        let (fl, sl) = mapping(1536);
        assert_eq!(fl, 6); // 10 - 4
        assert_eq!(sl, 8);
    }

    // --- alloc / free ---

    #[test]
    fn test_alloc_basic() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        let p = tlsf.alloc(2048);
        assert!(!p.is_null());
        assert!(tlsf.contains(p));

        // Write and verify
        unsafe {
            ptr::write_bytes(p, 0xAB, 2048);
            assert_eq!(*p, 0xAB);
            assert_eq!(*p.add(2047), 0xAB);
        }

        unsafe {
            tlsf.free(p);
            platform::os_free(pool, pool_size);
        }
    }

    #[test]
    fn test_alloc_multiple() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        let a = tlsf.alloc(1024);
        let b = tlsf.alloc(2048);
        let c = tlsf.alloc(4096);

        assert!(!a.is_null());
        assert!(!b.is_null());
        assert!(!c.is_null());
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);

        unsafe {
            tlsf.free(a);
            tlsf.free(b);
            tlsf.free(c);
            platform::os_free(pool, pool_size);
        }
    }

    #[test]
    fn test_alloc_zero() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        let p = tlsf.alloc(0);
        assert!(p.is_null());

        unsafe { platform::os_free(pool, pool_size); }
    }

    #[test]
    fn test_alloc_too_large() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        // Ask for more than the pool can hold
        let p = tlsf.alloc(pool_size);
        assert!(p.is_null());

        unsafe { platform::os_free(pool, pool_size); }
    }

    // --- coalescing ---

    #[test]
    fn test_coalesce_forward() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        let a = tlsf.alloc(1024);
        let b = tlsf.alloc(1024);
        let _c = tlsf.alloc(1024);

        // Free b then a → a should merge forward into b's space
        unsafe { tlsf.free(b); }
        unsafe { tlsf.free(a); }

        // Now we should be able to allocate a 2048+ block
        let big = tlsf.alloc(2048);
        assert!(!big.is_null());

        unsafe {
            tlsf.free(big);
            tlsf.free(_c);
            platform::os_free(pool, pool_size);
        }
    }

    #[test]
    fn test_coalesce_three_blocks() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        let a = tlsf.alloc(1024);
        let b = tlsf.alloc(1024);
        let c = tlsf.alloc(1024);

        // Free middle, then ends → should coalesce into one block
        unsafe {
            tlsf.free(b); // b alone
            tlsf.free(a); // a merges forward with b
            tlsf.free(c); // a+b merges forward with c
        }

        // Should be able to allocate 3× 1024 + 2 headers worth of space
        let big = tlsf.alloc(3 * 1024);
        assert!(!big.is_null());

        unsafe {
            tlsf.free(big);
            platform::os_free(pool, pool_size);
        }
    }

    // --- splitting ---

    #[test]
    fn test_splitting() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        // Allocate a small block from the initial large free block
        let a = tlsf.alloc(1024);
        assert!(!a.is_null());

        // Should still be able to allocate more (the remainder was split off)
        let b = tlsf.alloc(1024);
        assert!(!b.is_null());
        assert_ne!(a, b);

        unsafe {
            tlsf.free(a);
            tlsf.free(b);
            platform::os_free(pool, pool_size);
        }
    }

    // --- reuse after free ---

    #[test]
    fn test_reuse_after_free() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        let a = tlsf.alloc(2048);
        assert!(!a.is_null());

        unsafe { tlsf.free(a); }

        // Should reuse the freed block
        let b = tlsf.alloc(2048);
        assert!(!b.is_null());
        // On a fresh TLSF the freed block is the best fit, likely same address
        assert_eq!(a, b);

        unsafe {
            tlsf.free(b);
            platform::os_free(pool, pool_size);
        }
    }

    // --- usable_size ---

    #[test]
    fn test_usable_size() {
        let pool_size = 65536;
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        let p = tlsf.alloc(1000);
        assert!(!p.is_null());

        let usable = unsafe { TlsfAllocator::usable_size(p) };
        // Usable size should be >= requested size (aligned up to 16)
        assert!(usable >= 1000, "usable {} < 1000", usable);
        assert_eq!(usable % ALIGN, 0);

        unsafe {
            tlsf.free(p);
            platform::os_free(pool, pool_size);
        }
    }

    // --- stress test ---

    #[test]
    fn test_alloc_free_cycle() {
        let pool_size = 131072; // 128 KB
        let pool = make_pool(pool_size);
        let mut tlsf = TlsfAllocator::new();
        unsafe { tlsf.add_pool(pool, pool_size); }

        // Allocate many blocks, free them all, repeat
        for _ in 0..3 {
            let mut ptrs = [ptr::null_mut(); 20];
            for i in 0..20 {
                ptrs[i] = tlsf.alloc(1024 + i * 128);
                assert!(!ptrs[i].is_null(), "alloc {} failed", i);
            }
            for p in ptrs.iter().rev() {
                unsafe { tlsf.free(*p); }
            }
        }

        unsafe { platform::os_free(pool, pool_size); }
    }
}
