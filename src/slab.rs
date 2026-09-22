//! Slab allocator for small size classes.
//!
//! Manages memory for allocations <= 1024 bytes. Pages are requested from
//! the underlying arena and subdivided into slots of a specific size class.

use core::ptr;

use crate::arena::Arena;
use crate::platform::PAGE_SIZE;
use crate::size_class::{NUM_SIZE_CLASSES, class_to_size, slots_per_page};

/// A pointer/index representation to form linked lists of free slots within a page.
///
/// Since our slots are small, we can embed a `u16` index inside the free slot itself
/// instead of a full 64-bit pointer. This saves space and works because a 4KB page
/// can have at most 4096/8 = 512 slots.
const NULL_SLOT: u16 = 0xFFFF;

/// A null index for page linked lists.
const NULL_PAGE: *mut PageMeta = ptr::null_mut();

/// Metadata for a single page inside an Arena.
///
/// On 64-bit systems, this takes 24 bytes (1+2+2+3 pad + 8 + 8).
/// With 512 pages, the metadata array takes about 12 KB (3 pages).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PageMeta {
    /// Size class index (0..21). 0xFF means unallocated or large block.
    pub size_class: u8,
    /// Thread ID of the owner. 0xFFFF means unowned.
    pub owner_id: u16,
    /// Number of used slots in this page.
    pub used_slots: u16,
    /// Head of the free list (index of the first free slot).
    pub free_list_head: u16,
    /// Next page in the partially-free list (pointer).
    pub next_page: *mut PageMeta,
    /// Previous page in the partially-free list (pointer).
    pub prev_page: *mut PageMeta,
    /// Direct pointer to the page data (fast path without arena lock).
    pub page_addr: *mut u8,
}

impl Default for PageMeta {
    fn default() -> Self {
        Self::empty()
    }
}

impl PageMeta {
    pub const fn empty() -> Self {
        PageMeta {
            size_class: 0xFF,
            owner_id: 0xFFFF,
            used_slots: 0,
            free_list_head: NULL_SLOT,
            next_page: NULL_PAGE,
            prev_page: NULL_PAGE,
            page_addr: ptr::null_mut(),
        }
    }
}

/// The Slab Allocator manages size classes for a specific thread/context.
pub struct SlabAllocator {
    /// For each size class, the index of the first page that has free slots.
    /// Pages are referenced by their index within their owning arena, but since
    /// we only support one global arena for now, we just store the `*mut PageMeta`.
    /// In Phase 4, this will be tied to a specific `ThreadArena`.
    partial_pages: [*mut PageMeta; NUM_SIZE_CLASSES],
}

impl SlabAllocator {
    pub const fn new() -> Self {
        SlabAllocator {
            partial_pages: [ptr::null_mut(); NUM_SIZE_CLASSES],
        }
    }

    /// Check if we need to request a new page from the arena for this size class.
    pub fn alloc_needs_page(&self, size_class: u8) -> bool {
        let class_idx = size_class as usize;
        self.partial_pages[class_idx].is_null()
    }

    /// Fast path allocation that doesn't need to touch the arena.
    /// Panic if there are no partial pages.
    pub fn alloc_no_arena(&mut self, size_class: u8) -> *mut u8 {
        let class_idx = size_class as usize;
        let page_meta_ptr = self.partial_pages[class_idx];
        
        debug_assert!(!page_meta_ptr.is_null());

        let meta = unsafe { &mut *page_meta_ptr };
        let slot_idx = meta.free_list_head;
        let slot_size = class_to_size(size_class);
        
        let slot_addr = unsafe { meta.page_addr.add((slot_idx as usize) * slot_size) };
        
        // Read next free slot index from the slot itself
        let next_idx = unsafe { ptr::read(slot_addr as *const u16) };
        meta.free_list_head = next_idx;
        meta.used_slots += 1;

        // If the page is now full, remove it from the partial list
        if meta.free_list_head == NULL_SLOT {
            self.remove_partial_page(size_class, page_meta_ptr);
        }

        slot_addr
    }

    /// Allocate a slot for the given `size_class`.
    ///
    /// Requires access to the `Arena` to allocate new pages if all existing
    /// pages are full.
    pub fn alloc(&mut self, size_class: u8, arena: &mut Arena, owner_id: u16) -> *mut u8 {
        let class_idx = size_class as usize;
        let mut page_meta_ptr = self.partial_pages[class_idx];

        if page_meta_ptr.is_null() {
            // No partially free pages for this size class. We need a new page.
            if let Some((new_meta_ptr, page_addr)) = arena.alloc_page() {
                self.init_page(new_meta_ptr, page_addr, size_class, owner_id);
                page_meta_ptr = new_meta_ptr;
                self.push_partial_page(size_class, page_meta_ptr);
            } else {
                return ptr::null_mut(); // OOM
            }
        }

        let meta = unsafe { &mut *page_meta_ptr };
        debug_assert!(meta.size_class == size_class);
        debug_assert!(meta.free_list_head != NULL_SLOT);

        // Pop from free list
        let slot_idx = meta.free_list_head;
        let slot_size = class_to_size(size_class);
        
        let page_index = arena.meta_index(page_meta_ptr);
        let page_base = arena.page_addr(page_index);
        
        let slot_addr = unsafe { page_base.add((slot_idx as usize) * slot_size) };
        
        // Read next free slot index from the slot itself
        let next_idx = unsafe { ptr::read(slot_addr as *const u16) };
        meta.free_list_head = next_idx;
        meta.used_slots += 1;

        // If the page is now full, remove it from the partial list
        if meta.free_list_head == NULL_SLOT {
            self.remove_partial_page(size_class, page_meta_ptr);
        }

        slot_addr
    }

    /// Free a slot back to its page.
    pub fn free(&mut self, ptr: *mut u8, arena: &mut Arena) {
        // Find the page index from the pointer
        let page_index = arena.ptr_to_page_index(ptr);
        if page_index == 0 {
            // Pointer is not in this arena, or points to metadata
            return;
        }

        let meta_ptr = arena.meta_ptr(page_index);
        let meta = unsafe { &mut *meta_ptr };
        let size_class = meta.size_class;

        if size_class == 0xFF {
            // Double free or invalid pointer
            return;
        }

        let slot_size = class_to_size(size_class);
        let page_base = arena.page_addr(page_index);
        let offset = (ptr as usize) - (page_base as usize);
        let slot_idx = (offset / slot_size) as u16;

        let was_full = meta.free_list_head == NULL_SLOT;

        // Push to free list
        unsafe {
            ptr::write(ptr as *mut u16, meta.free_list_head);
        }
        meta.free_list_head = slot_idx;
        meta.used_slots -= 1;

        // If the page was full and now has 1 free slot, add it back to partials
        if was_full {
            self.push_partial_page(size_class, meta_ptr);
        }

        // If the page is completely empty, we *could* return it to the arena
        // (Deferred to Phase 3/4 integration)
    }

    /// Free a slot without touching the arena (fast path for ThreadCache).
    /// Assumes ownership and validity checks have already been performed by the caller.
    pub fn free_no_arena(&mut self, ptr: *mut u8, _page_index: usize, meta_ptr: *mut PageMeta) {
        let meta = unsafe { &mut *meta_ptr };
        let size_class = meta.size_class;
        let slot_size = class_to_size(size_class);
        
        let offset = (ptr as usize) - (meta.page_addr as usize);
        let slot_idx = (offset / slot_size) as u16;

        let was_full = meta.free_list_head == NULL_SLOT;

        // Push to free list
        unsafe {
            ptr::write(ptr as *mut u16, meta.free_list_head);
        }
        meta.free_list_head = slot_idx;
        meta.used_slots -= 1;

        // If the page was full and now has 1 free slot, add it back to partials
        if was_full {
            self.push_partial_page(size_class, meta_ptr);
        }
    }

    /// Initialize a newly allocated page for a specific size class.
    fn init_page(&self, meta_ptr: *mut PageMeta, page_base: *mut u8, size_class: u8, owner_id: u16) {
        let meta = unsafe { &mut *meta_ptr };
        meta.size_class = size_class;
        meta.owner_id = owner_id;
        meta.used_slots = 0;
        meta.free_list_head = 0;
        meta.next_page = NULL_PAGE;
        meta.prev_page = NULL_PAGE;
        meta.page_addr = page_base;

        let slot_size = class_to_size(size_class);
        let slots = slots_per_page(size_class, PAGE_SIZE);

        // Build the embedded free list (each slot contains the index of the next)
        unsafe {
            for i in 0..(slots - 1) {
                let slot_addr = page_base.add(i * slot_size);
                ptr::write(slot_addr as *mut u16, (i + 1) as u16);
            }
            // Last slot points to NULL_SLOT
            let last_slot = page_base.add((slots - 1) * slot_size);
            ptr::write(last_slot as *mut u16, NULL_SLOT);
        }
    }

    fn push_partial_page(&mut self, size_class: u8, meta_ptr: *mut PageMeta) {
        let class_idx = size_class as usize;
        let head = self.partial_pages[class_idx];
        
        unsafe {
            (*meta_ptr).prev_page = NULL_PAGE;
            (*meta_ptr).next_page = head;
            
            if !head.is_null() {
                (*head).prev_page = meta_ptr;
            }
        }
        self.partial_pages[class_idx] = meta_ptr;
    }

    fn remove_partial_page(&mut self, size_class: u8, meta_ptr: *mut PageMeta) {
        let class_idx = size_class as usize;
        
        unsafe {
            let prev = (*meta_ptr).prev_page;
            let next = (*meta_ptr).next_page;
            
            if !prev.is_null() {
                (*prev).next_page = next;
            } else {
                // Was head
                self.partial_pages[class_idx] = next;
            }
            
            if !next.is_null() {
                (*next).prev_page = prev;
            }
            
            (*meta_ptr).prev_page = NULL_PAGE;
            (*meta_ptr).next_page = NULL_PAGE;
        }
    }
}
