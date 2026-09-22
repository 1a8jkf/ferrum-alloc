//! Direct OS allocation for huge memory blocks.

use crate::platform::{self, align_up, PAGE_SIZE};
use core::ptr;

pub const HUGE_ALLOC_THRESHOLD: usize = 256 * 1024; // 256 KB
const HUGE_MAGIC: u32 = 0x48554745; // "HUGE"

#[repr(C, align(16))]
pub struct HugeHeader {
    pub magic: u32,
    pub total_size: usize,
}

pub unsafe fn huge_alloc(size: usize) -> *mut u8 {
    // We allocate an extra PAGE_SIZE so we can place our HugeHeader at the
    // start of the OS allocation, and return a perfectly page-aligned pointer
    // to the user.
    let total_size = align_up(size + PAGE_SIZE, PAGE_SIZE);
    let base = platform::os_alloc(total_size);
    if base.is_null() {
        return ptr::null_mut();
    }

    let header = base as *mut HugeHeader;
    (*header).magic = HUGE_MAGIC;
    (*header).total_size = total_size;

    // Return the next page boundary
    base.add(PAGE_SIZE)
}

pub unsafe fn huge_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }

    // The header is at the start of the previous page
    let base = ptr.sub(PAGE_SIZE);
    let header = base as *mut HugeHeader;

    if (*header).magic == HUGE_MAGIC {
        let total_size = (*header).total_size;
        platform::os_free(base, total_size);
    } else {
        // If it's not a huge allocation, we can't free it safely.
        // It's likely an invalid pointer passed to free.
        if cfg!(debug_assertions) {
            unreachable!("ferrum_free: pointer is not in arena and not a valid huge allocation");
        }
    }
}

pub unsafe fn huge_usable_size(ptr: *mut u8) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let base = ptr.sub(PAGE_SIZE);
    let header = base as *mut HugeHeader;
    if (*header).magic == HUGE_MAGIC {
        (*header).total_size - PAGE_SIZE
    } else {
        0
    }
}
