//! Size class definitions and lookup for slab allocation.
//!
//! Bins are chosen to minimize internal fragmentation while keeping the
//! number of classes manageable. The progression uses:
//! - 8-byte steps for tiny sizes (8-64)
//! - 16-byte steps for small sizes (64-128)  
//! - Geometric growth (~1.25x) for medium sizes (128-1024)
//!
//! This is similar to the approach used by mimalloc and jemalloc.
//!
//! ## Design Decision: Page-Level Metadata
//!
//! Instead of per-allocation headers (which pollute cache lines), we store
//! metadata at the page/slab level. Given a pointer, we can determine its
//! size class by computing `(ptr - arena_base) / page_size` and looking up
//! the page metadata array. This gives us O(1) size lookup with near-zero
//! per-allocation overhead.

/// Maximum size handled by the slab allocator. Anything larger goes to TLSF.
pub const SLAB_MAX_SIZE: usize = 1024;

/// The size class bins. Each entry is the maximum allocation size that fits
/// in that bin (i.e., the slot size).
pub const SIZE_CLASSES: [usize; 22] = [
    8, 16, 24, 32, 48, 64,           // tiny: 8-byte and 16-byte steps
    80, 96, 112, 128,                 // small: 16-byte steps
    160, 192, 224, 256,               // medium-small: 32-byte steps
    320, 384, 448, 512,               // medium: 64-byte steps
    640, 768, 896, 1024,              // medium-large: 128-byte steps
];

/// Number of size classes
pub const NUM_SIZE_CLASSES: usize = SIZE_CLASSES.len();

/// Lookup table: for a given size (0..=1024), maps to the index into SIZE_CLASSES.
/// Built at compile time for O(1) lookup.
///
/// Usage: `SIZE_CLASS_LUT[size]` gives the index into `SIZE_CLASSES` for
/// the smallest class that can hold `size` bytes.
static SIZE_CLASS_LUT: [u8; SLAB_MAX_SIZE + 1] = {
    let mut lut = [0u8; SLAB_MAX_SIZE + 1];
    let mut size = 0usize;
    while size <= SLAB_MAX_SIZE {
        // Find the smallest size class that fits this size
        let mut class_idx = 0u8;
        while (class_idx as usize) < NUM_SIZE_CLASSES {
            if SIZE_CLASSES[class_idx as usize] >= size {
                break;
            }
            class_idx += 1;
        }
        lut[size] = class_idx;
        size += 1;
    }
    lut
};

/// Given a requested allocation size, return the size class index.
///
/// Returns `None` if the size exceeds `SLAB_MAX_SIZE` (should go to TLSF).
/// Returns `Some(0)` for size 0 (minimum allocation is 8 bytes).
///
/// This is O(1) — a single array lookup.
#[inline(always)]
pub fn size_to_class(size: usize) -> Option<u8> {
    if size > SLAB_MAX_SIZE {
        return None;
    }
    // Size 0 maps to class 0 (8 bytes) — we never allocate zero bytes
    // but the caller might round up.
    Some(SIZE_CLASS_LUT[size])
}

/// Given a size class index, return the slot size (bytes per slot).
#[inline(always)]
pub fn class_to_size(class: u8) -> usize {
    debug_assert!((class as usize) < NUM_SIZE_CLASSES);
    SIZE_CLASSES[class as usize]
}

/// Compute how many slots of the given class fit in a page.
///
/// Each slab page is dedicated to a single size class. The number of slots
/// determines the bitmap size needed for the free list.
#[inline]
pub fn slots_per_page(class: u8, page_size: usize) -> usize {
    let slot_size = class_to_size(class);
    if slot_size == 0 {
        return 0;
    }
    page_size / slot_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_classes_are_sorted() {
        for i in 1..SIZE_CLASSES.len() {
            assert!(
                SIZE_CLASSES[i] > SIZE_CLASSES[i - 1],
                "Size classes must be strictly increasing: [{}]={} <= [{}]={}",
                i, SIZE_CLASSES[i], i - 1, SIZE_CLASSES[i - 1]
            );
        }
    }

    #[test]
    fn test_size_to_class_exact_matches() {
        for (i, &size) in SIZE_CLASSES.iter().enumerate() {
            let class = size_to_class(size).expect("Should fit in slab");
            assert_eq!(class as usize, i, "Size {} should map to class {}", size, i);
        }
    }

    #[test]
    fn test_size_to_class_round_up() {
        // 1 byte → class 0 (8 bytes)
        assert_eq!(size_to_class(1), Some(0));
        // 9 bytes → class 1 (16 bytes)
        assert_eq!(size_to_class(9), Some(1));
        // 17 bytes → class 2 (24 bytes)
        assert_eq!(size_to_class(17), Some(2));
        // 65 bytes → class 6 (80 bytes)
        assert_eq!(size_to_class(65), Some(6));
    }

    #[test]
    fn test_size_to_class_zero() {
        // Size 0 should map to the smallest class
        assert_eq!(size_to_class(0), Some(0));
    }

    #[test]
    fn test_size_to_class_too_large() {
        assert_eq!(size_to_class(1025), None);
        assert_eq!(size_to_class(2048), None);
        assert_eq!(size_to_class(usize::MAX), None);
    }

    #[test]
    fn test_class_to_size_roundtrip() {
        for i in 0..NUM_SIZE_CLASSES {
            let size = class_to_size(i as u8);
            let class = size_to_class(size).unwrap();
            assert_eq!(class as usize, i);
        }
    }

    #[test]
    fn test_no_wasted_class() {
        // Every size from 1 to SLAB_MAX_SIZE should map to a valid class
        for size in 1..=SLAB_MAX_SIZE {
            let class = size_to_class(size);
            assert!(class.is_some(), "Size {} has no class", size);
            let slot_size = class_to_size(class.unwrap());
            assert!(
                slot_size >= size,
                "Class for size {} has slot_size {} which is smaller",
                size, slot_size
            );
        }
    }

    #[test]
    fn test_internal_fragmentation_bounded() {
        // Worst-case internal fragmentation should be reasonable
        // For sizes > 8, check that (slot_size - size) / slot_size < 50%
        for size in 9..=SLAB_MAX_SIZE {
            let class = size_to_class(size).unwrap();
            let slot_size = class_to_size(class);
            let waste = slot_size - size;
            let waste_pct = (waste as f64 / slot_size as f64) * 100.0;
            assert!(
                waste_pct < 50.0,
                "Size {} maps to slot {} with {:.1}% waste",
                size, slot_size, waste_pct
            );
        }
    }

    #[test]
    fn test_slots_per_page() {
        let page = 4096usize;
        // 8-byte slots: 4096/8 = 512
        assert_eq!(slots_per_page(0, page), 512);
        // 1024-byte slots: 4096/1024 = 4
        let last_class = (NUM_SIZE_CLASSES - 1) as u8;
        assert_eq!(slots_per_page(last_class, page), 4);
    }

    #[test]
    fn test_max_size_class_is_slab_max() {
        assert_eq!(SIZE_CLASSES[NUM_SIZE_CLASSES - 1], SLAB_MAX_SIZE);
    }
}
