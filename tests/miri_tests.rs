//! Integration tests specifically designed to be validated by Miri.

use ferrum_alloc::{ferrum_malloc, ferrum_free};
use std::thread;

#[test]
fn test_alloc_free_basic() {
    unsafe {
        // Slab allocation (e.g. 32 bytes)
        let ptr1 = ferrum_malloc(32);
        assert!(!ptr1.is_null());
        *ptr1 = 42;
        assert_eq!(*ptr1, 42);

        // TLSF allocation (e.g. 4 KB)
        let ptr2 = ferrum_malloc(4096);
        assert!(!ptr2.is_null());
        *ptr2 = 99;
        assert_eq!(*ptr2, 99);

        // Huge allocation (e.g. 512 KB)
        let ptr3 = ferrum_malloc(512 * 1024);
        assert!(!ptr3.is_null());
        *ptr3 = 100;
        assert_eq!(*ptr3, 100);

        ferrum_free(ptr1);
        ferrum_free(ptr2);
        ferrum_free(ptr3);
    }
}

#[test]
fn test_cross_thread_free() {
    unsafe {
        let ptr = ferrum_malloc(64);
        assert!(!ptr.is_null());
        *ptr = 1;

        let handle = thread::spawn(move || {
            // Free the pointer from another thread
            // This pushes it to the remote_frees stack
            ferrum_free(ptr);
        });

        handle.join().unwrap();

        // Allocate again on the main thread to trigger `drain_remote_frees`
        let ptr2 = ferrum_malloc(64);
        ferrum_free(ptr2);
    }
}

#[test]
fn test_realloc_pattern() {
    unsafe {
        let mut ptrs = Vec::new();
        for _ in 0..100 {
            ptrs.push(ferrum_malloc(128));
        }

        // Free every other pointer to create fragmentation
        for i in (0..100).step_by(2) {
            ferrum_free(ptrs[i]);
        }

        // Free the rest
        for i in (1..100).step_by(2) {
            ferrum_free(ptrs[i]);
        }
    }
}
