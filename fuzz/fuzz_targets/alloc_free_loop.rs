#![no_main]

use libfuzzer_sys::fuzz_target;
use ferrum_alloc::{ferrum_malloc, ferrum_free};

fuzz_target!(|data: &[u8]| {
    // We interpret the random byte stream as a sequence of operations
    let mut ptrs: Vec<*mut u8> = Vec::new();
    
    let mut i = 0;
    while i < data.len() {
        let op = data[i] % 3;
        i += 1;
        
        match op {
            0 => { // Alloc small
                let size = (data.get(i).copied().unwrap_or(0) as usize) + 1; // 1 to 256
                i += 1;
                unsafe {
                    let p = ferrum_malloc(size);
                    if !p.is_null() {
                        // Write to the memory to ensure it's mapped and not overlapping headers
                        *p = 0xFF;
                        ptrs.push(p);
                    }
                }
            }
            1 => { // Alloc large
                if i + 3 <= data.len() {
                    let size = u32::from_le_bytes([data[i], data[i+1], data[i+2], 0]) as usize; // up to 16MB
                    i += 3;
                    // Cap it to prevent instant OOM from fuzzer picking huge numbers
                    let size = (size % (4 * 1024 * 1024)) + 1; 
                    unsafe {
                        let p = ferrum_malloc(size);
                        if !p.is_null() {
                            *p = 0xFF;
                            ptrs.push(p);
                        }
                    }
                }
            }
            2 => { // Free
                if !ptrs.is_empty() {
                    let idx = (data.get(i).copied().unwrap_or(0) as usize) % ptrs.len();
                    i += 1;
                    
                    let p = ptrs.swap_remove(idx);
                    unsafe { ferrum_free(p); }
                }
            }
            _ => unreachable!(),
        }
    }
    
    // Clean up anything left
    for p in ptrs {
        unsafe { ferrum_free(p); }
    }
});
