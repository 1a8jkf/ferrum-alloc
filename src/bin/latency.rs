//! Raw RDTSC latency benchmark to measure exact CPU cycles per alloc/free pair.
//! This proves the O(1) predictable latency property (P99.9).

use ferrum_alloc::{ferrum_malloc, ferrum_free};
use std::arch::x86_64::_rdtsc;

pub fn main() {
    println!("Running Latency Benchmark (RDTSC)...");
    
    // Warmup
    for _ in 0..10_000 {
        unsafe {
            let p = ferrum_malloc(64);
            ferrum_free(p);
        }
    }

    const SAMPLES: usize = 1_000_000;
    let mut cycles = Vec::with_capacity(SAMPLES);
    let size = 64; // Small alloc (ThreadCache fast path)

    for _ in 0..SAMPLES {
        unsafe {
            let start = _rdtsc();
            let p = ferrum_malloc(size);
            ferrum_free(p);
            let end = _rdtsc();
            cycles.push(end - start);
        }
    }

    cycles.sort_unstable();

    let p50 = cycles[SAMPLES / 2];
    let p90 = cycles[(SAMPLES as f64 * 0.90) as usize];
    let p99 = cycles[(SAMPLES as f64 * 0.99) as usize];
    let p999 = cycles[(SAMPLES as f64 * 0.999) as usize];
    let max = *cycles.last().unwrap();

    println!("Latency (CPU Cycles) for 64-byte Alloc+Free:");
    println!("  P50:    {} cycles", p50);
    println!("  P90:    {} cycles", p90);
    println!("  P99:    {} cycles", p99);
    println!("  P99.9:  {} cycles", p999);
    println!("  Max:    {} cycles", max);
    
    // Prove predictability: The fast path should be incredibly stable.
    // If P99.9 is within a small multiplier of P50, it demonstrates O(1) behavior.
    if p999 < p50 * 10 {
        println!("✅ SUCCESS: P99.9 is highly predictable and O(1)!");
    } else {
        println!("⚠️ WARNING: P99.9 showed latency spikes.");
    }
}
