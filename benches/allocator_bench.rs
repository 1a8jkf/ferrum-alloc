//! Benchmark harness for ferrum-alloc.
//!
//! Measures allocation performance against the system allocator baseline.
//! When jemalloc/mimalloc become available (requires C toolchain), they
//! can be added as additional baselines.
//!
//! Run with: `cargo bench --bench allocator_bench`

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::alloc::{GlobalAlloc, Layout, System};

#[repr(transparent)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}

// ============================================================================
// Benchmark helpers
// ============================================================================

/// Allocate using the system allocator (baseline)
#[inline]
unsafe fn system_alloc(size: usize) -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(size, 16);
    System.alloc(layout)
}

/// Free using the system allocator
#[inline]
unsafe fn system_dealloc(ptr: *mut u8, size: usize) {
    let layout = Layout::from_size_align_unchecked(size, 16);
    System.dealloc(ptr, layout);
}

/// Allocate using ferrum-alloc
#[inline]
unsafe fn ferrum_alloc(size: usize) -> *mut u8 {
    ferrum_alloc::ferrum_malloc(size)
}

/// Free using ferrum-alloc
#[inline]
unsafe fn ferrum_dealloc(ptr: *mut u8) {
    ferrum_alloc::ferrum_free(ptr);
}

// ============================================================================
// Benchmark: burst alloc/free
//
// Allocates N blocks in a burst, then frees them all. Measures raw
// allocation throughput without interleaving.
// ============================================================================

fn bench_burst_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("burst_alloc");

    for &size in &[8, 64, 256, 1024, 4096] {
        let count = 1000;
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("system", size),
            &size,
            |b, &size| {
                b.iter(|| unsafe {
                    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(count);
                    for _ in 0..count {
                        ptrs.push(black_box(system_alloc(size)));
                    }
                    for ptr in ptrs {
                        system_dealloc(ptr, size);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ferrum", size),
            &size,
            |b, &size| {
                b.iter(|| unsafe {
                    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(count);
                    for _ in 0..count {
                        ptrs.push(black_box(ferrum_alloc(size)));
                    }
                    for ptr in ptrs {
                        ferrum_dealloc(ptr);
                    }
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Benchmark: random size mix
//
// Realistic workload: 80% small (<256B), 15% medium (256B-4KB), 5% large (>4KB).
// ============================================================================

fn bench_random_size_mix(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_size_mix");

    // Pre-generate sizes to avoid RNG overhead in the benchmark loop
    let sizes: Vec<usize> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..1000)
            .map(|_| {
                let roll: f64 = rng.r#gen();
                if roll < 0.80 {
                    rng.gen_range(1..=256)
                } else if roll < 0.95 {
                    rng.gen_range(257..=4096)
                } else {
                    rng.gen_range(4097..=65536)
                }
            })
            .collect()
    };

    group.throughput(Throughput::Elements(sizes.len() as u64));

    group.bench_function("system", |b| {
        b.iter(|| unsafe {
            let mut ptrs: Vec<(*mut u8, usize)> = Vec::with_capacity(sizes.len());
            for &size in &sizes {
                let p = system_alloc(size);
                ptrs.push((black_box(p), size));
            }
            for (ptr, size) in ptrs {
                system_dealloc(ptr, size);
            }
        });
    });

    group.bench_function("ferrum", |b| {
        b.iter(|| unsafe {
            let mut ptrs: Vec<*mut u8> = Vec::with_capacity(sizes.len());
            for &size in &sizes {
                let p = ferrum_alloc(size);
                ptrs.push(black_box(p));
            }
            for ptr in ptrs {
                ferrum_dealloc(ptr);
            }
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: adversarial fragmentation
//
// Allocates N blocks, frees every other one (maximizing fragmentation),
// then allocates again into the gaps. Measures how well the allocator
// handles fragmentation.
//
// Pattern: alloc[0..N], free[1,3,5,...], alloc[0..N/2]
// ============================================================================

fn bench_adversarial_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("adversarial_fragmentation");
    let count = 500;

    group.throughput(Throughput::Elements(count as u64));

    group.bench_function("system", |b| {
        b.iter(|| unsafe {
            let size = 64;
            let mut ptrs: Vec<*mut u8> = Vec::with_capacity(count);

            // Phase 1: allocate all
            for _ in 0..count {
                ptrs.push(system_alloc(size));
            }

            // Phase 2: free odd indices (create holes)
            for i in (1..count).step_by(2) {
                system_dealloc(ptrs[i], size);
                ptrs[i] = std::ptr::null_mut();
            }

            // Phase 3: allocate into the gaps
            let mut new_ptrs: Vec<*mut u8> = Vec::with_capacity(count / 2);
            for _ in 0..(count / 2) {
                new_ptrs.push(black_box(system_alloc(size)));
            }

            // Cleanup
            for p in ptrs {
                if !p.is_null() {
                    system_dealloc(p, size);
                }
            }
            for p in new_ptrs {
                system_dealloc(p, size);
            }
        });
    });

    group.bench_function("ferrum", |b| {
        b.iter(|| unsafe {
            let size = 64;
            let mut ptrs: Vec<*mut u8> = Vec::with_capacity(count);

            for _ in 0..count {
                ptrs.push(ferrum_alloc(size));
            }

            for i in (1..count).step_by(2) {
                ferrum_dealloc(ptrs[i]);
                ptrs[i] = std::ptr::null_mut();
            }

            let mut new_ptrs: Vec<*mut u8> = Vec::with_capacity(count / 2);
            for _ in 0..(count / 2) {
                new_ptrs.push(black_box(ferrum_alloc(size)));
            }

            for p in ptrs {
                if !p.is_null() {
                    ferrum_dealloc(p);
                }
            }
            for p in new_ptrs {
                ferrum_dealloc(p);
            }
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: single alloc/free latency
//
// Measures the raw latency of a single allocation and free, to get
// p50/p99/p99.9 numbers.
// ============================================================================

fn bench_single_alloc_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_alloc_latency");

    for &size in &[8, 64, 512, 4096] {
        group.bench_with_input(
            BenchmarkId::new("system", size),
            &size,
            |b, &size| {
                b.iter(|| unsafe {
                    let p = black_box(system_alloc(size));
                    system_dealloc(p, size);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ferrum", size),
            &size,
            |b, &size| {
                b.iter(|| unsafe {
                    let p = black_box(ferrum_alloc(size));
                    ferrum_dealloc(p);
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Benchmark: multithread throughput
// ============================================================================

fn bench_multithread_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("multithread_throughput");
    
    // Test with 2, 4, 8 threads
    for &threads in &[2, 4, 8] {
        let count_per_thread = 2000;
        let total_count = threads * count_per_thread;
        
        group.throughput(Throughput::Elements(total_count as u64));

        group.bench_with_input(
            BenchmarkId::new("system", threads),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    std::thread::scope(|s| {
                        for _ in 0..threads {
                            s.spawn(move || unsafe {
                                let mut ptrs = Vec::with_capacity(count_per_thread);
                                for _ in 0..count_per_thread {
                                    ptrs.push(black_box(system_alloc(64)));
                                }
                                for ptr in ptrs {
                                    system_dealloc(ptr, 64);
                                }
                            });
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ferrum", threads),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    std::thread::scope(|s| {
                        for _ in 0..threads {
                            s.spawn(move || unsafe {
                                let mut ptrs = Vec::with_capacity(count_per_thread);
                                for _ in 0..count_per_thread {
                                    ptrs.push(black_box(ferrum_alloc(64)));
                                }
                                for ptr in ptrs {
                                    ferrum_dealloc(ptr);
                                }
                            });
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Benchmark: cross-thread free
// ============================================================================

fn bench_cross_thread_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_thread_free");
    let count = 2000;
    
    group.throughput(Throughput::Elements(count as u64));

    group.bench_function("system", |b| {
        b.iter(|| {
            let (tx, rx) = std::sync::mpsc::channel::<SendPtr>();
            
            std::thread::scope(|s| {
                // Allocator thread
                s.spawn(move || unsafe {
                    for _ in 0..count {
                        tx.send(SendPtr(black_box(system_alloc(64)))).unwrap();
                    }
                });
                
                // Deallocator thread
                s.spawn(move || unsafe {
                    for _ in 0..count {
                        let ptr = rx.recv().unwrap().0;
                        system_dealloc(ptr, 64);
                    }
                });
            });
        });
    });

    group.bench_function("ferrum", |b| {
        b.iter(|| {
            let (tx, rx) = std::sync::mpsc::channel::<SendPtr>();
            
            std::thread::scope(|s| {
                // Allocator thread
                s.spawn(move || unsafe {
                    for _ in 0..count {
                        tx.send(SendPtr(black_box(ferrum_alloc(64)))).unwrap();
                    }
                });
                
                // Deallocator thread
                s.spawn(move || unsafe {
                    for _ in 0..count {
                        let ptr = rx.recv().unwrap().0;
                        ferrum_dealloc(ptr);
                    }
                });
            });
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: Huge Allocations
// ============================================================================

fn bench_huge_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("huge_alloc");

    for &size in &[256 * 1024, 1024 * 1024] {
        let count = 10;
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("system", size),
            &size,
            |b, &size| {
                b.iter(|| unsafe {
                    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(count);
                    for _ in 0..count {
                        ptrs.push(black_box(system_alloc(size)));
                    }
                    for ptr in ptrs {
                        system_dealloc(ptr, size);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ferrum", size),
            &size,
            |b, &size| {
                b.iter(|| unsafe {
                    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(count);
                    for _ in 0..count {
                        ptrs.push(black_box(ferrum_alloc(size)));
                    }
                    for ptr in ptrs {
                        ferrum_dealloc(ptr);
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_alloc_latency,
    bench_burst_alloc,
    bench_random_size_mix,
    bench_adversarial_fragmentation,
    bench_multithread_throughput,
    bench_cross_thread_free,
    bench_huge_alloc,
);
criterion_main!(benches);
