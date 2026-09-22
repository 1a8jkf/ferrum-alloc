# FerrumAlloc

FerrumAlloc is an ultra-low latency, O(1) memory allocator tailored for high-performance computing, databases, network servers, and real-time systems.

By employing a lock-free ThreadCache over a Two-Level Segregated Fit (TLSF) global arena, it guarantees deterministic allocation and deallocation times (P99.9 latency under 150 CPU cycles) without suffering from thread contention in concurrent scenarios.

## Integration for Companies and Developers

FerrumAlloc is distributed primarily as source code to guarantee security and optimize compile-time link-time optimization (LTO) for your infrastructure.

### Rust Developers

To replace the default system allocator with FerrumAlloc in your Rust project, add the crate to your `Cargo.toml`:

```toml
[dependencies]
ferrum-alloc = "0.1"
```

Then, configure it as the global allocator in your `src/main.rs` or `src/lib.rs`:

```rust
use ferrum_alloc::FerrumAllocator;

#[global_allocator]
static GLOBAL: FerrumAllocator = FerrumAllocator;

fn main() {
    // Your high-performance code here
}
```

### C/C++ Developers (Servers, Databases, Custom Engines)

If you are integrating FerrumAlloc into a non-Rust codebase (e.g., overriding PostgreSQL, Redis, or a custom C++ game engine), you must compile FerrumAlloc as a static or dynamic library.

1. Clone the repository:
```bash
git clone https://github.com/1a8jkf/ferrum-alloc.git
cd ferrum-alloc
```

2. Compile the project in release mode:
```bash
cargo build --release
```

3. Link the resulting object file (`target/release/libferrum_alloc.a` or `target/release/ferrum_alloc.lib`) directly into your project's build system (CMake, Make, or MSBuild).

FerrumAlloc exports standard POSIX ABI signatures natively:
- `void* malloc(size_t size);`
- `void free(void* ptr);`
- `void* calloc(size_t nmemb, size_t size);`
- `void* realloc(void* ptr, size_t size);`
- `int posix_memalign(void** memptr, size_t alignment, size_t size);`

This guarantees drop-in compatibility with legacy codebases.

## Building Dynamic Libraries for Preloading

For deployment via `LD_PRELOAD` (Linux) or DLL overriding (Windows), the `Cargo.toml` is configured to build a `cdylib`.
The dynamic library will be available in `target/release/libferrum_alloc.so` or `target/release/ferrum_alloc.dll`.

## Architecture Details
- Lock-free Treiber Stack for cross-thread memory reclamation.
- Segregated boundary tags for O(1) coalescing.
- Fallback mapping to huge OS pages for extreme allocations.
