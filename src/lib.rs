//! # ferrum-alloc
//!
//! High-performance memory allocator with:
//! - O(1) deterministic allocation via TLSF (Two-Level Segregated Fit)
//! - Thread-local arenas for near-zero contention
//! - Lock-free cross-thread free via Treiber stack
//! - Slab allocation for small size classes
//! - Dual-platform support (Windows VirtualAlloc + Linux mmap)
//!
//! ## Architecture
//!
//! ```text
//! malloc(size) ──► Router
//!                   ├── ≤1024B ──► Slab Allocator (size classes)
//!                   ├── 1KB-64KB ─► TLSF (segregated fit, O(1))
//!                   └── >64KB ───► Huge Allocator (direct mmap)
//!
//! Each thread gets its own arena. Cross-thread frees go through
//! a lock-free Treiber stack, drained periodically by the owning thread.
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

mod platform;
mod arena;
mod size_class;
mod slab;
mod tlsf;
mod thread_cache;
pub mod huge;

// Re-export the C API
pub use crate::api::*;

mod api;
