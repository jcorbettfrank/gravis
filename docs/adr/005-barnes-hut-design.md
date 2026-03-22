# ADR 005: Barnes-Hut Octree Design

**Date**: 2026-03-22
**Status**: Accepted

## Decision

Use an arena-allocated octree with a temporary acceleration buffer for parallel force computation. Add `rayon` as an unconditional dependency.

## Context

M3 replaces the O(N^2) brute-force gravity solver with an O(N log N) Barnes-Hut tree code. Three design decisions were made:

### 1. Arena-allocated octree

All nodes live in a flat `Vec<OctreeNode>` with `[u32; 8]` child indices (sentinel `u32::MAX` for empty slots). This eliminates per-node heap allocations — one Vec allocation instead of thousands of Box allocations — and improves cache locality during tree walks. Safe Rust, no `unsafe`.

A depth limit of 64 prevents infinite recursion when particles overlap exactly.

### 2. Temporary acceleration buffer (no trait change)

The `GravitySolver` trait takes `&mut Particles`, which conflicts with rayon's requirement for shared read access during parallel tree walks. Rather than changing the trait signature (breaking backward compatibility), `BarnesHut::compute_accelerations` allocates a `Vec<[f64; 3]>` temporary buffer, computes accelerations in parallel via `par_iter`, and copies results back. The O(N) copy is negligible vs. the O(N log N) tree walk.

### 3. Rayon as unconditional dependency

`rayon` is added to `sim-core` without feature gating. The M6 WASM target will add `#[cfg(not(target_arch = "wasm32"))]` gating when needed. Adding it now would be premature abstraction.

Barnes-Hut force computation uses rayon for parallel tree walks, with a threshold (N >= 1000) to avoid dispatch overhead on small particle counts. The integrator's kick/drift loops remain sequential — they are memory-bound and rayon's per-call dispatch overhead exceeds the computation time even at large N.

## Consequences

- Momentum conservation is approximate (not machine-precision) because Barnes-Hut computes forces per-particle without Newton's third law pairing. The brute-force solver remains available for tests requiring exact conservation.
- Memory usage scales as ~O(N) for tree nodes. At N=500K, tree memory is modest (~tens of MB).
- The `sim-core` crate now depends on `rayon`, which pulls in `crossbeam` and thread pool infrastructure.

## Verification

- Force accuracy: <0.5% RMS at theta=0.5, <0.1% at theta=0.3 (N=1000 Plummer sphere)
- Scaling: log-log slope of 1.0-1.5 confirmed (vs. 2.0 for brute-force)
- Performance: 500K particles, 100 steps in ~65s on M5 Pro
- All M1 physics tests pass unchanged with brute-force solver
