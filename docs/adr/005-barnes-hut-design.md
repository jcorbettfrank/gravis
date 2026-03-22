# ADR 005: Barnes-Hut Octree Design

**Date**: 2026-03-22
**Status**: Accepted

## Decision

Use a recursive enum octree with a temporary acceleration buffer for parallel force computation. Add `rayon` as an unconditional dependency.

## Context

M3 replaces the O(N^2) brute-force gravity solver with an O(N log N) Barnes-Hut tree code. Three design decisions were made:

### 1. Recursive enum vs. arena-based octree

We use a recursive `OctreeNode` struct with `Option<Box<[Option<OctreeNode>; 8]>>` for children. This is safe Rust with no `unsafe` blocks, matching the project convention.

An arena-based (linearized) octree would improve cache locality during tree walks but adds implementation complexity and requires either `unsafe` or a crate like `typed-arena`. Since tree construction is not the bottleneck (the tree walk dominates), the recursive approach is sufficient.

A depth limit of 64 prevents stack overflow when particles overlap exactly.

### 2. Temporary acceleration buffer (no trait change)

The `GravitySolver` trait takes `&mut Particles`, which conflicts with rayon's requirement for shared read access during parallel tree walks. Rather than changing the trait signature (breaking backward compatibility), `BarnesHut::compute_accelerations` allocates a `Vec<[f64; 3]>` temporary buffer, computes accelerations in parallel via `par_iter`, and copies results back. The O(N) copy is negligible vs. the O(N log N) tree walk.

### 3. Rayon as unconditional dependency

`rayon` is added to `sim-core` without feature gating. The M6 WASM target will add `#[cfg(not(target_arch = "wasm32"))]` gating when needed. Adding it now would be premature abstraction.

Both Barnes-Hut force computation and the leapfrog integrator's kick/drift loops use rayon, with a threshold (N >= 1000) to avoid overhead on small particle counts.

## Consequences

- Momentum conservation is approximate (not machine-precision) because Barnes-Hut computes forces per-particle without Newton's third law pairing. The brute-force solver remains available for tests requiring exact conservation.
- Memory usage scales as ~O(N) for tree nodes. At N=500K, tree memory is modest (~tens of MB).
- The `sim-core` crate now depends on `rayon`, which pulls in `crossbeam` and thread pool infrastructure.

## Verification

- Force accuracy: <0.5% RMS at theta=0.5, <0.1% at theta=0.3 (N=1000 Plummer sphere)
- Scaling: log-log slope of 1.0-1.5 confirmed (vs. 2.0 for brute-force)
- Performance: 500K particles, 100 steps in ~65s on M5 Pro
- All M1 physics tests pass unchanged with brute-force solver
