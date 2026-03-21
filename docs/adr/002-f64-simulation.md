# ADR 002: f64 for Simulation, f32 for Rendering

**Date**: 2026-03-21
**Status**: Accepted

## Decision

All simulation quantities (positions, velocities, accelerations, energies, masses) use `f64`. Only GPU buffer packing and rendering use `f32`.

## Context

f32 has ~7 decimal digits of precision. For a galaxy-scale simulation spanning 50 kpc with softening of 0.05 kpc, the dynamic range is ~10^6, which consumes most of f32's precision. Long-run integrations (1000+ orbits) accumulate round-off errors that can masquerade as physical energy drift, making it impossible to distinguish integrator bugs from floating-point noise.

f64 has ~16 decimal digits, giving comfortable headroom. The cost is 2x memory per particle and potentially slower SIMD (128-bit SIMD holds 2 f64s vs 4 f32s). On the M5 Pro, the wider NEON units handle f64 efficiently, and the memory bandwidth (307 GB/s) makes this a non-issue for our target particle counts.

## Consequences

- Diagnostics (energy conservation, momentum) are trustworthy to ~10^-14.
- Benchmarks measure real algorithmic performance, not floating-point artifacts.
- GPU paths need an explicit f64→f32 conversion when packing vertex/storage buffers.
- WASM target may need special handling (f64 is supported but some WebGPU implementations have limited f64 compute shader support).
