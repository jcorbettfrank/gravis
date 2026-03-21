# ADR 001: Workspace Layout

**Date**: 2026-03-21
**Status**: Accepted

## Decision

Use a Cargo workspace with separate crates:
- `sim-core` — pure simulation library, no rendering, no IO
- `headless` — CLI batch runner
- `native-app` — macOS renderer (future)
- `web-app` — WASM+WebGPU target (future)

## Context

The simulation core needs to be usable from multiple frontends (headless batch, native GUI, web browser) without pulling in rendering dependencies. A single crate with feature flags could achieve this but makes dependency boundaries implicit. A workspace makes them explicit and compiler-enforced.

## Consequences

- `sim-core` can never accidentally depend on wgpu, winit, or egui.
- Integration tests for physics run without GPU setup.
- The `headless` crate is how we generate all artifacts (benchmarks, snapshots, diagnostic CSVs) for the mdBook lessons.
- Slightly more boilerplate (multiple Cargo.toml files) but worth it for clarity.
