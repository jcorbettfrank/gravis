# ADR 006: Web Target Architecture

**Date**: 2026-03-22
**Status**: Accepted

## Decision

1. Extract a `render-core` crate with platform-independent rendering (pipelines, camera math, GPU types, WGSL shaders).
2. Feature-gate rayon behind a `parallel` Cargo feature (default on) rather than `#[cfg(target_arch)]` — allows testing the sequential path on native.
3. Single-threaded sim loop on WASM with `requestAnimationFrame`.
4. iframe embedding for mdBook demos (not a custom preprocessor).

## Context

M4 adds a browser-playable WASM demo. We needed to share rendering code between `native-app` and `web-app`, handle the single-threaded WASM constraint, and embed demos in the mdBook.

### 1. render-core extraction

`native-app` previously owned all wgpu pipeline setup, camera math, and WGSL shaders. Duplicating that into `web-app` would create a maintenance burden. `render-core` factors out everything that does not depend on a windowing system or UI framework, so both targets consume the same rendering logic.

### 2. Feature-gated parallelism

ADR 005 added rayon as an unconditional dependency, noting that WASM gating would come later. A `parallel` Cargo feature (default on) is preferable to `#[cfg(target_arch = "wasm32")]` because it lets us test the sequential fallback path on native builds with `--no-default-features`, catching single-threaded bugs without cross-compiling.

### 3. Single-threaded WASM sim loop

The sim loop runs synchronously within a `requestAnimationFrame` callback. Each frame advances the simulation by one (or a few) timesteps, then re-renders. This avoids `SharedArrayBuffer` and cross-origin isolation requirements.

### 4. iframe embedding for mdBook

Live demos are standalone HTML pages loaded via `<iframe>` tags in mdBook chapters. This keeps the book build simple — no custom preprocessor, no wasm-bindgen glue in mdBook's build pipeline.

## Alternatives Rejected

- **wasm-bindgen-rayon with SharedArrayBuffer**: Complex setup requiring cross-origin isolation headers, limited browser support, and marginal benefit for N-body where force calculation dominates and the tree walk is inherently sequential per-particle.
- **winit web target**: Adds dependency weight for minimal benefit vs. raw `web-sys` canvas + mouse events. The web demo has simpler input needs than the native app.
- **mdBook preprocessor for live demos**: Brittle, hard to debug, and tightly couples the book build to WASM tooling. iframes are simpler and self-contained.

## Consequences

- `render-core` has zero platform dependencies (no winit, no egui).
- `web-app` depends on `sim-core` with `default-features = false` (disabling the `parallel` feature).
- Demo HTML pages must be copied into the book output directory during the book build.
- `getrandom` requires `wasm_js` configuration for WASM builds (handled by `.cargo/config.toml` in the web-app crate).
