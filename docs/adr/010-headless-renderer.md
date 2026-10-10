# ADR 010: Headless CPU Renderer and Run Summary

**Date**: 2026-10-10
**Status**: Accepted

## Decision

`headless` renders PNG frames on the CPU at requested sim times (`--render-at 0,25,50`) and writes a JSON summary of the run (`summary.json` beside the frames, or `--summary-json PATH`). Frames are projected density splats colored by particle type, dark matter hidden by default, tone-mapped with an asinh stretch. The summary carries energetics, per-component centers and sizes, and the closest approach between every pair of components.

## Context

M8's loop (describe → spec → run → check → revise) needs to *see* a run, and the sessions that drive it often run in cloud containers without a GPU or window server, where `native-app` can't start. The checks it needs are also partly numeric: did energy hold, did the dwarf actually reach the disk, was the cluster stripped. Reading those off an image is unreliable, so they go in the summary as numbers.

### 1. CPU splatting, not a headless GPU path

A wgpu offscreen renderer would match `native-app`'s look, but needs a GPU adapter (or a software Vulkan stack) that cloud containers don't reliably have. Splatting N ≤ 500K points into an 800×800 grid and blurring it takes well under a second, so a pure-Rust path (one new dependency, `png`) is simpler and runs everywhere, including CI. It does not depend on `render-core`, which pulls in wgpu.

### 2. Brightness per particle, by type

Each visible particle adds a fixed amount of its type's color (the hues of `render-core/src/color.rs`), independent of mass. This is what a telescope sees: light from stars, not mass. It also stays correct once star and halo particles get different masses (workstream A). Dark matter is hidden unless `--render-dm` is passed, and then drawn faint.

Point-mass components (black holes) are drawn as small white dots on top, rather than splatted, so a single particle stays visible however the exposure is set.

### 3. Cloud-in-cell deposit, Gaussian blur, asinh stretch

Bilinear (cloud-in-cell) deposit conserves light and avoids the aliasing of nearest-pixel binning. A Gaussian of ~1 px at 800 px (scaled with image size) suppresses shot noise without blurring structure. The asinh stretch (Lupton et al. 2004) is linear at the faint end and logarithmic at the bright end, so a single tidal-tail star and a galaxy core both register: β = exposure / 100.

Exposure is the 99.5th percentile of lit pixels in the *first* rendered frame of each view, then held fixed, so brightness is comparable across frames: a tail fading as it spreads looks like it. `--render-autoexpose` exposes each frame on its own instead, for when late, diffuse structure (a disrupted cluster's stream) is too faint against a compact first frame.

### 4. Fixed field of view

Frames are centered on the system's center of mass with a fixed half-width: `--render-radius`, else the spec's `view_radius()`, else a radius holding 90% of visible particles (×1.2) for named scenarios. A fixed frame shows motion; auto-framing each frame would hide it.

### 5. Component tracking

A spec's components are generated in order into contiguous index ranges, so headless follows each one without new bookkeeping in `sim-core`. A component's center is the mass center of its visible (non-dark-matter) particles within its initial visible half-mass radius of the previous center, iterated three times each step. Restricting to that radius keeps tidal tails and stripped halo from dragging the center. The minimum center separation of each pair is recorded every step (an O(N) pass, small next to the force calculation), so a fast pericenter passage isn't missed between frames.

Energetics at each frame use the exact O(N²) potential, like the rest of headless's diagnostics. At very large N this dominates a frame's cost; an approximate potential is workstream C's concern.

## Consequences

- Cloud sessions and CI can produce and inspect frames: `cargo run --release -p headless -- --spec specs/tilted-flyby.json --algorithm barnes-hut --render-at 0,25,50`.
- When `--render-at` is given without `--steps`, the run lasts until the last requested time.
- Frames don't match `native-app`'s look (no bloom, no perspective). They are diagnostic images, not media.
- The summary schema is the interface for the `/universe` skill (workstream E); fields should be added, not renamed.
