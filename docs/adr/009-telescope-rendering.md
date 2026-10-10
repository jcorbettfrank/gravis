# ADR 009: Telescope-Like Particle Rendering

**Date**: 2026-10-10
**Status**: Accepted

## Decision

Draw each particle by its **type**, never its mass. Stars and gas share one size and brightness per scene; dark matter is hidden unless the viewer asks for it; point masses (black holes, a central star) are small, very bright points that bloom into a glow. Every billboard has a minimum on-screen size, like a telescope's point-spread function. The styling lives in `render-core` (`color::ParticleStyler`), so the native app and the browser share it.

## Context

The particle shader scaled each billboard's size by `mass^(1/3)` and its brightness by `mass^(1/2)`. That only works while every particle has the same mass, and M8 breaks this in two ways:

- **Specs mix very different masses.** In `specs/cluster-and-black-hole.json` the black hole is 3 million times heavier than a cluster star, so it rendered as a large, flat blob that hid the stars near it.
- **Star/dark-matter mass splits** (workstream A) make star particles lighter and halo particles heavier. Under mass scaling, the visible disk would dim and shrink just as it gains particles, and the invisible halo would grow brighter.

Separately, dark matter got ~91% of a default galaxy's particles. Drawn as a translucent purple cloud, it dimmed the image without telling the viewer anything a telescope would.

## Design

### 1. Type decides look; count sets the scale

`ParticleStyler::new(types, show_dark_matter)` counts the luminous particles (every type except dark matter and point masses) and the dark-matter particles separately. Each population gets `size = 0.03 · (1000/n)^(1/3)` and `brightness = clamp((1000/n)^(1/2), 0.3, 1)`. For an equal-mass scene of total mass 1 these are exactly the old mass-based values, so the built-in scenarios look the same (a unit test pins this). The difference shows up only when masses differ or dark matter is present: a galaxy's stars are now sized by the number of *stars*, so hiding or showing the halo doesn't change them.

### 2. Dark matter is hidden by default

Native app: **Show dark matter** checkbox in the HUD, the **D** key, or `--show-dark-matter` for screenshots. Browser: a **Dark matter** checkbox in the demo toolbar. Hidden particles are left out of the instance buffer, not drawn transparent. When shown, dark matter is a faint violet haze.

### 3. Point masses are point sources

A new `ParticleType::PointMass = 5` marks spec `point_mass` components and the protoplanetary disk's star. The type changes nothing physical; it only tells renderers what the particle is. A point mass is drawn at 1.5× star size (capped at 0.1 world units) with HDR brightness 25, well above the bloom threshold, so it reads as a bright point with a glow at any mass.

### 4. A minimum angular size (point-spread function)

Billboards were sized in world units only, so in galaxy-scale specs (view radius ~10–30) stars were sub-pixel and mostly vanished. The vertex shader now uses `max(size, 0.003 · depth)`: no particle covers less than about 3 px of an 800 px-tall view. Close up, world size wins and nothing changes; from far away, stars stay visible and dense regions add up brighter, which is how an unresolved stellar population looks through a telescope.

### 5. Instance format

`GpuParticle.mass` became `GpuParticle.size` (still 32 bytes). Brightness is folded into `color.rgb`, which may exceed 1 in the HDR target. The renderer no longer needs particle masses, so `RenderSnapshot.masses` is gone.

## Consequences

- Rendering is independent of the mass model, so workstream A can change particle masses without touching the renderer.
- `render-core::color::types` mirrors `sim_core::particle::ParticleType` (render-core has no sim-core dependency); a test in render-core keeps the ids in sync.
- The minimum angular size is a fraction of the view, so the same scene looks alike at different window heights, but on a Retina display a star covers about 6 physical pixels rather than 3.
- Rendering is still not photometric: there is no stellar population model, extinction, or per-particle luminosity. Claims stay at "telescope-like".
