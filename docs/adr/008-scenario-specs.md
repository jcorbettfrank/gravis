# ADR 008: Declarative Scenario Specs

**Date**: 2026-10-10
**Status**: Accepted

## Decision

Describe custom initial conditions as JSON specs that compose a small set of building blocks (`disk_galaxy`, `plummer`, `point_mass`, `gas_cloud`), each placed with a position and bulk velocity. Parse them with serde in `sim-core` (`spec.rs`), validate them with errors that name the offending field, and let every runner load them: headless and native-app via `--spec`, the browser via `load_spec(json)`.

## Context

Scenarios were Rust structs with fixed geometry, so trying a new configuration meant writing code and recompiling, and the browser could only run what was compiled in. The goal ("Text-to-Universe") is a loop in which a scenario is described in plain English, turned into initial conditions, checked against the conservation diagnostics, and revised. That needs a data format that is easy to write, by hand, by script, or by a language model, and safe to reject when wrong.

### 1. Composition over parameterized scenarios

A spec is a list of components rather than a bigger `GalaxyCollision` with more knobs. Most interesting setups are "these things, here, moving like this": two disks, a dwarf and a disk, a cluster and a black hole. Each component reuses an existing, tested generator (`GalaxyModel`, `sample_plummer_particle`, `fill_uniform_gas_sphere`), so a spec introduces no new physics. `GalaxyModel` was extracted from `GalaxyCollision` for this, with the RNG call order unchanged; the built-in scenario reproduces its previous energies to 11 significant figures.

Disk orientation is a single `spin_axis` vector (the disk's angular momentum direction). A disk is axisymmetric, so this is its full orientation, and it states prograde/retrograde intent directly, which matters most for tidal response (Toomre & Toomre 1972).

### 2. One RNG stream per component

Each component draws from `ChaCha20Rng::seed_from_u64(seed)` with `set_stream(index)`. Editing one component (making the dwarf denser, say) leaves every other component's particles unchanged, so an iteration changes one thing at a time.

### 3. Strict validation, descriptive errors

`#[serde(deny_unknown_fields)]` turns typos into errors that list the valid fields. `validate()` checks every number before generation, including conditions that would otherwise panic deep in a sampler (an empty Hernquist sampling range when `halo_max_radius` is tiny), and a total-particle cap of 500,000 guards against typos that would allocate gigabytes. Error messages carry the component index and type, so whoever wrote the spec can fix it without reading Rust.

### 4. Derived defaults for dt, softening, and framing

A spec may set `dt`, `softening`, and `view_radius`; otherwise they are derived:

- **Softening:** the smallest any extended component asks for (the same rules as the built-in scenarios). With only point masses, 0.1% of the closest pericenter, as in `TwoBody`.
- **Timestep:** the minimum of each extended component's internal timescale; each point-mass pair's orbit at pericenter / 1000; and, for each point mass and extended component, the orbit at (pericenter − component size) / 50. The last rule uses the component's actual closest approach instead of the softening length; a softening-based rule made a dense cluster orbiting a black hole take ~100× more steps than necessary.
- **View radius:** the farthest component's distance from the center of mass plus its visible size. Renderers frame it with `OrbitalCamera::frame_radius`.

These are defaults, not guarantees. The loop this format exists for checks them by running the spec and reading energy conservation.

### 5. serde in sim-core

`sim-core` has no rendering or IO dependencies. serde and serde_json parse an in-memory string; they do no IO, add nothing to the build that wasn't already in `Cargo.lock`, and keep one parser and one validator for all three runners. File reading stays in headless and native-app.

## Consequences

- Any runner can show any scenario without recompiling, including the browser.
- The format is the contract for the next steps: a Claude Code skill that writes specs from descriptions, and a public page where visitors type their own.
- Equal-mass particles mean dark matter gets ~91% of a default galaxy's particles, so disks look sparse at modest N. Unequal particle masses (more, lighter star particles) are a natural next field; their cost is disk heating by heavy halo particles, which should be measured on an isolated galaxy before choosing defaults.
- Gas specs inherit an existing SPH problem: self-gravitating gas blows up at peak compression (the Evrard bounce, colliding clouds). No example spec uses gas until that is fixed.
