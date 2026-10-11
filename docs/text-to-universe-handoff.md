# Text-to-Universe: handoff

Brief for whoever picks up M8 next, including a Claude Code project coordinator
running parallel threads. Read with [`PLAN.md` § M8](PLAN.md),
[ADR 008](adr/008-scenario-specs.md), and [`specs/README.md`](../specs/README.md).

## Goal

Describe a scenario in plain English ("a dwarf galaxy plunges through a Milky
Way disk") and watch it run. Claude writes a JSON scenario spec, runs it,
checks the physics (energy conservation, whether the encounter happens) and a
rendered frame, and revises until both hold up. Later, a public page where
visitors type their own descriptions.

## Where things stand (2026-10-10)

Step 1 is done on branch `feat/scenario-spec`, which builds on `main` at `3f0ce2c`:

- `crates/sim-core/src/spec.rs`: `ScenarioSpec` with components `disk_galaxy`
  (oriented by `spin_axis`), `plummer`, `point_mass`, `gas_cloud`. It handles
  validation with field-naming errors, per-component RNG streams,
  `with_particle_budget`, and derived `dt`/`softening`/`view_radius`.
  `ScenarioSpec` implements `Scenario`.
- `GalaxyModel` was extracted from `GalaxyCollision`. The built-in scenario
  reproduces its pre-refactor energies to 11 significant figures.
- Runners: `headless --spec`, `native-app --spec` (camera auto-framed via
  `OrbitalCamera::frame_radius`), `native-app --screenshot-time T` (runs flat
  out to sim time T, then captures), browser `load_spec(json)` plus a
  **Spec…** panel in `crates/web-app/static/index.html`.
- Example specs in `specs/`, with the format reference in `specs/README.md`.
  Tests are in `crates/sim-core/tests/spec.rs` (14).
- Verified: full `cargo test -p sim-core --release` passes. Clippy is clean
  with `-D warnings` for sim-core, render-core, headless, and native-app. The
  wasm build passes. In-browser load, errors, rescale, and the scenario switch
  were checked in Chrome (WebGPU). Headless energy drift over 200 steps at
  N=1000: tilted flyby 1.1e-4, minor merger 9.3e-5, cluster + black hole 7e-7.
  The native renders of the tilted flyby at t=0/25/50 and the cluster + black
  hole at t=40 show the expected structure (the cluster stripped into a stream
  along its orbit).

## Known issues (found during step 1, not caused by it)

1. **SPH + self-gravity blows up at peak compression.** The built-in Evrard
   collapse goes from dE/E 6e-4 to ~1e4 around t≈1 at N=1000 and N=5000, and
   colliding gas clouds do the same. `tests/evrard_collapse.rs` only runs to
   t=0.5, so it never sees this. **A separate session is already working on
   it** (fix + extended test); don't duplicate it. Until it lands, no example
   spec uses gas.
2. ~~**Native-app energy readout is wrong after the first second.** The sim
   thread switches to `diagnostics::compute_fast` (U=0), so the UI shows
   E = K and dE/E ≈ +2. Headless diagnostics are correct.~~ Fixed by
   workstream C ([ADR 012](adr/012-native-energy-diagnostics.md)).
3. **Disks look sparse.** Particles have equal mass, so dark matter gets ~91% of
   a default galaxy's particles. 40K particles gives ~1,300 disk stars per
   galaxy.
4. ~~**The native-app initial snapshot runs O(N²) `diagnostics::compute`**, which
   takes seconds at N≥40K.~~ Fixed by workstream C (ADR 012).

## Parallel workstreams

Grouped by the files they touch, so threads don't collide. Workstreams in the
same row share files; run those in sequence.

| # | Workstream | Main files | Notes |
|---|---|---|---|
| A | **Star/dark-matter particle split.** Optional `star_particle_fraction` (or similar) on `disk_galaxy`: lighter star particles, heavier halo particles. Measure disk heating (scale-height growth, virial drift) on an isolated galaxy over ~50 time units at a few ratios, then pick a default and document the trade-off. | `scenarios/galaxy_collision.rs` (`GalaxyModel`), `spec.rs`, `tests/` | Physics-only; headless is enough. Keep `GalaxyCollision` output unchanged when the field is absent. |
| B | **Telescope-like rendering.** Hide dark matter by default with a toggle; set brightness/size by particle type, not mass (needed once A makes masses unequal); render point masses as small bright points, not mass-scaled blobs. | `render-core/src/{particles.rs,color.rs}`, `native-app/src/render/`, `web-app/src/lib.rs`, `static/index.html` | Visual; verify with native screenshots (macOS only) and the browser demo. |
| C | **Native diagnostics fixes** (known issues 2 and 4): full energy at a sustainable cadence (or Barnes-Hut potential), cheap initial snapshot. | `native-app/src/sim_thread.rs`, `native-app/src/render/ui.rs` | Shares `native-app/src/render/` with B: run after B or coordinate. |
| D | **Headless PNG renderer.** CPU density splat (log scale, type colors, no DM) at chosen times, e.g. `--render-at 0,25,50 --render-dir …`, plus a JSON summary of diagnostics. | `crates/headless/` | Lets the step 2 loop and **cloud threads** see frames without a GPU or window. |
| E | **Step 2: `/universe` skill.** Description → spec → headless run → read the JSON diagnostics and frames → revise. Include encounter-design guidance from `specs/README.md`. | `.claude/skills/universe/` (new) | Depends on D for visuals in the cloud. It can start with diagnostics only. |
| F | **Book chapter on specs**: the format, derived defaults, Toomre geometry, with live demos. | `book/src/` (new chapter), `SUMMARY.md` | Independent. Follow the milestone tag conventions in `CLAUDE.md`. |

Later: step 3 (the public page) once E works, and the sim-time-dependent view
radius (the camera follows the expanding debris).

## Environment notes

- **Cloud threads can't run `native-app`.** It needs a macOS window server and
  GPU. Use headless (and D's renderer once it exists).
- Locally in a sandboxed Claude Code session, `native-app` and `wasm-pack`'s
  `wasm-opt` step need the sandbox bypassed (window-server XPC, and
  wasm-pack's cache directory).
- Browser check: `bash scripts/build_book.sh`, serve `artifacts/book`, open
  `/demos/full.html`, click **Spec…**. In a hidden browser pane,
  `requestAnimationFrame` pauses, so FPS readings there are meaningless.
- CI (`.github/workflows/ci.yml`) runs sim-core tests in release, clippy with
  `-D warnings` on sim-core/render-core/headless, the wasm build, and the
  book build.

## Conventions that matter here

From `CLAUDE.md`: f64 for simulation math, deterministic seeds, N-body units
(G=1), safe Rust, and no rendering/IO dependencies in `sim-core`. Make honest
astrophysics claims only (e.g. "cloud collapse", not "star formation"). Work on
feature branches and don't commit to `main`. Milestone tags go on the commit
that completes a milestone.
