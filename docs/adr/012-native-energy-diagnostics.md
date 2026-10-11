# ADR 012: Native-App Energy Diagnostics

**Date**: 2026-10-11
**Status**: Accepted

## Decision

The native app measures total energy (K + U + thermal) for the whole run, on its own worker thread, at a cadence set by what each measurement costs:

- **Method by N.** N ≤ 50,000: exact softened pair sum. N > 50,000: Barnes-Hut tree estimate of U at θ = 0.3 (`diagnostics::compute_tree`). Pure-hydro scenarios (gravity off) report K + thermal only. The HUD labels which one it shows ("U exact", "U tree estimate (θ=0.3)") and the sim time it was taken at.
- **Off the sim thread.** The sim thread sends a copy of the particles to an `EnergyWorker` thread and keeps stepping; the result comes back over a channel. A measurement that takes seconds never freezes the frames.
- **Cadence.** The next measurement starts `max(1 s, 10 × last cost)` after one lands, so the worker is busy at most ~10% of the time.
- **Cheap first frame.** The initial snapshot goes out before any energy is computed; the t = 0 energy, which the dE/E readout uses as its reference, follows from the worker.

In sim-core, the exact pair sum in `diagnostics::compute` now runs rows in parallel (with the `parallel` feature) and adds the row sums in index order, so its result does not depend on the thread count. `BarnesHut::potential_energy` provides the tree estimate the same way.

## Context

Handoff known issues 2 and 4 (`docs/text-to-universe-handoff.md`):

- After the first second the sim thread switched to `diagnostics::compute_fast`, which sets U = 0, so the HUD showed E = K and dE/E ≈ +2. The energy plot was meaningless after t ≈ 1.
- The initial snapshot ran the single-threaded O(N²) `diagnostics::compute` on the sim thread before the first frame: several seconds at N ≥ 40K.

The text-to-universe loop (M8) leans on energy conservation as its physics check, so the live readout has to be right whenever a spec runs in the window.

## Measurement

`cargo bench -p sim-core --bench gravity -- potential_energy` (Plummer sphere, default softening), on a 4-core x86 cloud container. "Before" is `main` at `863324d`.

| N | exact, before (serial) | exact, after (parallel) | tree θ=0.3 |
|---|---|---|---|
| 5,000 | 45.7 ms | 14.4 ms | — |
| 20,000 | 784 ms | 201 ms | 286 ms |
| 50,000 | 4.41 s | 1.28 s | 1.07 s |
| 100,000 | (~18 s, extrapolated) | (~5.1 s, extrapolated) | 3.35 s |

Tree-estimate accuracy, |E_tree − E_exact| / |E| on the tilted-flyby spec (Barnes-Hut θ = 0.5 dynamics, sampled every 50 steps over 400 steps at N = 20K; single snapshots at 5K, 40K, 100K):

| θ | N = 5K | N = 20K (range over run) | N = 40K | N = 100K |
|---|---|---|---|---|
| 0.2 | 1.7e-5 | 1.7e-5 – 4.4e-5 | 1.1e-5 | 1.1e-5 |
| **0.3** | 1.3e-5 | 4.6e-5 – 1.1e-4 | 6.0e-5 | 2.6e-5 |
| 0.5 | 1.8e-4 | 2.0e-4 – 3.6e-4 | 2.6e-4 | 1.7e-4 |

Most of that error is a slowly varying offset, so it partly cancels in dE/E: over the 20K run the θ = 0.3 drift tracked the exact drift to within ~3e-5 (exact −1.6e-4 at step 400, tree −1.3e-4). `tests/barnes_hut.rs` checks θ = 0 against the exact sum (1e-12) and θ = 0.3 on a 5K Plummer sphere (< 1e-4).

## Alternatives considered

- **Tree estimate at all N.** On this hardware the parallel exact sum is as fast as the θ = 0.3 tree up to ~40K and has no approximation error, so exact stays the default where it's affordable. 50K is where the two cost about the same; above it exact grows as N² and the tree as N log N.
- **θ = 0.5 for the tree (the force default).** 3–4× cheaper, but its 2–4e-4 error is as large as the real drift of a good run (~1e-4), so dE/E would mostly show tree noise.
- **Measure on the sim thread.** Simpler, but at 100K a measurement takes seconds, and the render would freeze for that long every interval.
- **Reuse the force pass to get φ.** The force walk could accumulate each particle's potential alongside its acceleration, but that changes the `GravitySolver` trait and adds work to every step (and every scenario) to serve a readout needed every few seconds.

## Consequences

- At N > 50K the dE/E readout has a noise floor of ~1e-4. The HUD says it's a tree estimate; a drift below that floor needs headless `diagnostics::compute` (exact) to confirm.
- Measurements are sparse at large N: every ~10 s at 50K and ~30 s at 100K on 4 cores (faster on more cores). The energy plot stays honest because each point is plotted at the sim time it was measured.
- The worker shares the rayon pool with the gravity solver, so the sim slows by up to ~10% while a measurement runs.
- Headless `diagnostics::compute` also gets the parallel speedup. Its results change in the last few bits (summation order); no test or artifact depends on that.
