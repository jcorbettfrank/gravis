# Rust N-Body Astrophysics Simulator

## Context

Build a gravitational N-body simulation in Rust that pushes an M5 Pro (18 cores, 64GB), with an accompanying mdBook teaching the physics and code together. The simulation core is a library crate (`sim-core`) fully decoupled from rendering. Lessons are artifact-driven: each milestone produces reproducible tests, benchmarks, snapshots, and generated media — not a parallel writing project.

Later milestones add SPH gas dynamics, GPU compute, and a WebGPU browser target. But the MVP is a headless, correct, benchmarked gravity solver with one polished native macOS visualization.

**Astrophysics scope honesty**: We model what we model. No sub-grid physics (cooling, sink particles, star formation recipes) means we call it "cloud collapse / fragmentation," not "star formation." No embedded perturber means no gap formation claims. No star-formation model means gas compression and shocks, not starburst.

---

## Prerequisites

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install mdbook mdbook-katex
# Later: rustup target add wasm32-unknown-unknown
```

---

## Workspace Structure

```
gravis/
  CLAUDE.md                          # Points to docs/PLAN.md, conventions, ADRs
  Cargo.toml                         # Workspace root

  crates/
    sim-core/                        # Library crate: pure simulation, no rendering, no IO
      Cargo.toml
      src/
        lib.rs
        particle.rs                  # SoA storage (f64 positions/velocities/accelerations)
        gravity.rs                   # Force calculation trait + brute-force O(N²)
        octree.rs                    # Barnes-Hut octree
        barnes_hut.rs                # Tree-walk force calc with θ parameter
        integrator.rs                # KDK leapfrog, later Yoshida 4th-order
        diagnostics.rs               # Energy, momentum, angular momentum, COM drift
        snapshot.rs                  # Serialize/deserialize simulation state
        units.rs                     # Physical constants, unit system
        scenario.rs                  # Scenario trait + initial condition generators
        scenarios/
          plummer_sphere.rs          # Equilibrium test
          two_body.rs                # Kepler orbit verification
          cold_collapse.rs           # Uniform sphere, zero velocity
          galaxy_collision.rs        # Disk galaxies (later: + dark matter halo)
      tests/
        kepler.rs                    # Two-body orbital period, energy conservation
        plummer.rs                   # Virial equilibrium 2K/|U| ≈ 1
        conservation.rs              # Energy, momentum, angular momentum, COM drift
        scaling.rs                   # O(N²) vs O(N log N) timing verification
      benches/
        gravity.rs                   # Brute-force vs Barnes-Hut at various N

    native-app/                      # Binary crate: macOS renderer + UI
      Cargo.toml
      src/
        main.rs                      # Entry point, CLI (clap), launch app
        app.rs                       # winit ApplicationHandler + wgpu init
        sim_thread.rs                # Simulation thread, snapshot exchange
        render/
          mod.rs
          renderer.rs                # wgpu render pipeline
          camera.rs                  # Orbital camera (orbit/pan/zoom)
          particles.rs               # Instanced billboard rendering
          bloom.rs                   # 3-pass bloom post-processing
          ui.rs                      # egui overlay (diagnostics, controls)
          shaders/
            particle.wgsl
            bloom_threshold.wgsl
            bloom_blur.wgsl
            bloom_composite.wgsl
        input/
          mod.rs                     # Camera + simulation controls

    headless/                        # Binary crate: CLI batch runner
      Cargo.toml
      src/
        main.rs                      # Run sim to target time, write snapshots + CSV

    # Later crates:
    # web-app/                       # WASM + WebGPU target
    # sph/                           # SPH extension to sim-core

  docs/
    PLAN.md                          # This plan (persistent, version-controlled)
    adr/
      001-workspace-layout.md        # Why workspace split
      002-f64-simulation.md          # Why f64 in solver, f32 in GPU
      003-integrator-choice.md       # Why leapfrog (symplectic) over RK4
      # Added as decisions are made

  book/                              # mdBook source
    book.toml                        # KaTeX enabled
    src/
      SUMMARY.md
      intro.md
      ch01_gravity.md                # N-body problem, softening, O(N²)
      ch02_integrators.md            # Hamiltonians, symplectic geometry, leapfrog
      ch03_barnes_hut.md             # Octrees, multipole approximation, θ
      ch04_parallel.md               # Rayon, work stealing, Amdahl's law
      ch05_scenarios.md              # Galaxy dynamics, tidal forces, Jeans instability
      ch06_rendering.md              # Instancing, bloom, HDR
      ch07_sph.md                    # Kernel interpolation, density, pressure
      ch08_shocks.md                 # Riemann problem, artificial viscosity
      ch09_gpu_compute.md            # Tiling, shared memory, Metal/WebGPU
      ch10_stellar_physics.md        # Blackbody, Wien's law, spectral types
      appendix_verification.md       # All test problems with analytical solutions

  artifacts/                         # Generated outputs (gitignored except examples)
    benchmarks/                      # CSV/JSON benchmark results
    snapshots/                       # Binary simulation snapshots
    plots/                           # Generated diagnostic plots
    media/                           # Screenshots, videos for the book
```

---

## Key Design Decisions

- **Workspace split**: `sim-core` has zero rendering dependencies. It's a pure library usable by `native-app`, `headless`, and later `web-app`. Enforced by Cargo dependency graph.
- **f64 simulation, f32 rendering**: The solver uses `f64` for positions, velocities, accelerations, and energy tracking. Only the GPU buffer packing (`GpuParticle`) and rendering use `f32`. This prevents misleading long-run orbital and self-gravitating tests.
- **Safe snapshot exchange**: Simulation thread sends snapshots to render thread via `std::sync::mpsc::channel` or `Arc<Mutex<Option<Snapshot>>>`. No unsafe `AtomicPtr`. Optimize only if profiling shows contention.
- **Deterministic seeds**: All scenarios use fixed RNG seeds by default for reproducibility. Every run with the same parameters produces the same result.
- **Headless is infrastructure**: Snapshot export and batch running are how we generate lesson artifacts, benchmarks, and plots. Not a stretch goal.

---

## Milestones

### M1: Headless N-Body Core
**Goal**: Correct, benchmarked gravity solver running headless with deterministic output.

**Build**:
1. Workspace `Cargo.toml` + `sim-core` + `headless` crates
2. `CLAUDE.md` pointing to `docs/PLAN.md` and conventions
3. `particle.rs`: SoA storage with f64 components
4. `gravity.rs`: Brute-force O(N²) with Plummer softening
5. `integrator.rs`: Kick-Drift-Kick leapfrog
6. `units.rs`: Gravitational constant, N-body units (G=1, M=1, R=1)
7. `diagnostics.rs`: Total energy (K+U), linear momentum, angular momentum, center-of-mass position/velocity
8. `snapshot.rs`: Binary serialization of full particle state + metadata (time, step count, scenario params)
9. `scenario.rs` + `plummer_sphere.rs` + `two_body.rs`: Scenario trait with deterministic seeded IC generation
10. `headless/main.rs`: CLI runner — select scenario, run to target time, write snapshots at intervals, print diagnostics to CSV
11. Integration tests: Kepler orbit (period, energy, angular momentum), Plummer virial equilibrium, conservation laws, COM drift
12. `docs/adr/001-workspace-layout.md`, `002-f64-simulation.md`, `003-integrator-choice.md`

**Acceptance criteria**:
- [ ] Physics: Two-body Kepler orbit conserves energy to <0.01% over 1000 periods (leapfrog). Angular momentum conserved to machine precision. COM stationary.
- [ ] Physics: Plummer sphere virial ratio 2K/|U| fluctuates around 1.0 for 100 dynamical times, no secular drift.
- [ ] Performance: Brute-force 10K particles completes 1000 steps in <30s on M5 Pro (single-threaded baseline).
- [ ] Artifacts: `artifacts/benchmarks/m1_brute_force.csv` (N vs wall-clock), `artifacts/snapshots/` with Plummer evolution, diagnostic CSV with energy/momentum time series.
- [ ] Docs: `docs/PLAN.md` committed, first 2 ADRs, `CLAUDE.md` in place.
- [ ] Tests: `cargo test -p sim-core` passes, `cargo bench -p sim-core` runs.

**Write lessons**: `ch01_gravity.md`, `ch02_integrators.md` — with generated energy conservation plots from the Kepler and Plummer tests, links to exact git tag `m1`.

---

### M2: Native macOS Renderer
**Goal**: Real-time 3D visualization of the validated sim-core on macOS.

**Build**:
1. `native-app` crate depending on `sim-core`
2. `app.rs`: winit `ApplicationHandler` + wgpu surface + device init
3. `sim_thread.rs`: Spawn `sim-core` on a thread, send snapshots via channel. Speed multiplier (1x/10x/100x). Pause/resume.
4. `particle.wgsl` + `particles.rs`: Instanced camera-facing billboard quads. f64→f32 conversion in the buffer packing.
5. `camera.rs`: Orbital camera — left-drag orbit, scroll zoom, smooth interpolation
6. `ui.rs`: egui overlay — fps, particle count, sim time, energy plot, speed slider, pause button
7. `renderer.rs`: Depth buffer, clear color, coordinate axes for orientation
8. Wire up Plummer sphere and two-body scenarios as selectable via CLI

**Acceptance criteria**:
- [ ] Physics: Renderer shows same simulation as headless (deterministic comparison possible by running both with same seed and comparing snapshots).
- [ ] Performance: 10K particles renders at 60fps while simulation runs concurrently.
- [ ] Artifacts: `artifacts/media/m2_plummer.png` screenshot, `artifacts/media/m2_kepler.png`.
- [ ] Docs: ADR for sim/render thread architecture.

**Write lessons**: None new — renderer is engineering, not physics. Update `intro.md` with project screenshots.

---

### M3: CPU Barnes-Hut
**Goal**: O(N log N) force calculation, scaling to 100K+ particles.

**Build**:
1. `octree.rs`: Recursive octree — insert, center-of-mass aggregation, bounding box
2. `barnes_hut.rs`: Tree-walk with θ parameter (default 0.5, configurable)
3. Parallelize force summation with `rayon::par_iter` (each particle walks tree independently)
4. Parallelize integration step with rayon
5. Add `--algorithm {brute-force, barnes-hut}` and `--theta` CLI flags
6. Benchmark suite: brute-force vs Barnes-Hut at N = 1K, 5K, 10K, 50K, 100K, 500K
7. Force accuracy test: compare Barnes-Hut forces to brute-force reference at θ = 0.3, 0.5, 0.7, 1.0

**Acceptance criteria**:
- [ ] Physics: Barnes-Hut forces agree with brute-force within 1% RMS at θ=0.5. Energy conservation comparable to brute-force for Plummer sphere.
- [ ] Performance: 100K particles at >30fps interactive. 500K completes 100 steps headless in <60s.
- [ ] Artifacts: `artifacts/benchmarks/m3_scaling.csv` (N vs time, brute-force vs BH), `artifacts/benchmarks/m3_accuracy.csv` (θ vs force error). Generated log-log scaling plot.
- [ ] Tests: Scaling test confirms O(N log N) slope. All M1 physics tests still pass with Barnes-Hut.

**Write lessons**: `ch03_barnes_hut.md` (octree diagrams, θ explanation, scaling plot from benchmarks), `ch04_parallel.md` (rayon, work stealing, speedup vs core count).

---

### M4: Showcase Scenario — Galaxy Collision
**Goal**: One visually stunning, physically motivated scenario.

**Build**:
1. `galaxy_collision.rs`: Two Milky Way-type disk galaxies
   - Exponential disk profile (stars) + Plummer bulge
   - Optional dark matter halo (NFW or Hernquist profile, collisionless particles)
   - Hyperbolic approach trajectory
   - Galactic units: kpc, 10^10 M_sun, ~10^8 yr
2. Mass-based particle coloring: bulge (yellow-orange), disk (blue-white), dark matter (dim, translucent)
3. `bloom.rs`: 3-pass bloom post-processing (threshold → separable Gaussian blur → additive composite)
4. HDR rendering + tone mapping
5. Adaptive particle sizing (mass-proportional)
6. Headless run: evolve collision for ~2 Gyr, write snapshots every 50 Myr
7. Generate replay video from snapshots

**Acceptance criteria**:
- [ ] Physics: Tidal tails and bridges form. Total energy conserved within 1% over full merger. Dark matter halos (if included) produce correct dynamical friction timescale.
- [ ] Performance: 200K particles (100K per galaxy) at >30fps interactive.
- [ ] Artifacts: `artifacts/media/m4_collision_sequence/` — snapshots at key moments. Bloom-rendered screenshots. Energy conservation plot over the merger.
- [ ] Visual: Looks like an astrophysics visualization, not a debug scatter plot.

**Write lessons**: `ch05_scenarios.md` (disk galaxy models, tidal dynamics, dynamical friction, dark matter halos). Uses collision screenshots and energy plots as figures.

---

### M5: mdBook Polish + Artifact Integration
**Goal**: Publishable lesson series backed by reproducible artifacts.

**Build**:
1. `book.toml` configured with KaTeX, custom CSS
2. All chapters (ch01–ch05) finalized with:
   - Equations rendered via KaTeX
   - Figures generated from `artifacts/` (plots, screenshots)
   - Code snippets linking to exact source files at git tag
   - Reproducibility instructions: `cargo run -p headless -- --scenario plummer --steps 10000 --snapshot-interval 100`
3. `appendix_verification.md` — all test problems with analytical solutions
4. `SUMMARY.md` table of contents
5. `mdbook build` produces static site

**Acceptance criteria**:
- [ ] All figures generated from `artifacts/`, not hand-drawn.
- [ ] Every code snippet references a real file path and git tag.
- [ ] `mdbook build` succeeds with no warnings.
- [ ] A reader can reproduce every result by running the cited commands.

---

### M6: Web Target (WASM + WebGPU)
**Goal**: Browser-playable demo compiled from the same codebase.

**Build**:
1. Feature-gate native-only code (`#[cfg(not(target_arch = "wasm32"))]` for rayon, file I/O)
2. `web-app` crate: winit web canvas + wgpu WebGPU backend
3. Scaled-down scenarios (10K–50K particles) for browser performance
4. `wasm-pack build --target web`
5. Static landing page: scenario picker + embedded canvas + link to mdBook
6. Single-threaded sim loop on web (no rayon), or `wasm-bindgen-rayon` if SharedArrayBuffer available

**Acceptance criteria**:
- [ ] Runs in Chrome/Edge with WebGPU enabled.
- [ ] 10K particles at 30fps in browser.
- [ ] Same physics results as native (verified by comparing snapshots at matching seeds/steps).

---

### M7: SPH Gas Dynamics
**Goal**: Gas particles with pressure, density, and viscosity.

**Build**:
1. `sph` crate or module in `sim-core` (TBD based on coupling needs)
2. Spatial hash neighbor search within smoothing radius h
3. Density summation: ρᵢ = Σⱼ mⱼ W(|rᵢ-rⱼ|, h), cubic spline kernel
4. Ideal gas EOS: P = (γ-1)ρu, γ=5/3
5. SPH acceleration: pressure gradient + Monaghan artificial viscosity
6. Energy equation: du/dt from PdV work + viscous heating
7. Adaptive smoothing length (~50 neighbors)
8. Particle types: star (gravity only) vs gas (gravity + SPH)
9. Scenarios:
   - `cold_collapse.rs` updated with gas: uniform gas sphere → fragmentation (not "star formation" — no cooling/sinks)
   - `protoplanetary.rs`: central star + Keplerian gas disk with temperature gradient → spiral structure (not gap formation — no embedded perturber)
10. Verification: Sod shock tube, Sedov-Taylor blast, Evrard collapse, optionally Kelvin-Helmholtz (exposes classic SPH tension instability)

**Acceptance criteria**:
- [ ] Sod shock: correct shock/rarefaction/contact positions vs analytical solution.
- [ ] Sedov blast: correct radius vs time scaling (R ∝ t^(2/5)).
- [ ] Evrard collapse: energy conservation and density profile match reference.
- [ ] Total energy conserved with SPH terms included.

**Write lessons**: `ch07_sph.md`, `ch08_shocks.md` — with generated Sod/Sedov/Evrard plots.

---

### M8: Advanced (stretch)
- GPU tiled brute-force compute shader (comparison/accelerator path, not replacing CPU BH)
- GPU Barnes-Hut (linearized octree in compute shader)
- Yoshida 4th-order integrator
- Adaptive individual timesteps
- Radiative cooling for gas
- Sink particles (enables actual "star formation" claims)
- Dark matter halo live generation
- Particle trails
- Simulation replay viewer

---

## Verification Matrix

| Test | Milestone | Method | Expected |
|------|-----------|--------|----------|
| Energy conservation | M1+ | Track E_total | <0.01% drift over 1000 orbits (leapfrog) |
| Linear momentum | M1+ | Track p_total | Conserved to machine precision |
| Angular momentum | M1+ | Track L_total | Conserved to machine precision |
| COM drift | M1+ | Track x_com, v_com | Stationary to machine precision |
| Kepler orbit | M1 | 2-body, e=0.5, 1000 periods | Correct period, no precession |
| Virial equilibrium | M1 | Plummer sphere, 2K/\|U\| | Fluctuates around 1.0, no drift |
| BH force accuracy | M3 | Compare to brute-force | <1% RMS at θ=0.5 |
| BH scaling | M3 | Time vs N, log-log | O(N log N) slope |
| Sod shock tube | M7 | 1D SPH vs analytical | Correct discontinuity positions |
| Sedov blast | M7 | Spherical, R vs t | R ∝ t^(2/5) |
| Evrard collapse | M7 | Adiabatic sphere | Energy + density match reference |
| Kelvin-Helmholtz | M7 | Shear instability (optional) | Exposes SPH tension instability |

---

## Artifact Generation

Each milestone produces and commits:
- `artifacts/benchmarks/mN_*.csv` — timing and accuracy data
- `artifacts/snapshots/mN_*/` — binary simulation state at key moments
- `artifacts/plots/mN_*/` — generated diagnostic plots (energy, scaling, etc.)
- `artifacts/media/mN_*/` — screenshots or rendered frames for the book

Plots generated via headless runs + a simple plotting script (Python matplotlib or Rust `plotters` crate — TBD).

---

## Files to Create First (M1)

1. `CLAUDE.md` — project conventions, pointer to `docs/PLAN.md` and `docs/adr/`
2. `Cargo.toml` — workspace root
3. `crates/sim-core/Cargo.toml` + `src/lib.rs`
4. `crates/sim-core/src/particle.rs` — f64 SoA storage
5. `crates/sim-core/src/gravity.rs` — brute-force with softening
6. `crates/sim-core/src/integrator.rs` — KDK leapfrog
7. `crates/sim-core/src/units.rs` — G, unit system
8. `crates/sim-core/src/diagnostics.rs` — energy, momentum, angular momentum, COM
9. `crates/sim-core/src/snapshot.rs` — serialize/deserialize state
10. `crates/sim-core/src/scenario.rs` — trait + registry
11. `crates/sim-core/src/scenarios/plummer_sphere.rs`
12. `crates/sim-core/src/scenarios/two_body.rs`
13. `crates/headless/Cargo.toml` + `src/main.rs` — CLI batch runner
14. `crates/sim-core/tests/kepler.rs`
15. `crates/sim-core/tests/plummer.rs`
16. `crates/sim-core/tests/conservation.rs`
17. `crates/sim-core/benches/gravity.rs`
18. `docs/PLAN.md` — copy of this plan
19. `docs/adr/001-workspace-layout.md`
20. `docs/adr/002-f64-simulation.md`
21. `docs/adr/003-integrator-choice.md`
