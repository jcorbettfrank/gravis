# Rust N-Body Astrophysics Simulator

## Context

An interactive textbook on gravitational N-body simulation, built in Rust and targeting Apple M5 Pro. The primary deliverable is an mdBook where every chapter ships alongside the code it teaches, backed by reproducible artifacts (benchmarks, snapshots, plots). Each milestone produces both a working simulation capability and its corresponding book chapter — the code exists to make the book's claims verifiable.

The simulation core is a library crate (`sim-core`) fully decoupled from rendering. A browser-playable WebGPU target embeds live interactive demos directly in the book, so readers can experiment with every concept as they learn it. Later milestones add SPH gas dynamics (astrophysical SPH in Rust does not exist in the open-source ecosystem) and GPU compute.

**Astrophysics scope honesty**: We model what we model. No sub-grid physics (cooling, sink particles, star formation recipes) means we call it "cloud collapse / fragmentation," not "star formation." No embedded perturber means no gap formation claims. No star-formation model means gas compression and shocks, not starburst.

---

## Prerequisites

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install mdbook mdbook-katex
rustup target add wasm32-unknown-unknown  # For M4 web target
cargo install wasm-pack                   # For M4 web target
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
          galaxy_collision.rs        # Disk galaxies + dark matter halo
      tests/
        kepler.rs                    # Two-body orbital period, energy conservation
        plummer.rs                   # Virial equilibrium 2K/|U| ≈ 1
        conservation.rs              # Energy, momentum, angular momentum, COM drift
        barnes_hut.rs                # Force accuracy vs brute-force
        scaling.rs                   # O(N²) vs O(N log N) timing verification
      benches/
        gravity.rs                   # Brute-force vs Barnes-Hut at various N

    render-core/                     # Shared rendering: particle pipeline, camera, shaders
      Cargo.toml
      src/
        lib.rs
        particles.rs                 # Instanced billboard rendering
        camera.rs                    # Orbital camera math
        gpu_types.rs                 # GPU buffer types (f32)
      shaders/
        particle.wgsl
        bloom_threshold.wgsl
        bloom_blur.wgsl
        bloom_composite.wgsl

    native-app/                      # Binary crate: macOS renderer + UI
      Cargo.toml
      src/
        main.rs                      # Entry point, CLI (clap), launch app
        app.rs                       # winit ApplicationHandler + wgpu init
        sim_thread.rs                # Simulation thread, snapshot exchange
        render/
          mod.rs
          renderer.rs                # wgpu render pipeline (uses render-core)
          ui.rs                      # egui overlay (diagnostics, controls)

    web-app/                         # WASM + WebGPU target
      Cargo.toml
      src/
        lib.rs                       # wasm-bindgen entry point, single-threaded sim loop
      static/
        index.html                   # Landing page, scenario picker, canvas

    headless/                        # Binary crate: CLI batch runner
      Cargo.toml
      src/
        main.rs                      # Run sim to target time, write snapshots + CSV

  docs/
    PLAN.md                          # This plan (persistent, version-controlled)
    adr/
      001-workspace-layout.md        # Why workspace split
      002-f64-simulation.md          # Why f64 in solver, f32 in GPU
      003-integrator-choice.md       # Why leapfrog (symplectic) over RK4
      004-sim-render-threads.md      # Channel-based sim/render communication
      005-barnes-hut-design.md       # Octree + tree-walk design choices

  book/                              # mdBook source
    book.toml                        # KaTeX enabled
    src/
      SUMMARY.md
      intro.md
      ch01_gravity.md                # N-body problem, softening, O(N²)
      ch02_integrators.md            # Hamiltonians, symplectic geometry, leapfrog
      ch03_barnes_hut.md             # Octrees, multipole approximation, θ
      ch04_parallel.md               # Rayon, work stealing, Amdahl's law
      ch05_rendering.md              # wgpu, instanced billboards, camera, web target
      ch06_scenarios.md              # Galaxy dynamics, tidal forces, dark matter
      ch07_sph.md                    # Kernel interpolation, density, pressure
      ch08_shocks.md                 # Riemann problem, artificial viscosity
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
- **Book-first milestones**: Every milestone ships code AND its book chapter(s). The book is the primary deliverable; the simulation makes the book's claims verifiable.

---

## Milestones

### M1: Headless N-Body Core ✓
**Goal**: Correct, benchmarked gravity solver running headless with deterministic output.

**Built**:
1. Workspace `Cargo.toml` + `sim-core` + `headless` crates
2. `particle.rs`: SoA storage with f64 components
3. `gravity.rs`: Brute-force O(N²) with Plummer softening
4. `integrator.rs`: Kick-Drift-Kick leapfrog
5. `diagnostics.rs`: Total energy (K+U), linear momentum, angular momentum, center-of-mass
6. `snapshot.rs`: Binary serialization of full particle state + metadata
7. `scenario.rs` + `plummer_sphere.rs` + `two_body.rs`: Scenario trait with deterministic seeded IC generation
8. `headless/main.rs`: CLI runner with snapshot/diagnostic export
9. Integration tests: Kepler orbit, Plummer virial equilibrium, conservation laws
10. ADRs 001–003

**Acceptance criteria**:
- [x] Physics: Two-body Kepler orbit conserves energy to <0.01% over 1000 periods. Angular momentum conserved to machine precision. COM stationary.
- [x] Physics: Plummer sphere virial ratio 2K/|U| fluctuates around 1.0, no secular drift.
- [x] Performance: Brute-force 10K particles completes 1000 steps in <30s on M5 Pro.
- [x] Tests: `cargo test -p sim-core` passes, `cargo bench -p sim-core` runs.

**Book**: `ch01_gravity.md`, `ch02_integrators.md`

---

### M2: Native macOS Renderer ✓
**Goal**: Real-time 3D visualization of the validated sim-core on macOS.

**Built**:
1. `native-app` crate with wgpu + winit + egui
2. Dedicated sim thread with snapshot channel + command channel
3. Instanced camera-facing billboard quads (WGSL shader)
4. Orbital camera, egui overlay, speed control, pause/resume
5. ADR 004 (sim/render thread architecture)

**Acceptance criteria**:
- [x] Physics: Renderer shows same simulation as headless.
- [x] Performance: 10K particles at 60fps with concurrent simulation.
- [x] Artifacts: Screenshots in `artifacts/media/`.

**Book**: Updated `intro.md` with screenshots and running instructions.

---

### M3: CPU Barnes-Hut ✓
**Goal**: O(N log N) force calculation, scaling to 100K+ particles.

**Built**:
1. `octree.rs`: Recursive octree with center-of-mass aggregation
2. `barnes_hut.rs`: Tree-walk with configurable θ parameter (default 0.5)
3. Rayon parallelism for force summation and integration
4. `--algorithm {brute-force, barnes-hut}` and `--theta` CLI flags
5. Benchmark suite: brute-force vs Barnes-Hut scaling
6. Force accuracy tests: Barnes-Hut vs brute-force reference
7. ADR 005 (Barnes-Hut design)

**Acceptance criteria**:
- [x] Physics: Barnes-Hut forces agree with brute-force within 1% RMS at θ=0.5.
- [x] Performance: O(N log N) scaling confirmed. All M1 physics tests pass with Barnes-Hut.
- [x] Artifacts: Scaling benchmarks and accuracy data.

**Book**: `ch03_barnes_hut.md`, `ch04_parallel.md`

---

### M4: Web Target + Rendering Chapter
**Goal**: Browser-playable demo with live interactive simulations embedded in the book.

**Why now**: wgpu already targets WebGPU from the same WGSL shaders. `sim-core` has zero platform deps. Barnes-Hut enables 10K–50K particles single-threaded in the browser. This is refactoring + plumbing — not new physics — and it unlocks interactive demos for every future chapter.

**Build**:
1. Extract `render-core` crate from `native-app` — particle pipeline, camera math, GPU types, WGSL shaders (everything platform-independent)
2. Refactor `native-app` to depend on `render-core`, keeping winit event loop + egui HUD + sim thread locally
3. `web-app` crate: WASM entry point via `wasm-pack`, single-threaded sim loop (no rayon), HTML controls for scenario/algorithm selection
4. Feature-gate rayon behind `#[cfg(not(target_arch = "wasm32"))]` in `sim-core`
5. mdBook embed system (iframe or preprocessor) — chapters include live WebGPU canvases with static screenshot fallback
6. Retroactively add interactive demos to Ch 1–4

**Acceptance criteria**:
- [ ] 5K particles at 30fps in Chrome/Edge with WebGPU.
- [ ] 50K particles interactive in browser with Barnes-Hut (single-threaded).
- [ ] `native-app` still works identically (regression test: same screenshots).
- [ ] At least one live demo embedded in the mdBook.
- [ ] Same physics results as native (verified by comparing snapshots at matching seeds/steps).

**Book**: `ch05_rendering.md` — wgpu architecture, instanced billboards, camera math, native-vs-web abstraction, WGSL shader walkthrough. Includes embedded "try it yourself" live demo.

---

### M5: Showcase Scenario — Galaxy Collision
**Goal**: Visually stunning, physically motivated scenario with interactive web demos.

**Build**:
1. `galaxy_collision.rs`: Two Milky Way-type disk galaxies
   - Exponential disk profile (stars) + Plummer bulge
   - Dark matter halo (NFW or Hernquist profile, collisionless particles)
   - Hyperbolic approach trajectory
   - Galactic units: kpc, 10^10 M_sun, ~10^8 yr
2. `cold_collapse.rs`: Uniform sphere, zero velocity — simple but dramatic
3. `bloom.rs` in `render-core`: 3-pass bloom post-processing (threshold → separable Gaussian blur → additive composite)
4. HDR rendering + tone mapping
5. Mass-based particle coloring: bulge (yellow-orange), disk (blue-white), dark matter (dim, translucent)
6. Update `web-app` with galaxy collision scenario and bloom

**Acceptance criteria**:
- [ ] Physics: Tidal tails and bridges form. Total energy conserved within 1% over full merger.
- [ ] Performance: 200K particles (100K per galaxy) at >30fps interactive (native). 20K+ with bloom at 30fps in browser.
- [ ] Visual: Looks like an astrophysics visualization, not a debug scatter plot.
- [ ] Artifacts: Bloom-rendered screenshots, energy conservation plot.

**Book**: `ch06_scenarios.md` — disk galaxy models, rotation curves, tidal dynamics, dynamical friction, dark matter halos, Jeans instability. Interactive galaxy collision demo embedded in chapter.

---

### M6: SPH Gas Dynamics
**Goal**: Gas particles with pressure, density, and viscosity. This is the project's main technical novelty — astrophysical SPH in Rust does not exist in the open-source ecosystem.

**Build**:
1. SPH module in `sim-core` (or separate `sph` crate if coupling is loose)
2. Tree-based neighbor search within smoothing radius h
3. Density summation: ρᵢ = Σⱼ mⱼ W(|rᵢ-rⱼ|, h), cubic spline kernel
4. Ideal gas EOS: P = (γ-1)ρu, γ=5/3
5. SPH acceleration: pressure gradient + Monaghan artificial viscosity
6. Energy equation: du/dt from PdV work + viscous heating
7. Adaptive smoothing length (~50 neighbors)
8. Particle types: star (gravity only) vs gas (gravity + SPH)
9. Scenarios:
   - `cold_collapse.rs` updated with gas: uniform gas sphere → fragmentation (not "star formation" — no cooling/sinks)
   - `protoplanetary.rs`: central star + Keplerian gas disk → spiral structure (not gap formation — no embedded perturber)
10. Verification: Sod shock tube, Sedov-Taylor blast, Evrard collapse, optionally Kelvin-Helmholtz
11. Gas rendering: color particles by temperature/density in `render-core`
12. Update `web-app` with SPH scenarios

**Acceptance criteria**:
- [ ] Sod shock: correct shock/rarefaction/contact positions vs analytical solution.
- [ ] Sedov blast: correct radius vs time scaling (R ∝ t^(2/5)).
- [ ] Evrard collapse: energy conservation and density profile match reference.
- [ ] Total energy conserved with SPH terms included.
- [ ] At least one SPH demo interactive in the browser.

**Book**: `ch07_sph.md` (kernel interpolation, density estimation, pressure forces, variational derivation), `ch08_shocks.md` (Riemann problem, artificial viscosity, Sedov blast, Evrard collapse). Interactive Sod shock tube and Sedov blast demos in the book.

---

### M7: Polish + Stretch Goals
**Goal**: Publishable state. All chapters finalized, advanced topics as appendices.

**Code (any subset)**:
- GPU tiled brute-force compute shader
- GPU Barnes-Hut (linearized octree in compute shader)
- Yoshida 4th-order symplectic integrator
- Adaptive individual timesteps
- Radiative cooling for gas
- Sink particles (enables actual "star formation" claims)
- Particle trails
- Simulation replay viewer (load snapshots, scrub timeline)
- Touch-friendly mobile controls for web demos

**Book**:
- Appendix B: GPU Compute (if implemented)
- Appendix C: Stellar Physics (blackbody, Wien's law, mass-to-color rendering)
- Final polish pass: consistent notation, cross-references, every figure regenerated, every code link verified against git tags
- Landing page / project homepage with embedded demos

---

## Book Table of Contents

```
Introduction

Part I: Foundations
  Ch 1: The N-Body Problem              (M1) ✓
  Ch 2: Symplectic Integrators           (M1) ✓

Part II: Scaling Up
  Ch 3: The Barnes-Hut Algorithm         (M3) ✓
  Ch 4: Parallel Computing               (M3) ✓

Part III: Seeing It
  Ch 5: Rendering Particles              (M4)

Part IV: Astrophysical Scenarios
  Ch 6: Galaxy Models & Collisions       (M5)

Part V: Gas Dynamics
  Ch 7: Smoothed Particle Hydrodynamics  (M6)
  Ch 8: Shocks & Verification            (M6)

Appendices
  A: Verification Tests                  (updated each milestone)
  B: GPU Compute                         (M7 stretch)
  C: Stellar Physics                     (M7 stretch)
```

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
| Sod shock tube | M6 | 1D SPH vs analytical | Correct discontinuity positions |
| Sedov blast | M6 | Spherical, R vs t | R ∝ t^(2/5) |
| Evrard collapse | M6 | Adiabatic sphere | Energy + density match reference |
| Kelvin-Helmholtz | M6 | Shear instability (optional) | Exposes SPH tension instability |

---

## Artifact Generation

Each milestone produces and commits:
- `artifacts/benchmarks/mN_*.csv` — timing and accuracy data
- `artifacts/snapshots/mN_*/` — binary simulation state at key moments
- `artifacts/plots/mN_*/` — generated diagnostic plots (energy, scaling, etc.)
- `artifacts/media/mN_*/` — screenshots or rendered frames for the book

Plots generated via headless runs + a simple plotting script (Python matplotlib or Rust `plotters` crate — TBD).
