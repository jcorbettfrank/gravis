# Introduction

This book walks through building a gravitational N-body simulator from scratch in Rust. Each chapter pairs a physics concept with the code that implements it, and every claim is backed by a reproducible test you can run yourself.

## What we're building

An N-body simulator solves this problem: given N masses with known positions, velocities, and masses at time $t$, compute their positions and velocities at time $t + \Delta t$ under mutual gravitational attraction. Then repeat.

That one-sentence description hides real depth. Doing it *correctly* requires:

- Choosing a force calculation algorithm that scales (Chapter 1; Barnes-Hut in a future chapter)
- Choosing a time integrator that doesn't silently corrupt your physics (Chapter 2)
- Generating initial conditions that represent real astrophysical systems (future chapter)
- Verifying everything against known analytical solutions (Appendix)

Future chapters will cover parallelism, GPU compute, rendering, and SPH gas dynamics as the project grows. This book is written alongside the code — new chapters ship when the features they describe are implemented and verified.

## Who this is for

You should be comfortable with:
- Basic programming (we use Rust, but explain Rust-specific patterns as they appear)
- Undergraduate physics (Newton's laws, energy conservation, basic mechanics)
- Vectors and calculus (dot products, gradients, integrals)

No prior astrophysics, numerical methods, or GPU programming experience is assumed.

## Conventions

All simulation code uses **N-body units** where $G = 1$, $M_{\text{total}} = 1$, $R_{\text{virial}} = 1$. This makes the dynamical time $t_{\text{dyn}} = 1$. Physical units (kpc, $M_\odot$, Myr) appear only when discussing specific astrophysical scenarios.

All simulation math uses **f64** (64-bit floating point). GPU rendering uses f32. This distinction matters and is discussed in Chapter 1.

All scenarios use **deterministic RNG seeds**. Every result in this book can be reproduced exactly by running the cited command.

## How to reproduce results

Every chapter includes commands you can run:

```bash
# Clone the repo
git clone https://github.com/jcorbettfrank/gravis.git
cd gravis

# Run all physics tests
cargo test -p sim-core --release

# Run a specific scenario
cargo run -p headless --release -- --scenario plummer -n 1000 --steps 5000

# Run benchmarks
cargo bench -p sim-core
```

## Project structure

```
crates/sim-core/    Pure simulation library (no rendering, no IO)
crates/headless/    CLI batch runner for benchmarks and artifacts
crates/native-app/  Real-time macOS renderer with egui HUD
book/               This book (mdBook + KaTeX)
artifacts/          Generated benchmarks, snapshots, plots, media
docs/               Plan and architecture decision records
```

The `native-app` crate provides a real-time 3D renderer for macOS. It runs the simulation on a background thread while rendering at 60fps with interactive camera controls and a diagnostic HUD.

![Plummer sphere (5000 particles)](../artifacts/media/m2_plummer.png)
![Kepler two-body orbit](../artifacts/media/m2_kepler.png)

Future milestones add `crates/web-app/` (WASM+WebGPU). See `docs/PLAN.md` for the roadmap.

The simulation core (`sim-core`) has zero dependencies on rendering or IO. It's a pure library that computes physics. The `headless` crate wraps it with a CLI for batch runs, snapshot generation, and diagnostic output. This separation is enforced by the Cargo workspace — `sim-core` literally cannot import windowing or GPU crates.
