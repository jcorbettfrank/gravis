# gravis

A gravitational N-body simulator written in Rust, targeting Apple Silicon. Built alongside an accompanying book that teaches the physics and code together.

**[Read the book →](https://jcorbettfrank.github.io/gravis/)**

## What it does

Simulates N masses evolving under mutual gravitational attraction. The simulation core is a pure Rust library (`sim-core`) with no rendering or IO dependencies — verified against analytical solutions (Kepler orbits, Plummer virial equilibrium, conservation laws).

Later milestones add Barnes-Hut O(N log N), a macOS renderer (wgpu), WebGPU browser target, and SPH gas dynamics. See [`docs/PLAN.md`](docs/PLAN.md) for the full roadmap.

## Prerequisites

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install mdbook mdbook-katex  # for building the book locally
```

## Usage

```bash
# Run all physics tests
cargo test -p sim-core

# Headless simulation run
cargo run -p headless --release -- --scenario plummer --particles 1000 --steps 10000

# Run benchmarks
cargo bench -p sim-core

# Build the book locally
mdbook build book/
# Output: artifacts/book/index.html
```

## Project structure

```
crates/sim-core/    Pure simulation library (no rendering, no IO)
crates/headless/    CLI batch runner for benchmarks and artifacts
book/               mdBook source (physics + code walkthroughs)
artifacts/          Generated benchmarks, snapshots, plots
docs/               Plan and architecture decision records
```

## Conventions

- `f64` for all simulation math; `f32` only for GPU buffer packing
- N-body units: G=1, M=1, R=1
- Deterministic RNG seeds — same parameters always produce the same result
- No unsafe code
