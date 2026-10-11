# gravis

Gravitational N-body simulation in Rust, targeting Apple M5 Pro.

## Project Layout

- **`docs/PLAN.md`** — master plan with milestones and acceptance criteria
- **`docs/adr/`** — architecture decision records
- **`crates/sim-core/`** — pure simulation library (no rendering, no IO)
- **`crates/headless/`** — CLI batch runner for benchmarks and snapshot generation
- **`crates/render-core/`** — platform-independent rendering (pipelines, camera, shaders)
- **`crates/native-app/`** — real-time macOS renderer + egui HUD
- **`crates/web-app/`** — browser WASM demo (wasm-pack + web-sys)
- **`book/`** — mdBook lessons (physics + code walkthroughs)
- **`specs/`** — JSON scenario specs; format reference in `specs/README.md`
- **`.claude/skills/universe/`** — `/universe` skill: description → spec → headless run → check → revise
- **`artifacts/`** — generated benchmarks, snapshots, plots, media

## Conventions

- **f64** for all simulation math (positions, velocities, accelerations, energy)
- **f32** only for GPU buffer packing and rendering
- **Deterministic seeds** — all scenarios use fixed RNG seeds for reproducibility
- **N-body units** — G=1, M=1, R=1 in sim-core
- **Safe Rust** — no unsafe. Use channels or `Arc<Mutex>` for thread communication.
- **Workspace**: `sim-core` must have zero rendering/IO dependencies

## Commands

```bash
# Run all sim-core tests
cargo test -p sim-core

# Run benchmarks
cargo bench -p sim-core

# Headless simulation run
cargo run -p headless -- --scenario plummer --particles 1000 --steps 10000

# Real-time renderer (macOS)
cargo run -p native-app --release -- --scenario plummer -n 5000
cargo run -p native-app --release -- --scenario two-body --eccentricity 0.3
cargo run -p native-app --release -- --scenario plummer -n 5000 --speed 10.0

# Capture a screenshot without interactive window
cargo run -p native-app --release -- --scenario plummer -n 5000 --screenshot artifacts/media/plummer.png

# Dark matter is hidden by default (toggle with D in the window)
cargo run -p native-app --release -- --scenario galaxy-collision --show-dark-matter

# Run a JSON scenario spec (headless or native; -n rescales it)
cargo run -p headless --release -- --spec specs/tilted-flyby.json --algorithm barnes-hut --steps 2000
cargo run -p native-app --release -- --spec specs/tilted-flyby.json --algorithm barnes-hut

# Screenshot at a given sim time (runs flat out to it, then captures)
cargo run -p native-app --release -- --spec specs/tilted-flyby.json -n 40000 --algorithm barnes-hut --screenshot artifacts/media/flyby.png --screenshot-time 25

# Build WASM web app
cd crates/web-app && wasm-pack build --target web --release

# Build book with live demos
bash scripts/build_book.sh

# Build in release mode
cargo build --release
```

## Milestone Tags

Each milestone is marked with a git tag (`m1`, `m2`, ...) applied after all code and book content for that milestone lands on `main`. Book chapters link to source files at the exact tag (e.g. `blob/m1/crates/sim-core/src/gravity.rs`) for reproducibility. CI and infrastructure commits are not milestone boundaries — apply tags to the commit that completes the milestone's acceptance criteria.

## Astrophysics Scope

We model what we model. Honest claims only:
- Without cooling/sink particles: "cloud collapse / fragmentation", not "star formation"
- Without embedded perturber: no gap formation claims
- Without star-formation model: "gas compression and shocks", not "starburst"
