# ADR 004: Sim/Render Thread Architecture

**Date**: 2026-03-21
**Status**: Accepted

## Decision

The simulation runs on a dedicated OS thread, communicating with the render thread (main thread) via `std::sync::mpsc::channel`. The sim thread sends `RenderSnapshot` structs containing f32 positions, masses, and diagnostic summaries. The render thread sends `SimCommand` enums back for pause/resume/speed control.

## Context

winit requires the event loop to run on the main thread (macOS restriction). The simulation is compute-bound (O(N^2) for brute-force gravity). Running simulation on the main thread would block rendering and make the UI unresponsive. We need both to run concurrently at independent rates.

## Alternatives Considered

1. **`Arc<Mutex<Particles>>`** — Rejected. The sim thread would lock the mutex for the entire step duration (~3ms for 10K particles), blocking the render thread from reading positions. Even with try_lock, this creates frame drops.

2. **Double-buffering with atomics** — Unnecessary complexity. Requires careful memory ordering, adds cognitive overhead, and is harder to reason about correctness. No meaningful performance advantage over channels for our throughput needs (~60 snapshots/sec).

3. **async/tokio** — The simulation is CPU-bound, not IO-bound. Async runtimes add scheduling overhead without benefit. `std::thread` gives us a dedicated OS thread with predictable, priority-scheduled execution.

## Design Details

**Sim thread → Render thread**: `mpsc::channel::<RenderSnapshot>()`. The sim thread sends at most ~60 snapshots per second (wall-clock throttled). The render thread drains all pending messages each frame and keeps only the latest. This means the channel never grows unbounded.

**Render thread → Sim thread**: `mpsc::channel::<SimCommand>()`. Commands include Pause, Resume, SetSpeed(f64), and Stop. The sim thread checks for commands via `try_recv` each iteration.

**f64→f32 conversion** happens on the sim thread when building the snapshot. The render thread never sees f64 data. This keeps the render path simple and avoids shipping 8 bytes per coordinate when 4 suffice.

**Diagnostics** (energy, virial ratio) are computed on the sim thread at ~1 second intervals, not every snapshot, to avoid doubling the O(N^2) cost.

**Determinism**: The sim thread uses the exact same code path as the headless runner — same Scenario::generate(), same BruteForce, same LeapfrogKDK. Snapshot throttling does not affect the simulation trajectory.

## Consequences

- The render thread may display data up to ~16ms stale. This is imperceptible.
- No `unsafe` code is needed. All communication uses safe Rust channels.
- The sim thread is fully deterministic regardless of rendering frame rate.
- Speed multiplier works by advancing sim_time to match `wall_elapsed * speed_factor`. At very high speeds, the sim thread simply runs as fast as possible.
