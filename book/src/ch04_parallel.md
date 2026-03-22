# Parallel N-Body with Rayon

The Barnes-Hut algorithm reduced our force calculation from $O(N^2)$ to $O(N \log N)$. But at 500,000 particles, each force evaluation still takes ~650ms on a single core. Modern hardware offers another dimension of speedup: parallelism. The M5 Pro has 18 cores — can we use them?

## The Parallel Opportunity

Two operations dominate each timestep:

1. **Force computation**: for each particle, walk the octree and sum gravitational interactions. Each particle's walk is independent — it reads the shared tree and writes only to its own acceleration.
2. **Integration** (kick/drift): update each particle's velocity and position. Again, each particle's update is independent.

Both are **embarrassingly parallel**: the work decomposes into independent tasks with no communication between them. This is the best-case scenario for parallelism.

## Amdahl's Law

Before writing any parallel code, it's worth understanding the theoretical limit. Amdahl's law says:

$$S(p) = \frac{1}{(1 - f) + f/p}$$

where $f$ is the fraction of work that can be parallelized and $p$ is the number of cores. For our N-body simulation at large $N$, force computation is >99% of the work, so $f \approx 0.99$. With 18 cores:

$$S(18) = \frac{1}{0.01 + 0.99/18} \approx 14.4 \times$$

We can't expect more than ~14× speedup even with perfect parallelization. In practice, overhead from thread synchronization, cache effects, and load imbalance reduces this further. A 10× speedup on 18 cores would be good.

## Rayon: Work Stealing in Rust

[Rayon](https://docs.rs/rayon) is Rust's standard library for data parallelism. It provides parallel iterators — `par_iter()`, `par_iter_mut()` — that distribute work across a thread pool using **work stealing**.

Work stealing means each thread has a local queue of tasks. When a thread finishes its work, it steals tasks from other threads' queues. This automatically balances load without manual partitioning — critical for Barnes-Hut, where different particles have different tree-walk depths depending on their local density.

## Parallelizing Force Computation

Each particle's tree walk is independent: it reads particle positions (shared) and the immutable octree, and writes to its own acceleration slot. In Rust:

```rust
let accels: Vec<[f64; 3]> = (0..n)
    .into_par_iter()
    .map(|i| tree_walk_accel(&tree, positions[i], i))
    .collect();
```

The temporary `Vec<[f64; 3]>` buffer avoids the borrow conflict that would arise from writing into `particles.ax[i]` while reading `particles.x[j]` in parallel. After the parallel walk, we copy the buffer back — a trivial $O(N)$ operation.

**Why not compute in-place?** The `GravitySolver` trait takes `&mut Particles`. With rayon, we'd need simultaneous shared reads (positions, masses) and exclusive writes (accelerations). Rust's borrow checker prevents this on the same struct. The temporary buffer is the clean, safe solution.

## Parallelizing the Integrator

The leapfrog kick/drift loops are even simpler to parallelize. Each particle's update depends only on its own state:

```rust
// Half-kick: v += a * dt/2
p.vx.par_iter_mut().zip(&p.ax)
    .for_each(|(v, &a)| *v += a * half_dt);
```

Rayon's `par_iter_mut().zip()` automatically chunks the arrays across threads. The borrow checker ensures safety: `vx` is borrowed mutably, `ax` immutably — no aliasing possible.

## The Threshold

Parallel overhead isn't free. Rayon's thread pool dispatch, work queue management, and cache synchronization cost microseconds per invocation. For small $N$ (< 1000), this overhead exceeds the work itself, making parallel execution *slower* than sequential.

We use a simple threshold: parallelize only when $N \geq 1000$. Below that, the sequential path runs. This is a pragmatic choice — the two-body Kepler test (N=2) shouldn't pay for thread pool overhead on every timestep.

## The Borrow Checker as Safety Net

In C/C++ with OpenMP, data races in parallel force computation are a common source of bugs. If two threads write to the same acceleration slot (because the Newton's third law optimization applies forces to both particles), you get a race condition. These bugs are intermittent, architecture-dependent, and hard to reproduce.

In Rust, the compiler prevents this entirely. Rayon's `par_iter_mut()` guarantees that each element is accessed by exactly one thread. If you try to share mutable state, the code doesn't compile. This isn't just convenient — it eliminates an entire class of bugs that plague parallel N-body codes.

## Implementation

The parallel code lives in two places:

- [`barnes_hut.rs`](blob/m3/crates/sim-core/src/barnes_hut.rs): `compute_accelerations` uses `(0..n).into_par_iter()` for the tree walk, with a sequential fallback for N < 1000.
- [`integrator.rs`](blob/m3/crates/sim-core/src/integrator.rs): `LeapfrogKDK::step` uses `par_iter_mut().zip()` for kick/drift loops, also with a threshold.

Rayon is added as a dependency of `sim-core` (not feature-gated). The WASM target (M6) will gate it with `#[cfg]` when needed.
