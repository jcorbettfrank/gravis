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

## Why the Integrator Stays Sequential

The leapfrog kick/drift loops are embarrassingly parallel in principle — each particle's update depends only on its own state. But in practice, these loops are **memory-bound**: each element does one multiply and one add (~2 FLOPs). At 500K particles, the sequential loop takes ~2ms. Rayon's thread pool dispatch overhead for 6 separate `par_iter_mut().zip()` calls (3 kick + 3 drift) adds ~120ms — far exceeding the computation itself.

The lesson: parallelism only helps when the work per element is large enough to amortize dispatch overhead. The force computation (tree walk with ~100 operations per particle) benefits enormously from rayon. The kick/drift (2 operations per particle) does not.

## The Parallelism Threshold

For the force computation, rayon's dispatch overhead dominates at small $N$. We use a threshold: parallelize tree walks only when $N \geq 1000$. Below that, sequential iteration runs. This ensures the two-body Kepler test (N=2) doesn't pay thread pool overhead on every timestep.

## The Borrow Checker as Safety Net

In C/C++ with OpenMP, data races in parallel force computation are a common source of bugs. If two threads write to the same acceleration slot (because the Newton's third law optimization applies forces to both particles), you get a race condition. These bugs are intermittent, architecture-dependent, and hard to reproduce.

In Rust, the compiler prevents this entirely. Rayon's `par_iter_mut()` guarantees that each element is accessed by exactly one thread. If you try to share mutable state, the code doesn't compile. This isn't just convenient — it eliminates an entire class of bugs that plague parallel N-body codes.

## Implementation

The parallel code lives in two places:

- [`barnes_hut.rs`](blob/m3/crates/sim-core/src/barnes_hut.rs): `compute_accelerations` uses `(0..n).into_par_iter()` for the tree walk, with a sequential fallback for N < 1000.
- [`integrator.rs`](blob/m3/crates/sim-core/src/integrator.rs): `LeapfrogKDK::step` remains sequential — kick/drift loops are memory-bound and don't benefit from parallelism.

Rayon is added as a dependency of `sim-core` (not feature-gated). The WASM target (M6) will gate it with `#[cfg]` when needed.
