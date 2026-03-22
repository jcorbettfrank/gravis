# The Barnes-Hut Algorithm

In Chapter 1, we computed gravitational forces by summing over every particle pair — $O(N^2)$ work per timestep. At $N = 10{,}000$, that's about 50 million interactions per step. By $N = 100{,}000$, it's 5 billion. The brute-force approach hits a wall.

The [Barnes-Hut algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) reduces this to $O(N \log N)$ by exploiting a physical insight: a distant cluster of particles looks almost the same as a single particle at the cluster's [center of mass](https://en.wikipedia.org/wiki/Center_of_mass). We only need to resolve individual particles when they're nearby.

## The Key Idea

Imagine computing the gravitational pull of the Andromeda galaxy on you. You could sum the force from each of its trillion stars individually. Or you could note that Andromeda is 2.5 million light-years away and treat it as a single point mass. The error from this approximation is negligible — Andromeda's diameter (~220,000 light-years) is small compared to its distance.

The Barnes-Hut algorithm applies this reasoning recursively: partition space into a hierarchy of cells, aggregate each cell's mass at its center of mass, and replace the full sum with an approximate sum that uses aggregate cells where possible. This is a form of [multipole expansion](https://en.wikipedia.org/wiki/Multipole_expansion) truncated at the monopole term — higher-order variants (quadrupole, octupole) exist but the monopole approximation is sufficient for most astrophysical N-body work.

## The Octree

We partition 3D space using an [**octree**](https://en.wikipedia.org/wiki/Octree): a tree where each node represents a cubic region and has up to 8 children (one per octant). The 2D analog is a [quadtree](https://en.wikipedia.org/wiki/Quadtree). Construction works by inserting particles one at a time:

1. Start with a single root node encompassing all particles.
2. If a node is empty, the inserted particle becomes its sole occupant.
3. If a node already contains a particle, subdivide it into 8 children, re-insert the existing particle into the correct child, then insert the new particle.
4. If a node is already subdivided (internal node), insert into the correct child octant.

Each node maintains the total mass and center of mass of all particles it contains, updated incrementally during insertion.

**Octant indexing.** Given a node centered at $(c_x, c_y, c_z)$, a particle at $(p_x, p_y, p_z)$ maps to octant:

$$\text{octant} = [p_x \geq c_x] \;|\; [p_y \geq c_y] \ll 1 \;|\; [p_z \geq c_z] \ll 2$$

This produces indices 0–7 using three bit operations — no conditionals.

**Degenerate cases.** Two particles at the same position would cause infinite subdivision. We impose a depth limit (64 levels), beyond which overlapping particles share a cell.

## The Opening Angle Criterion

For each particle $i$, we walk the tree to compute its acceleration. At each internal node, we decide whether to "open" the node (recurse into children) or treat it as a single body:

$$\frac{s}{d} < \theta$$

where $s$ is the cell's side length, $d$ is the distance from particle $i$ to the cell's center of mass, and $\theta$ is the **opening angle parameter**.

- **$\theta = 0$**: every node is opened → exact (brute-force) result
- **$\theta = 0.5$**: the standard choice, balancing speed and accuracy
- **$\theta = 1.0$**: aggressive approximation, fast but imprecise

When the criterion is satisfied, we compute:

$$\vec{a}_i \mathrel{+}= \frac{G \, M_\text{cell} \, \vec{r}}{\left(|\vec{r}|^2 + \epsilon^2\right)^{3/2}}$$

using the cell's total mass $M_\text{cell}$ and center of mass, exactly as we would for a single particle (with softening).

## Force Accuracy

How much error does the approximation introduce? We measure the RMS relative force error by comparing Barnes-Hut accelerations to the brute-force reference on a 1,000-particle Plummer sphere (test: [`tests/barnes_hut.rs`](https://github.com/jcorbettfrank/gravis/blob/m3/crates/sim-core/tests/barnes_hut.rs)):

| $\theta$ | RMS Error | Max Error |
|-----------|-----------|-----------|
| 0.3 | 0.11% | 0.87% |
| 0.5 | 0.49% | 3.59% |
| 0.7 | 1.35% | 9.99% |
| 1.0 | 3.32% | 23.9% |

At $\theta = 0.5$, the typical force error is under 0.5% — well within acceptable bounds for N-body simulation. The error grows monotonically with $\theta$, as expected.

## Scaling

The fundamental payoff of Barnes-Hut is scaling (benchmark: [`benches/gravity.rs`](https://github.com/jcorbettfrank/gravis/blob/m3/crates/sim-core/benches/gravity.rs)). On an M5 Pro:

| N | Brute-force | Barnes-Hut ($\theta=0.5$) | Speedup |
|---|------------|--------------------------|---------|
| 1,000 | 1.4 ms | 0.4 ms | 3.5× |
| 5,000 | 37 ms | 2.1 ms | 18× |
| 10,000 | 152 ms | 4.7 ms | 32× |
| 50,000 | — | 35 ms | — |
| 100,000 | — | 86 ms | — |
| 500,000 | — | 651 ms | — |

Brute-force becomes impractical above $N \approx 10{,}000$. Barnes-Hut handles 500,000 particles in under a second per force evaluation — a regime where brute-force would take over a minute.

The log-log slope of the Barnes-Hut timing is approximately 1.2, consistent with $O(N \log N)$. The brute-force slope is 2.0, confirming $O(N^2)$.

![M3 Scaling Plot](../artifacts/plots/m3_scaling.png)

## Momentum Conservation

There's a trade-off hidden in this approach. The brute-force solver computes each pair interaction once and applies [Newton's third law](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion#Third_law) ($\vec{F}_{ij} = -\vec{F}_{ji}$), guaranteeing exact [momentum conservation](https://en.wikipedia.org/wiki/Conservation_of_momentum). Barnes-Hut computes each particle's acceleration independently — particle $i$ might approximate the force from particle $j$'s cell differently than particle $j$ approximates the force from particle $i$'s cell.

This means momentum is conserved only approximately, not to machine precision. In practice, with $\theta = 0.5$, the drift is small enough to be irrelevant over the timescales we simulate. But it's worth knowing: if a test expects machine-precision momentum conservation, use brute-force.

## Implementation Notes

Our implementation in Rust ([`octree.rs`](https://github.com/jcorbettfrank/gravis/blob/m3/crates/sim-core/src/octree.rs), [`barnes_hut.rs`](https://github.com/jcorbettfrank/gravis/blob/m3/crates/sim-core/src/barnes_hut.rs)) follows the design above:

- **Arena-allocated octree**: all nodes live in a flat `Vec<OctreeNode>` with `[u32; 8]` child indices. One allocation per tree build instead of thousands of per-node `Box` allocations. Safe Rust, no `unsafe`.
- **Tree rebuilt each step**: since particles move, the tree is rebuilt from scratch on every force evaluation. This is simpler than maintaining a dynamic tree and the build cost is $O(N \log N)$, dominated by the tree walk.
- **Parallel tree walk**: each particle walks the (immutable) tree independently — a natural fit for `rayon::par_iter`. We'll cover parallelization in Chapter 4.

## Usage

```bash
# Barnes-Hut with default theta=0.5
cargo run -p headless --release -- --scenario plummer -n 100000 --algorithm barnes-hut --steps 1000

# More accurate (theta=0.3)
cargo run -p headless --release -- --scenario plummer -n 100000 --algorithm barnes-hut --theta 0.3 --steps 1000

# Interactive renderer at 100K particles
cargo run -p native-app --release -- --scenario plummer -n 100000 --algorithm barnes-hut
```

## Live Demo

The Barnes-Hut algorithm makes this many particles possible in your browser. This demo runs 5,000 particles with $\theta = 0.5$ — try it with brute-force in the [full interactive demo](../index.html) to feel the difference.

<div class="live-demo">
  <iframe src="demos/plummer.html" width="100%" loading="lazy"
          title="Live Barnes-Hut Plummer sphere demo"></iframe>
  <p class="demo-fallback" style="display:none">
    <img src="images/m2_plummer.png" alt="Plummer sphere with Barnes-Hut">
    <em>Live demo requires a WebGPU-enabled browser (Chrome 113+, Edge 113+, Safari 18+).</em>
  </p>
</div>

## Further Reading

- [Barnes & Hut (1986)](https://ui.adsabs.harvard.edu/abs/1986Natur.324..446B) — the original paper introducing the hierarchical tree algorithm in *Nature*
- [Barnes-Hut simulation](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) — overview of the algorithm, opening angle criterion, and complexity analysis
- [Octree](https://en.wikipedia.org/wiki/Octree) — the 3D spatial partitioning structure underlying the tree
- [Multipole expansion](https://en.wikipedia.org/wiki/Multipole_expansion) — the mathematical framework for approximating distant mass distributions; Barnes-Hut uses the monopole (zeroth-order) term
- [Fast multipole method](https://en.wikipedia.org/wiki/Fast_multipole_method) — an $O(N)$ extension that includes higher-order multipoles, for reference
- [Center of mass](https://en.wikipedia.org/wiki/Center_of_mass) — the mass-weighted position used for monopole approximation
