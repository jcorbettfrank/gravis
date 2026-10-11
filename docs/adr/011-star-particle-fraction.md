# ADR 011: Unequal Star and Dark-Matter Particle Masses

**Date**: 2026-10-10
**Status**: Accepted

## Decision

`GalaxyModel` gains an optional `star_particle_fraction`: the share of a galaxy's particles that are stars (disk + bulge), whatever their share of the mass. The halo gets the rest, so star particles are lighter than halo particles. Spec `disk_galaxy` components default to **0.5**; `"star_particle_fraction": null` restores the old split by mass. The built-in `GalaxyCollision` scenario keeps the split by mass, so its output is unchanged.

## Context

With equal-mass particles, the halo (10× the baryon mass by default) gets 10 of every 11 particles. A 40K-particle galaxy then has ~2,500 disk stars, and the disk, the part anyone looks at, renders sparse and noisy (handoff known issue 3, ADR 008 consequences).

Making star particles lighter puts more of the particle budget where it shows. The cost is physical: heavy halo particles passing through the disk scatter its stars, a numerical two-body heating that thickens the disk and raises its velocity dispersion (Lacey & Ostriker 1985). The heating rate scales with the halo particle mass, so the split cannot be pushed arbitrarily far.

At star fraction *f*, the mass ratio is m_dm / m_star = `halo_mass_ratio` × *f* / (1 − *f*): 10 at *f* = 0.5 for the default galaxy.

## Measurement

`crates/sim-core/examples/disk_heating.rs` evolves one isolated default galaxy for 50 time units (~7 disk orbits at the scale radius; ~750 Myr in galactic units) with Barnes-Hut (θ = 0.5), the default dt (0.0148) and softening (0.05), and records the disk stars' RMS height and velocity dispersions in the annulus 1 < R < 3. Raw output: `artifacts/benchmarks/disk_heating.csv`.

```text
cargo run -p sim-core --release --example disk_heating -- --particles 20000 --star-fraction 0.5
```

| N | star fraction | disk stars | m_dm/m_star | z_rms t=10 → 50 | σ_z t=50 | σ_R t=50 | 2K/\|W\| t=50 |
|---|---|---|---|---|---|---|---|
| 20K | equal mass (0.09) | 1,273 | 1 | 0.099 → 0.182 | 0.135 | 0.258 | 1.074 |
| 20K | 0.25 | 3,500 | 3.3 | 0.093 → 0.165 | 0.133 | 0.245 | 1.064 |
| 20K | **0.5** | 7,000 | 10 | 0.099 → 0.188 | 0.141 | 0.249 | 1.064 |
| 20K | 0.75 | 10,500 | 30 | 0.117 → 0.330 | 0.206 | 0.287 | 1.085 |
| 20K | 0.9 | 12,600 | 90 | 0.202 → 0.414 | 0.246 | 0.384 | 1.089 |
| 40K | equal mass (0.09) | 2,545 | 1 | 0.081 → 0.137 | 0.116 | 0.216 | 1.071 |
| 40K | **0.5** | 14,000 | 10 | 0.083 → 0.131 | 0.116 | 0.216 | 1.075 |

The initial z_rms is ~0.083 in every run. Energy is conserved to |dE/E| < 4e-4 throughout.

What the numbers say:

- **Up to *f* = 0.5, heating is indistinguishable from equal mass.** At both N, the disk at *f* = 0.5 ends as thick and as hot as the equal-mass disk (differences are within the run-to-run scatter, e.g. 0.25 coming out slightly cooler than equal mass), while carrying 5.5× as many disk stars. At this N the heating floor is set by the overall particle count, not by the halo particle mass.
- **Beyond 0.5 it breaks down.** At 0.75 (5,000 halo particles, m_dm/m_star = 30) the disk thickens ~2.5× as fast after the initial transient; at 0.9 it doubles in height within the first 10 time units and σ_R climbs steadily.
- **Doubling N helps both splits**: from t = 10 to 50 the disk thickens by 0.056 (equal mass) and 0.048 (*f* = 0.5) at 40K, against 0.083 and 0.089 at 20K.

All runs share a pre-existing transient: σ_R jumps from 0.075 to ~0.21 in the first 5 time units and 2K/|W| starts at ~1.15. The disk and halo velocity setups (uniform-distribution disk dispersions, Gaussian halo with σ = V_c/√2) are not in exact equilibrium. That is independent of the particle split and is not addressed here.

## Choice

**Default 0.5.** It is the largest tested fraction whose heating matches the equal-mass baseline, and it gives ~5.5× more disk star particles for the same total N. Keeping the halo at half the particles also keeps its density profile and potential smooth, which matters for merger dynamics (dynamical friction acts on the halo).

The trade-off scales with `halo_mass_ratio`: for a galaxy with a much heavier halo, *f* = 0.5 gives a larger m_dm/m_star, so a spec author with a halo ratio of, say, 30 should lower the fraction (≈ 0.25 keeps the ratio at 10).

## Consequences

- Spec galaxies change: the example specs' particles are now half stars. Their dt, softening, and view radius are unchanged (none depend on the split).
- Particle mass no longer tracks visual importance. A renderer that sizes or brightens particles by mass now draws halo particles 10× heavier than stars; renderers should style by `particle_type` (handoff workstream B).
- `GalaxyCollision`, the named built-in scenario, is untouched (`star_particle_fraction: None`).
- `GalaxyModel::particle_split` exposes the counts, and keeps at least one halo particle when the halo has mass (so `star_particle_fraction: 1` still carries the halo's mass, as a single particle).
