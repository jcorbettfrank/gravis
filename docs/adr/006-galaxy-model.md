# ADR 006: Galaxy Model Choices

**Date**: 2026-03-22
**Status**: Accepted

## Decision

Use the Hernquist profile for dark matter halos, an exponential disk with sech^2 vertical structure for the stellar disk, and a Plummer sphere for the bulge. Stay in N-body units (G=1). Use the virial approximation (sigma = V_c / sqrt(2)) for isotropic halo velocities rather than solving the full Eddington inversion.

## Context

M5 introduces multi-component galaxy models for collision scenarios. Three design decisions were made:

### 1. Hernquist over NFW for dark matter halos

Both the Hernquist and NFW profiles reproduce the inner density cusp (rho ~ 1/r) observed in cosmological simulations, and they produce similar rotation curves inside a few scale radii -- the region that matters most for galaxy dynamics.

We chose Hernquist for two practical reasons:

- **Finite total mass.** The NFW enclosed mass diverges as ln(1 + r/r_s), requiring a truncation radius and making the total mass an additional free parameter. Hernquist's M(<r) = M * r^2 / (r+a)^2 converges to a finite M, eliminating this parameter.
- **Analytical CDF inversion.** The Hernquist CDF inverts to r = a * sqrt(U) / (1 - sqrt(U)) for uniform U, giving direct radius sampling with no rejection. The NFW CDF involves a logarithm with no closed-form inverse, requiring numerical root-finding or rejection sampling.

The trade-off is in the outer halo: Hernquist falls as r^{-4} while NFW falls as r^{-3}. For collision dynamics where the outer halo is tidally stripped early, this difference is negligible.

### 2. Disk enclosed mass via analytical formula

The exponential disk surface density Sigma(R) = Sigma_0 * exp(-R/R_d) has a closed-form enclosed mass: M(<R) = M_total * [1 - (1 + R/R_d) * exp(-R/R_d)]. We pre-compute this into a 200-bin lookup table for fast interpolation during the circular velocity calculation. The analytical formula avoids numerical integration and gives machine-precision enclosed masses.

### 3. Virial approximation for halo velocities

The exact velocity distribution for a Hernquist halo embedded in a multi-component potential (disk + bulge + halo) requires solving Eddington's integral equation numerically -- inverting the density-potential pair to obtain the isotropic distribution function f(E). This is straightforward but adds implementation complexity.

Instead, we use the virial approximation: at radius r, the 1D velocity dispersion is sigma(r) = V_c(r) / sqrt(2), where V_c is the circular velocity of the combined potential. Each velocity component is drawn from a Gaussian with this dispersion. This produces halos that are stable for tens of dynamical times -- sufficient for collision runs that develop interesting structure within 5-10 t_dyn.

### 4. N-body units throughout

All galaxy parameters are specified in N-body units (G=1, baryon mass = 1.0, disk scale radius = 1.0). The natural physical mapping is: 1 length unit = 1 kpc, 1 mass unit = 10^10 M_sun, 1 time unit ~ 14.9 Myr. This keeps the code free of physical constants and makes parameter exploration dimensionless.

## Consequences

- Galaxy models are deterministic (fixed RNG seed via ChaCha20Rng) and reproducible across platforms.
- The Hernquist choice means the outer halo density profile differs from cosmological predictions at r >> a_h. This is acceptable for our use case (visual demos and pedagogical examples) but would need revisiting for quantitative cosmological comparisons.
- The virial approximation introduces a small initial disequilibrium in the halo. Isolated galaxies show ~2% radial pulsation over the first few dynamical times before settling. This is within acceptable bounds and is not visible in collision scenarios where tidal forces dominate.
- Adding new galaxy components (e.g., a gas disk for SPH in M6) requires extending the circular_velocity function and the ParticleType enum, but does not change the existing structure.
- The disk stability depends on the velocity dispersion parameterization (sigma_R = 0.15 * V_c). A too-cold disk develops spiral instabilities within a few orbits; a too-hot disk puffs up vertically. The current values give Q ~ 1.5 at 2 R_d, which is stable without being unrealistically hot.
