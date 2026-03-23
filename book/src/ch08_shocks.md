# Shocks & Verification

Testing an SPH implementation requires problems with known solutions. A simulation that runs without crashing and produces pretty pictures is not the same as a simulation that gets the physics right. This chapter walks through four classical test problems -- the Sod shock tube, Sedov-Taylor blast wave, Evrard collapse, and Kelvin-Helmholtz instability -- plus two astrophysical scenarios that push gravity and hydrodynamics together. Each test targets a different aspect of the solver and has a clear pass/fail criterion.

## The Riemann Problem

The theoretical foundation for shock tube tests is the Riemann problem: two constant states of gas separated by a membrane, with the membrane removed at $t = 0$. Despite its simplicity, the resulting flow contains three distinct wave families that probe different aspects of the solver.

The gas on each side is described by density $\rho$, pressure $P$, and velocity $u$. When the membrane vanishes, three waves emerge:

1. **A left-going rarefaction fan** -- a smooth, self-similar expansion wave where the flow transitions continuously from the undisturbed left state to the post-rarefaction state.
2. **A contact discontinuity** -- a surface across which density jumps but pressure and velocity are continuous. Entropy changes across the contact; momentum does not.
3. **A right-going shock** -- a discontinuous compression front satisfying the Rankine-Hugoniot jump conditions.

The Rankine-Hugoniot conditions relate the pre-shock and post-shock states across a shock moving at speed $S$:

$$\rho_1 (u_1 - S) = \rho_2 (u_2 - S)$$
$$P_1 + \rho_1 (u_1 - S)^2 = P_2 + \rho_2 (u_2 - S)^2$$

These express conservation of mass and momentum flux across the discontinuity. Combined with the energy jump condition and the equation of state, they determine the post-shock state uniquely given the pre-shock state and shock strength. Solving the full Riemann problem amounts to finding the intermediate pressure $P^*$ that makes the left and right wave solutions match at the contact -- a nonlinear equation solved by Newton-Raphson iteration.

## Sod Shock Tube

The Sod problem is the standard first test for any hydro code. The initial conditions are:

$$\text{Left: } \rho_L = 1, \; P_L = 1, \; u_L = 0 \qquad \text{Right: } \rho_R = 0.125, \; P_R = 0.1, \; u_R = 0$$

A high-pressure, high-density region separated from a low-pressure, low-density region at $x = 0$. Our code includes an analytical Riemann solver ([`sod_analytical`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/src/scenarios/sod_shock.rs)) that computes the exact density, velocity, and pressure profiles at any time $t$ via Newton-Raphson iteration on the intermediate pressure.

We map the 1D problem to 3D by building a slab of particles: 120 layers along $x$ on the left (spacing $\Delta x_L = 0.5 / 120$) and 30 on the right (spacing $\Delta x_R = 0.5 / 30$), with 4 layers each in $y$ and $z$. Particle masses are set so the local density matches $\rho_L$ and $\rho_R$ respectively. This unequal-mass approach is standard for SPH Riemann problems -- it ensures the number density of particles is roughly uniform even though the mass density differs by a factor of 8.

What to look for in the output:

- **Shock position**: the right-going density jump should land near the analytical prediction.
- **Rarefaction fan**: the smooth expansion to the left should follow the isentropic analytical profile.
- **Contact discontinuity**: the hardest feature for SPH. Because SPH estimates density by kernel averaging over neighbors, any sharp density jump is smeared over $\sim 2h$. The contact is a jump in density at constant pressure -- exactly the configuration where kernel smoothing introduces the most error. Grid codes resolve contacts within 2--3 cells; SPH typically needs 5--10 smoothing lengths to approximate the same jump.

The test ([`tests/sod_shock.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/tests/sod_shock.rs)) evolves to $t = 0.05$ and verifies that the shock has formed (rightward-moving particles exist), the maximum velocity is within an order of magnitude of the analytical $u^*$, and no particle has developed NaN in density or internal energy. With only 400 particles in the 3D slab, we check qualitative wave structure rather than pointwise convergence.

## Sedov-Taylor Blast Wave

Deposit a large amount of energy $E_0$ into a single point in a uniform medium of density $\rho_0$. The resulting blast wave has a self-similar solution discovered independently by Sedov, Taylor, and von Neumann in the context of nuclear explosions. The blast radius scales as:

$$R(t) = \xi_0 \left(\frac{E_0}{\rho_0}\right)^{1/5} t^{2/5}$$

where $\xi_0$ is a dimensionless constant that depends on the adiabatic index $\gamma$. For $\gamma = 5/3$ in three dimensions, $\xi_0 \approx 1.15$, computed from the general formula:

$$\xi_0 = \left(\frac{75 (\gamma - 1)(\gamma + 1)^2}{16 \pi (3\gamma - 1)}\right)^{1/5}$$

Our [`sedov_radius()`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/src/scenarios/sedov_blast.rs) function evaluates this expression. The initial conditions place 10,000 gas particles in a uniform sphere of radius 1.0 with $\rho_0 = 1.0$, then deposit all the blast energy ($E_0 = 1.0$) into the 10 innermost particles as internal energy. The background gas has negligible thermal energy ($u = 10^{-5}$).

The test ([`tests/sedov_blast.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/tests/sedov_blast.rs)) measures the blast radius at $t = 0.01$, $0.02$, and $0.04$ by computing the 90th-percentile radius of particles with $|\vec{v}| > 0.1$. Each measured radius is compared against the analytical prediction with a tolerance of 60%. That tolerance is generous -- SPH with 5,000 particles at early times has a thick shock front and the energy deposition into a discrete set of central particles introduces an initialization transient. The important check is the $R \propto t^{2/5}$ scaling: the blast radius increases monotonically and follows the correct power law.

## Evrard Collapse

The Evrard collapse ([Evrard 1988, MNRAS 235, 911](https://ui.adsabs.harvard.edu/abs/1988MNRAS.235..911E)) is the canonical coupled gravity+SPH test. An initially uniform, cold gas sphere ($M = 1$, $R = 1$, $u_0 = 0.05$) collapses under self-gravity. The thermal energy is far too small to support the sphere against gravity -- the virial temperature would require $u \sim GM / (5R) = 0.2$, and we start at one quarter of that.

The sphere falls inward on the free-fall timescale $t_{ff} = \pi / (2\sqrt{2}) \approx 1.11$. As the gas compresses, PdV work heats it. The density at the center spikes, pressure builds, and the core bounces. In a well-resolved simulation, the system eventually reaches an approximate virial equilibrium with a hot dense core and a tenuous envelope.

The verification criterion is energy conservation. Total energy $E = K + U_{\text{grav}} + U_{\text{thermal}}$ should be conserved by the symplectic integrator and the SPH energy equation together. Our test evolves to $t = 0.5$ (about half the free-fall time, during active collapse but before the bounce) and checks:

$$\frac{|\Delta E|}{|E_0|} < 5\%$$

It also verifies that kinetic energy has at least doubled (the sphere is actively falling) and that thermal energy has increased (compressive heating is working). The 5% tolerance accounts for the fact that 1,000-particle SPH has coarse density estimation and the adaptive timestep cannot perfectly track the rapid compression phase.

## Kelvin-Helmholtz Instability

The [Kelvin-Helmholtz instability](https://en.wikipedia.org/wiki/Kelvin%E2%80%93Helmholtz_instability) develops when two fluid layers shear past each other. It is SPH's hardest classical test -- and for good reason. The instability feeds on mixing at a density discontinuity, which is precisely what SPH handles worst.

Our setup: a periodic box with uniform pressure $P = 2.5$, a central strip at $\rho = 2$ moving at $v_x = +0.5$, and surrounding layers at $\rho = 1$ moving at $v_x = -0.5$. A sinusoidal perturbation seeds the instability:

$$v_y(x) = A \sin\left(\frac{2 \pi x}{L}\right)$$

with amplitude $A = 0.025$. The particle spacing in the high-density strip is reduced by $(\rho_{\text{low}} / \rho_{\text{high}})^{1/3}$ to match the density ratio while keeping particle masses consistent.

The linear growth rate of the KH instability is $\sigma \sim k \, v_{\text{shear}}$, giving a growth timescale $\tau \sim L / (\pi \, v_{\text{shear}}) \approx 0.64$. After one growth time, the perturbation should have amplified substantially.

Why is this hard for SPH? Two reasons. First, artificial viscosity -- necessary to stabilize shocks -- also damps the shear that drives the instability. The viscosity switch (Morris & Monaghan 1997) mitigates this by reducing $\alpha_{\text{visc}}$ away from shocks, but some damping remains. Second, SPH is Lagrangian: particles move with the flow, and the kernel smoothing is done over the particle's local neighborhood. At a density discontinuity, the kernel interpolation includes particles from both sides of the interface, leading to a pressure discontinuity at constant density -- the so-called "tensile instability" or "SPH blending problem." This suppresses the mixing that KH requires.

Our test ([`tests/kelvin_helmholtz.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/tests/kelvin_helmholtz.rs)) checks that $v_y$ variance increases by at least 50% over one growth time and that no particle develops NaN. This is a deliberately modest criterion. A grid code would produce dramatic roll-up vortices; our SPH at $64^3$ resolution produces measurable instability growth but not the characteristic cat's-eye vortex pattern. That is an honest limitation of the method.

## Protoplanetary Disk

A massive central star ($M_\star = 1$) surrounded by a self-gravitating gas disk in near-Keplerian rotation. The disk mass is 10% of the stellar mass, making the [Toomre parameter](https://en.wikipedia.org/wiki/Toomre_stability_criterion) $Q \approx 1$ -- the threshold of gravitational instability. The surface density follows $\Sigma \propto r^{-1}$, the disk has a constant aspect ratio $h/r = 0.05$, and the orbital velocities include a sub-Keplerian pressure support correction.

At $Q \sim 1$, disk self-gravity drives spiral density waves. These are not imposed -- they emerge spontaneously from the interaction of pressure, rotation, and gravity. The spiral arms compress gas, heat it through PdV work, and the increased pressure stabilizes the disk locally until it cools and fragments again. This interplay between heating and cooling sets the saturated state of the disk.

What we can and cannot claim with this model:

- **Spiral arms from gravitational instability** -- yes, this is well-resolved at our particle counts and is the primary science target.
- **Gap formation** -- no. Gaps in protoplanetary disks are carved by embedded planets or massive companions. Without an embedded perturber, our disk will not produce gaps, and we do not claim otherwise.
- **Planet formation** -- no. Planet formation requires cooling (to allow fragments to collapse further) and sink particles (to represent collapsed objects). Our adiabatic equation of state prevents runaway fragmentation. What we model is gas compression and spiral structure.

The test ([`tests/protoplanetary.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/tests/protoplanetary.rs)) evolves 500 gas particles plus one star for 50 steps and verifies that all particles remain bound (no ejections beyond $10 \times r_{\text{out}}$) and that no NaN appears.

## Gas Cloud Collapse

The cold collapse scenario from [Chapter 6](ch06_scenarios.md) has an SPH variant: set `sph: true` and the uniform sphere is built from gas particles with low internal energy ($u_0 = 0.05$) instead of gravity-only point masses. The gas collapses under self-gravity just like the collisionless case, but now pressure forces resist compression, PdV work heats the gas, and the interplay between gravity and thermal pressure produces richer structure.

In the collisionless cold collapse, all particles pass through the center and undergo violent relaxation. With gas, pressure halts the infall before shell crossing. The core heats up, bounces, and sends a shock outward. Depending on the ratio of thermal to gravitational energy, the gas may fragment into multiple clumps as thermal Jeans masses compete with the global collapse.

Honest scope: this is cloud collapse and fragmentation, not star formation. Without radiative cooling, the gas cannot shed the thermal energy it gains during compression. Without sink particles, collapsed regions cannot be replaced by point-mass objects that would free the timestep from resolving their internal dynamics. What we see is the initial gravitational fragmentation of a cold gas cloud -- the first act of a much longer story that our code does not tell.

```bash
# Run the gas cloud collapse
cargo run -p native-app --release -- --scenario cold-collapse-gas -n 5000
```

## Verification Summary

| Test | Command | Criterion |
|------|---------|-----------|
| Sod Shock Tube | `cargo test -p sim-core --release sod_shock` | Shock forms, $v_{\max}$ matches analytical $u^*$ within 10x, no NaN |
| Sedov-Taylor Blast | `cargo test -p sim-core --release sedov_blast` | $R(t)$ within 60% of analytical, monotonically increasing |
| Evrard Collapse | `cargo test -p sim-core --release evrard` | $\|\Delta E\| / \|E_0\| < 5\%$, $K$ and $U_{\text{th}}$ increase |
| Kelvin-Helmholtz | `cargo test -p sim-core --release kelvin_helmholtz` | $v_y$ variance grows $> 1.5\times$, no NaN |
| Protoplanetary Disk | `cargo test -p sim-core --release protoplanetary` | All particles bound, no NaN after 50 steps |
| Gas Cloud Collapse | (visual / energy check) | Core forms, gas heats, no blow-up |

These tests run in CI on every push. The tolerances are deliberately loose -- we are testing that the physics is qualitatively correct at modest resolution, not that we match a reference solution to 1%. Production SPH codes (GADGET-4, SWIFT, Phantom) run these same tests at $10^5$--$10^6$ particles and demand tighter convergence. Our tests serve a different purpose: catching regressions. If a refactor breaks the pressure gradient calculation, the Sod shock will fail to form. If the energy equation has a sign error, Evrard's energy conservation will blow up. If artificial viscosity is too aggressive, KH growth will stall.

The full test suite runs in under 30 seconds in release mode:

```bash
# Run all SPH verification tests
cargo test -p sim-core --release
```

## Try It: Sedov Blast

Watch the blast wave expand from a point energy injection. The shock front should be roughly spherical and expand as $R \propto t^{2/5}$.

<div class="demo-container">
<iframe src="../demos/index.html?scenario=sedov-blast" width="100%" height="500" frameborder="0"></iframe>
</div>
