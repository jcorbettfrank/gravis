# Smoothed Particle Hydrodynamics

Galaxies aren't just stars and dark matter. They contain gas -- vast reservoirs of hydrogen and helium that collapse to form new stars, get heated by supernovae, and develop shocks when galaxies collide. Gas has pressure, viscosity, and thermal energy. It dissipates. It radiates. It behaves fundamentally differently from the collisionless particles we've simulated so far.

Smoothed Particle Hydrodynamics (SPH) is a Lagrangian, mesh-free method for simulating fluid dynamics. Instead of discretizing space into a fixed grid, SPH discretizes *mass* into particles that carry fluid properties -- density, pressure, internal energy -- and move with the flow. Each particle's properties are smoothed over a finite region using a kernel function, turning the discrete particle set into continuous fields that obey the equations of hydrodynamics. Invented independently by Lucy (1977) and Gingold & Monaghan (1977) for astrophysical problems, SPH remains one of the most widely used methods in computational astrophysics precisely because it slots naturally into an existing N-body framework like ours.

## Why SPH?

SPH has several properties that make it attractive for astrophysics:

- **Adaptive resolution.** Particles cluster where the mass is. Dense regions automatically get more resolution; voids get less. No adaptive mesh refinement machinery needed.
- **Free surfaces and vacuum.** No special treatment required for gas boundaries -- particles simply stop being near other particles. Grid codes must handle vacuum cells carefully.
- **Angular momentum conservation.** SPH conserves angular momentum exactly (in the continuum limit), which matters for rotating disks and accretion flows.
- **No mesh generation.** No Voronoi tessellation, no Delaunay triangulation, no hanging nodes. Just particles and a kernel.
- **Natural coupling to gravity.** Gas particles are just particles. The same octree and leapfrog integrator from previous chapters work unchanged.

The main trade-off is that SPH has difficulty with mixing and instabilities at contact discontinuities (the Kelvin-Helmholtz problem). Modern SPH formulations mitigate this, and for our purposes -- shock tubes, cloud collapse, galaxy gas dynamics -- classical SPH with the right kernel and dissipation terms works well.

## The Kernel

The central idea of SPH is *kernel interpolation*: any field $A(\vec{r})$ can be approximated from discrete particle values as

$$A(\vec{r}) \approx \sum_j \frac{m_j}{\rho_j} A_j \, W(|\vec{r} - \vec{r}_j|, \, h)$$

where $m_j$ and $\rho_j$ are the mass and density of particle $j$, and $W(r, h)$ is a smoothing kernel with compact support. The smoothing length $h$ controls the interpolation radius. The kernel must be normalized ($\int W \, dV = 1$), positive, monotonically decreasing, and approach a Dirac delta as $h \to 0$.

We use the **Wendland C2** kernel rather than the more traditional cubic spline. The cubic spline has a flat inflection region near $r = 0$, which causes a *pairing instability*: particles clump into pairs because the kernel gradient vanishes at the origin and the restoring force between nearby particles disappears. The Wendland C2 kernel has a non-zero second derivative at the origin, which eliminates this instability (Dehnen & Aly 2012).

The Wendland C2 kernel in 3D with compact support at $q = r/h = 2$ is:

$$W(q) = \frac{\sigma}{h^3} \left(1 - \frac{q}{2}\right)^4 (1 + 2q) \quad \text{for } q \leq 2, \quad W = 0 \text{ otherwise}$$

where $\sigma = 21/(16\pi)$ is the normalization constant. The compact support at $2h$ means each particle interacts with roughly 58 neighbors in 3D -- enough for smooth interpolation, few enough for efficiency.

In code ([`kernel.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/src/sph/kernel.rs)):

```rust
const NORM_3D: f64 = 21.0 / (16.0 * PI);

pub fn w(r: f64, h: f64) -> f64 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let h3 = h * h * h;
    let s = 1.0 - 0.5 * q; // (1 - q/2)
    let s4 = s * s * s * s;
    NORM_3D / h3 * s4 * (1.0 + 2.0 * q)
}
```

No branches inside the hot path, no transcendental functions. Four multiplications for `s4`, one for the linear term, and a division by $h^3$.

## Density Estimation

The most fundamental SPH quantity is density. Each particle's density is computed by summing the kernel over its neighbors:

$$\rho_i = \sum_j m_j \, W(|\vec{r}_i - \vec{r}_j|, \, h_i)$$

This includes the self-contribution ($j = i$), which is simply $m_i \, W(0, h_i) = m_i \sigma / h_i^3$.

But there's a circularity: the smoothing length $h$ depends on the density (we want more resolution where gas is denser), and the density depends on $h$. We resolve this with the relation $h_i = \eta (m_i / \rho_i)^{1/3}$ and iterate to self-consistency ([`density.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/src/sph/density.rs)):

```rust
for _iter in 0..MAX_H_ITERATIONS {
    let (rho, dw_dh_sum) = density_and_dw_dh(particles, &nlist, i, h);

    // h-ρ relation: h = η(m/ρ)^{1/3}
    let h_new = (eta * (m_i / rho).cbrt()).min(h_max);

    // Grad-h correction: Ω_i = 1 - (∂h/∂ρ) Σ_j m_j ∂W/∂h
    let dh_drho = -h_new / (3.0 * rho);
    omega[i] = 1.0 - dh_drho * dw_dh_sum;

    if (h_new - h).abs() / h < H_TOLERANCE { break; }
    h = h_new;
}
```

The $\Omega_i$ factor is the **grad-h correction** from Springel & Hernquist (2002). When $h$ varies spatially, the kernel no longer exactly conserves energy unless we account for $\partial W / \partial h$. The correction factor $\Omega_i = 1 - (\partial h / \partial \rho) \sum_j m_j (\partial W / \partial h)$ enters the force equations as a denominator, restoring exact energy conservation. In uniform-density regions $\Omega_i \approx 1$; near density gradients it departs from unity.

## Equation of State

With density in hand, we need pressure. For a monatomic ideal gas:

$$P = (\gamma - 1) \rho u, \qquad c_s = \sqrt{\gamma (\gamma - 1) u}$$

where $u$ is the specific internal energy and $\gamma = 5/3$ is the adiabatic index. The sound speed $c_s$ sets the maximum signal propagation speed and thus the CFL timestep constraint.

## Pressure Forces

The SPH momentum equation is derived from a variational principle -- minimizing the fluid Lagrangian subject to the SPH density constraint. The result is a symmetric pairwise force that exactly conserves momentum and energy (with the grad-h correction):

$$\frac{d\vec{v}_i}{dt} = -\sum_j m_j \left[\frac{P_i}{\Omega_i \rho_i^2} \nabla_i W(h_i) + \frac{P_j}{\Omega_j \rho_j^2} \nabla_i W(h_j)\right]$$

Each term uses the kernel evaluated at the *respective particle's* smoothing length, not a symmetrized average. This is what makes the grad-h formulation special: each particle "sees" forces through its own kernel, and the $\Omega$ corrections ensure the asymmetry is thermodynamically consistent.

In code, the force loop is **gather-only**: each particle $i$ loops over its neighbors and accumulates forces onto itself ([`forces.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/src/sph/forces.rs)). This doubles the work compared to a scatter approach (each pair is computed twice), but it's trivially parallel and works on WASM where shared mutable state is not an option.

```rust
// Momentum equation
ax_acc -= mj * (p_over_rho2_i * grad_wi[0]
              + p_over_rho2_j * grad_wj[0]
              + pi_ij * grad_w_avg[0]);
```

The `pi_ij` term is the artificial viscosity, discussed next.

## Artificial Viscosity

Real gases develop shocks -- discontinuous jumps in density, pressure, and velocity. SPH particles, being smooth by construction, cannot represent a true discontinuity. Without explicit dissipation, particles would oscillate through the shock front indefinitely.

Artificial viscosity adds a repulsive pressure between approaching particles, converting kinetic energy into thermal energy and broadening the shock over a few smoothing lengths. We use the standard Monaghan form:

$$\Pi_{ij} = \begin{cases} \displaystyle\frac{-\bar{\alpha} \, \bar{c} \, \mu_{ij} + \beta \, \mu_{ij}^2}{\bar{\rho}} & \text{if } \vec{v}_{ij} \cdot \vec{r}_{ij} < 0 \\ 0 & \text{otherwise} \end{cases}$$

where $\mu_{ij} = \bar{h} \, (\vec{v}_{ij} \cdot \vec{r}_{ij}) / (r_{ij}^2 + \eta^2)$, bars denote pairwise averages, and $\eta^2 = 0.01 \bar{h}^2$ prevents divergence at zero separation. The linear term ($\alpha$) handles weak shocks; the quadratic term ($\beta$) prevents particle interpenetration in strong shocks.

The problem with a fixed $\alpha$ is that viscosity is applied everywhere, not just at shocks. This damps real physical motions -- shear flows, vortices, sound waves. The **Morris & Monaghan (1997) switch** fixes this by giving each particle its own time-varying $\alpha_i$:

$$\frac{d\alpha_i}{dt} = -\frac{\alpha_i - \alpha_{\min}}{\tau_i} + S_i$$

The decay timescale $\tau_i = h_i / (2 \sigma c_i)$ drags $\alpha$ back toward $\alpha_{\min} = 0.1$ in smooth regions. The source term $S_i = \max(-\nabla \cdot \vec{v}_i, 0)(\alpha_{\max} - \alpha_i)$ spikes $\alpha$ toward $\alpha_{\max} = 2.0$ wherever there's convergent flow (negative velocity divergence). The result: strong viscosity at shocks, nearly zero viscosity elsewhere.

## Artificial Conductivity

Shocks produce the right post-shock density and velocity, but standard SPH has trouble with **contact discontinuities** -- interfaces where pressure is continuous but density and temperature jump. Without thermal diffusion, the contact surface stays artificially sharp and develops a pressure blip.

Price (2008) introduced artificial thermal conductivity to smooth contact discontinuities:

$$\left(\frac{du_i}{dt}\right)_{\text{cond}} = \sum_j m_j \, \alpha_u \, v_{sig}^u \, \frac{u_i - u_j}{\rho_i + \rho_j} \, |\nabla W_{ij}|$$

where $v_{sig}^u = \sqrt{|P_i - P_j| / \bar{\rho}}$ is a signal velocity based on the pressure difference. This only activates where there's a pressure jump -- in smoothly varying flows $P_i \approx P_j$ and the conductivity vanishes.

## Energy Equation

Internal energy evolves from three contributions: PdV work, viscous heating, and artificial conductivity:

$$\frac{du_i}{dt} = \frac{P_i}{\Omega_i \rho_i^2} \sum_j m_j \, \vec{v}_{ij} \cdot \nabla_i W(h_i) + \frac{1}{2}\sum_j m_j \, \Pi_{ij} \, \vec{v}_{ij} \cdot \nabla_i \bar{W} + \left(\frac{du_i}{dt}\right)_{\text{cond}}$$

The first term is reversible compression/expansion work. The second converts kinetic energy to heat at shocks (always positive for approaching particles, since $\Pi_{ij} > 0$ and $\vec{v}_{ij} \cdot \nabla W < 0$ for converging flow). The third smooths thermal energy across contact discontinuities.

We enforce an energy floor ($u \geq 10^{-10}$) to prevent negative temperatures from numerical undershoot.

## Adaptive Smoothing and Timestepping

The smoothing length adapts via the relation $h_i = \eta (m_i / \rho_i)^{1/3}$ with $\eta = 1.2$. Dense regions get small $h$ (high resolution); rarefied regions get large $h$. The h-$\rho$ iteration converges in 2--3 iterations for most particles.

The timestep is set by the CFL condition:

$$\Delta t = C_{\text{CFL}} \cdot \min_i \frac{h_i}{v_{sig,i}}$$

where $v_{sig,i} = c_i + h_i |\nabla \cdot \vec{v}_i| + 1.2(\alpha_i c_i + \beta \mu_{\max,i})$ is the maximum signal velocity for particle $i$. With $C_{\text{CFL}} = 0.3$, information cannot travel more than 30% of a smoothing length per step.

## Neighbor Search

Finding the ~58 neighbors within $2h$ of each particle could be $O(N^2)$ by brute force. We reuse the octree from [Chapter 3](ch03_barnes_hut.md) with a **ball query**: starting from the root, descend into any child whose bounding box overlaps the search sphere, and collect all particles within the radius.

The neighbor lists are stored in a flat array with per-particle offset ranges ([`neighbors.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/src/sph/neighbors.rs)), avoiding per-particle heap allocations. On native builds with Rayon, the searches run in parallel; on WASM they run sequentially.

## Code Walkthrough: KDK Leapfrog with SPH

The integration scheme is the same kick-drift-kick (KDK) leapfrog from [Chapter 2](ch02_integrators.md), extended with half-kicks for internal energy and the viscosity switch. The full structure ([`solver.rs`](https://github.com/jcorbettfrank/gravis/blob/m6/crates/sim-core/src/sph/solver.rs)):

```rust
pub fn step_with_sph(
    particles: &mut Particles,
    gravity: &dyn GravitySolver,
    sph: &mut SphSolver,
    dt: f64,
) -> f64 {
    let half_dt = 0.5 * dt;

    // Half-kick: velocities
    for i in 0..n {
        particles.vx[i] += particles.ax[i] * half_dt;
        // ...
    }

    // Half-kick: internal energy (gas only)
    for i in 0..n {
        if particles.is_gas(i) {
            particles.internal_energy[i] += particles.du_dt[i] * half_dt;
            // Energy floor
        }
    }

    // Drift
    for i in 0..n {
        particles.x[i] += particles.vx[i] * dt;
        // ...
    }

    // Recompute all forces
    particles.clear_accelerations();
    particles.clear_sph_rates();
    gravity.compute_accelerations(particles);
    let intermediates = sph.compute(particles);  // density → EOS → forces

    // Half-kick: velocities (second half)
    // Half-kick: internal energy + viscosity switch update

    // Return CFL-limited timestep for next step
    sph.compute_timestep(particles, &intermediates)
}
```

The `sph.compute()` call is where the physics lives. It rebuilds the octree, runs the density-$h$ iteration with grad-h corrections, applies the equation of state, then computes pressure forces, artificial viscosity, and thermal conductivity in a single pass over the neighbor lists.

The function returns the next timestep, so the simulation automatically adapts: small steps during the shock passage, larger steps in smooth regions.

## Try It: Sod Shock Tube

The [Sod shock tube](https://en.wikipedia.org/wiki/Sod_shock_tube) is the canonical SPH test problem. A tube is filled with gas at two different states -- high density and pressure on the left, low on the right -- separated by a membrane at $x = 0.5$. At $t = 0$ the membrane is removed.

The resulting flow has four distinct features, all with known analytical solutions: an expansion fan propagating left into the high-density gas, a contact discontinuity where the original interface was, a shock front propagating right into the low-density gas, and the undisturbed regions at each end. If the SPH solver handles all four correctly, it's ready for astrophysics.

<div class="demo-container">
<iframe src="demos/sod-shock.html" width="100%" height="500" frameborder="0"></iframe>
</div>

## Further Reading

- [Wendland (1995)](https://doi.org/10.1007/BF02123482) -- piecewise polynomial, positive definite, and compactly supported radial basis functions
- [Dehnen & Aly (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.425.1068D/) -- improving convergence in SPH; why Wendland kernels avoid pairing instability
- [Springel & Hernquist (2002)](https://ui.adsabs.harvard.edu/abs/2002MNRAS.333..649S/) -- the grad-h formulation for fully conservative SPH
- [Price (2012)](https://ui.adsabs.harvard.edu/abs/2012JCoPh.231..759P/) -- comprehensive review of SPH methods and best practices
- [Morris & Monaghan (1997)](https://ui.adsabs.harvard.edu/abs/1997JCoPh.136...41M/) -- time-dependent artificial viscosity for reducing shear dissipation
- [Price (2008)](https://ui.adsabs.harvard.edu/abs/2008JCoPh.227.10040P/) -- artificial thermal conductivity for contact discontinuities
- [Monaghan (1992)](https://ui.adsabs.harvard.edu/abs/1992ARA%26A..30..543M/) -- the classic annual reviews article on SPH
