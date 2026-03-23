//! Top-level SPH solver that orchestrates the full computation pipeline.
//!
//! Each timestep: build tree → find neighbors → density/h iteration →
//! EOS → forces + viscosity switch + conductivity → return CFL timestep.

use crate::gravity::GravitySolver;
use crate::octree::Octree;
use crate::particle::Particles;
use crate::sph::density;
use crate::sph::forces::{self, ForceIntermediates, ForceParams};

/// Floor for internal energy to prevent negative temperatures.
const ENERGY_FLOOR: f64 = 1e-10;

/// Minimum CFL timestep to prevent collapse from isolated particles
/// with bad density estimates or extreme signal velocity.
const DT_FLOOR: f64 = 1e-6;

/// SPH solver configuration and state.
pub struct SphSolver {
    /// Adiabatic index (5/3 for monatomic ideal gas).
    pub gamma: f64,
    /// Smoothing length scaling: h = η(m/ρ)^{1/3}.
    pub eta: f64,
    /// Force computation parameters.
    pub force_params: ForceParams,
    /// CFL safety factor for hydrodynamic timestep.
    pub cfl_factor: f64,
    /// Safety factor for acceleration-based timestep.
    pub force_factor: f64,
    /// Upper bound on smoothing length. Prevents runaway h growth for
    /// isolated particles whose density drops toward zero. `f64::INFINITY`
    /// disables the clamp.
    pub h_max: f64,
    /// Reusable octree to avoid repeated allocations.
    tree: Octree,
}

impl SphSolver {
    /// Create a new SPH solver with default parameters.
    pub fn new() -> Self {
        Self {
            gamma: 5.0 / 3.0,
            eta: 1.2,
            force_params: ForceParams::default(),
            cfl_factor: 0.3,
            force_factor: 0.25,
            h_max: f64::INFINITY,
            tree: Octree {
                nodes: Vec::new(),
            },
        }
    }

    /// Compute all SPH quantities: density, pressure, forces, du/dt.
    ///
    /// Call this AFTER gravity has been computed (accelerations accumulated).
    /// SPH accelerations and du/dt are added to the existing values.
    ///
    /// Returns the force intermediates needed for CFL timestep computation.
    pub fn compute(&mut self, particles: &mut Particles) -> ForceIntermediates {
        // Build/rebuild octree from current positions
        if self.tree.nodes.is_empty() {
            self.tree = Octree::build(particles);
        } else {
            self.tree.rebuild(particles);
        }

        // Phase 1: Density + adaptive h + EOS + grad-h corrections
        let density_result =
            density::compute_density(particles, &self.tree, self.gamma, self.eta, self.h_max);

        // Phase 2: Forces (pressure gradient + viscosity + conductivity)
        forces::compute_forces(
            particles,
            &density_result.neighbors,
            &density_result.omega,
            &self.force_params,
        )
    }

    /// Compute the CFL-limited timestep for SPH.
    ///
    /// dt = min over all gas particles of:
    ///   - CFL: C_CFL · h_i / v_sig,i
    ///   - Force: C_force · sqrt(h_i / |a_i|)
    ///
    /// Non-gas particles are skipped — their timestep is constrained by the
    /// scenario's suggested_dt, not the adaptive CFL criterion.
    pub fn compute_timestep(
        &self,
        particles: &Particles,
        intermediates: &ForceIntermediates,
    ) -> f64 {
        let mut dt_min = f64::INFINITY;

        for i in 0..particles.count {
            if !particles.is_gas(i) {
                continue;
            }

            let h = particles.smoothing_length[i];

            // Force-based timestep
            let a_mag = (particles.ax[i].powi(2)
                + particles.ay[i].powi(2)
                + particles.az[i].powi(2))
            .sqrt();
            if a_mag > 0.0 {
                let dt_force = self.force_factor * (h / a_mag).sqrt();
                dt_min = dt_min.min(dt_force);
            }

            // CFL timestep for gas particles
            let h = particles.smoothing_length[i];
            let c = particles.sound_speed[i];
            let alpha = particles.alpha_visc[i];
            let div_v_abs = intermediates.div_v[i].abs();
            let mu_max = intermediates.max_mu[i];

            let v_sig = c + h * div_v_abs
                + 1.2 * (alpha * c + self.force_params.beta_visc * mu_max);

            if v_sig > 0.0 {
                let dt_cfl = self.cfl_factor * h / v_sig;
                dt_min = dt_min.min(dt_cfl);
            }
        }

        dt_min.clamp(DT_FLOOR, f64::INFINITY)
    }
}

impl Default for SphSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Perform one KDK leapfrog step with SPH.
///
/// This extends the standard leapfrog with energy and viscosity half-kicks
/// for gas particles. The existing `Integrator::step` remains unchanged for
/// backward compatibility.
pub fn step_with_sph(
    particles: &mut Particles,
    gravity: &dyn GravitySolver,
    sph: &mut SphSolver,
    dt: f64,
) -> f64 {
    let half_dt = 0.5 * dt;
    let n = particles.count;

    // --- Half-kick: velocities ---
    for i in 0..n {
        particles.vx[i] += particles.ax[i] * half_dt;
        particles.vy[i] += particles.ay[i] * half_dt;
        particles.vz[i] += particles.az[i] * half_dt;
    }

    // --- Half-kick: internal energy + viscosity (gas only) ---
    for i in 0..n {
        if particles.is_gas(i) {
            particles.internal_energy[i] += particles.du_dt[i] * half_dt;
            if particles.internal_energy[i] < ENERGY_FLOOR {
                particles.internal_energy[i] = ENERGY_FLOOR;
            }
        }
    }

    // --- Drift ---
    for i in 0..n {
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
    }

    // --- Recompute forces ---
    particles.clear_accelerations();
    particles.clear_sph_rates();

    // Gravity
    gravity.compute_accelerations(particles);

    // SPH (density → EOS → forces)
    let intermediates = sph.compute(particles);

    // --- Half-kick: velocities ---
    for i in 0..n {
        particles.vx[i] += particles.ax[i] * half_dt;
        particles.vy[i] += particles.ay[i] * half_dt;
        particles.vz[i] += particles.az[i] * half_dt;
    }

    // --- Half-kick: internal energy + viscosity switch (gas only) ---
    for i in 0..n {
        if particles.is_gas(i) {
            particles.internal_energy[i] += particles.du_dt[i] * half_dt;
            if particles.internal_energy[i] < ENERGY_FLOOR {
                particles.internal_energy[i] = ENERGY_FLOOR;
            }

            // Update viscosity parameter (Morris & Monaghan switch)
            particles.alpha_visc[i] += intermediates.dalpha_dt[i] * dt;
            particles.alpha_visc[i] = particles.alpha_visc[i]
                .clamp(sph.force_params.alpha_min, sph.force_params.alpha_max);
        }
    }

    // Compute adaptive timestep for next step
    sph.compute_timestep(particles, &intermediates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity::BruteForce;
    use crate::particle::Particles;

    #[test]
    fn static_gas_ball_stable() {
        // A uniform gas ball with balanced pressure should remain roughly stable
        // over a few timesteps (no large accelerations).
        let n_side = 6;
        let spacing = 0.5;
        let u = 1.0;
        let h = 0.8;
        let mut particles = Particles::new(n_side * n_side * n_side);
        for ix in 0..n_side {
            for iy in 0..n_side {
                for iz in 0..n_side {
                    particles.add_gas(
                        ix as f64 * spacing,
                        iy as f64 * spacing,
                        iz as f64 * spacing,
                        0.0, 0.0, 0.0,
                        1.0, u, h,
                    );
                }
            }
        }

        let gravity = BruteForce::new(0.1);
        let mut sph = SphSolver::new();

        // Do initial force computation to seed accelerations
        particles.clear_accelerations();
        gravity.compute_accelerations(&mut particles);
        let _intermediates = sph.compute(&mut particles);

        // Take a few steps
        let mut dt = 0.001;
        for _ in 0..5 {
            dt = step_with_sph(&mut particles, &gravity, &mut sph, dt);
            assert!(dt > 0.0, "Timestep should be positive");
            assert!(dt < 1.0, "Timestep should be reasonable, got {}", dt);
        }

        // Particles should not have flown apart
        for i in 0..particles.count {
            let r = (particles.x[i].powi(2)
                + particles.y[i].powi(2)
                + particles.z[i].powi(2))
            .sqrt();
            assert!(r < 100.0, "Particle {} at r={} — system blew up", i, r);
        }
    }

    #[test]
    fn gravity_only_regression() {
        // Ensure that gravity-only particles are unaffected by SPH solver
        let mut particles = Particles::new(2);
        particles.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        particles.add(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let gravity = BruteForce::new(0.1);
        let mut sph = SphSolver::new();

        // Compute initial forces
        particles.clear_accelerations();
        gravity.compute_accelerations(&mut particles);
        let _intermediates = sph.compute(&mut particles);

        // The particles should attract each other (gravity only, no SPH)
        assert!(particles.ax[0] > 0.0, "Particle 0 should be pulled toward particle 1");
        assert!(particles.ax[1] < 0.0, "Particle 1 should be pulled toward particle 0");

        // du_dt should be zero (no gas particles)
        assert_eq!(particles.du_dt[0], 0.0);
        assert_eq!(particles.du_dt[1], 0.0);
    }
}
