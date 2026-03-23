//! Evrard collapse: self-gravitating adiabatic gas sphere.
//!
//! Initial conditions: uniform density sphere, M=1, R=1, very low thermal energy
//! (u₀ = 0.05). The sphere collapses under self-gravity, forms a hot dense core,
//! and bounces. Tests gravity + SPH coupling.
//!
//! Reference: Evrard (1988), MNRAS 235, 911.

use crate::particle::Particles;
use crate::scenario::Scenario;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Evrard collapse initial conditions.
pub struct EvrardCollapse {
    /// Number of gas particles.
    pub n_particles: usize,
    /// Total mass.
    pub total_mass: f64,
    /// Sphere radius.
    pub radius: f64,
    /// Initial specific internal energy (cold).
    pub u_0: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for EvrardCollapse {
    fn default() -> Self {
        Self {
            n_particles: 5_000,
            total_mass: 1.0,
            radius: 1.0,
            u_0: 0.05,
            seed: 42,
        }
    }
}

impl Scenario for EvrardCollapse {
    fn name(&self) -> &str {
        "Evrard Collapse"
    }

    fn generate(&self) -> Particles {
        let mut particles = Particles::new(self.n_particles);
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);

        let mass = self.total_mass / self.n_particles as f64;
        let (_, _, h) = super::sphere_spacing(self.radius, self.n_particles);

        super::fill_uniform_gas_sphere(
            &mut particles, &mut rng, self.n_particles,
            self.radius, mass, self.u_0, h,
        );

        particles
    }

    fn suggested_dt(&self) -> f64 {
        // Free-fall time t_ff = π/2 √(R³/(2GM))
        // With G=1, M=1, R=1: t_ff ≈ 1.11
        let t_ff = std::f64::consts::FRAC_PI_2
            * (self.radius.powi(3) / (2.0 * self.total_mass)).sqrt();
        t_ff / 500.0 // Will be overridden by CFL
    }

    fn suggested_softening(&self) -> f64 {
        let (_, mean_spacing, _) = super::sphere_spacing(self.radius, self.n_particles);
        0.3 * mean_spacing
    }
}
