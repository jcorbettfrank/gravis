use crate::particle::Particles;
use crate::scenario::Scenario;
use crate::units::G;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Cold collapse: a uniform density sphere with zero initial velocity.
///
/// All particles start at rest within a uniform sphere and collapse under
/// self-gravity. This produces dramatic shell crossings and eventual
/// violent relaxation into an approximate virial equilibrium state.
///
/// The free-fall time for a uniform sphere is:
///   t_ff = π/2 * √(R³ / (2GM))
///
/// This scenario is useful because:
/// 1. Simple initial conditions (no velocity sampling needed).
/// 2. Tests the integrator under strong, rapidly changing forces.
/// 3. Visually dramatic — the sphere collapses, bounces, and relaxes.
///
/// When `sph` is true, particles are created as gas with low internal energy
/// (u₀ = 0.05), enabling SPH pressure/density computation during collapse.
/// Without cooling or sink particles, this models gas compression and
/// fragmentation — not star formation.
pub struct ColdCollapse {
    /// Number of particles
    pub n: usize,
    /// Sphere radius
    pub radius: f64,
    /// Total mass
    pub total_mass: f64,
    /// RNG seed for deterministic generation
    pub seed: u64,
    /// If true, create gas particles (SPH) instead of gravity-only particles.
    pub sph: bool,
}

impl Default for ColdCollapse {
    fn default() -> Self {
        Self {
            n: 5000,
            radius: 1.0,
            total_mass: 1.0,
            seed: 42,
            sph: false,
        }
    }
}

impl Scenario for ColdCollapse {
    fn name(&self) -> &str {
        if self.sph { "Cold Collapse (Gas)" } else { "Cold Collapse" }
    }

    fn generate(&self) -> Particles {
        assert!(self.n > 0, "Particle count must be > 0, got {}", self.n);
        assert!(
            self.radius > 0.0,
            "Radius must be positive, got {}",
            self.radius
        );
        assert!(
            self.total_mass > 0.0,
            "Total mass must be positive, got {}",
            self.total_mass
        );

        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
        let mut particles = Particles::new(self.n);
        let m_per_particle = self.total_mass / self.n as f64;

        if self.sph {
            let (_, _, h) = super::sphere_spacing(self.radius, self.n);
            let u_0 = 0.05; // Cold but nonzero — same as Evrard
            super::fill_uniform_gas_sphere(
                &mut particles, &mut rng, self.n,
                self.radius, m_per_particle, u_0, h,
            );
        } else {
            for _ in 0..self.n {
                // Rejection sampling: uniform in cube, accept if inside sphere
                let (x, y, z) = loop {
                    let x: f64 = rng.random_range(-1.0..1.0);
                    let y: f64 = rng.random_range(-1.0..1.0);
                    let z: f64 = rng.random_range(-1.0..1.0);
                    if x * x + y * y + z * z <= 1.0 {
                        break (x * self.radius, y * self.radius, z * self.radius);
                    }
                };

                // Zero velocity — cold collapse
                particles.add(x, y, z, 0.0, 0.0, 0.0, m_per_particle);
            }
        }

        // Shift to center-of-mass frame (should be near origin already,
        // but this ensures exact symmetry for conservation tests)
        particles.shift_to_com_frame();

        particles
    }

    fn suggested_dt(&self) -> f64 {
        // Free-fall time: t_ff = π/2 * √(R³ / (2GM))
        let t_ff = std::f64::consts::FRAC_PI_2
            * (self.radius.powi(3) / (2.0 * G * self.total_mass)).sqrt();
        if self.sph {
            // Finer initial dt — CFL will take over after first step
            t_ff / 500.0
        } else {
            // ~200 steps per free-fall time
            t_ff / 200.0
        }
    }

    fn suggested_softening(&self) -> f64 {
        // Mean inter-particle spacing in a uniform sphere
        let mean_spacing = (4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3)
            / self.n as f64)
            .powf(1.0 / 3.0);
        mean_spacing * 0.3
    }
}
