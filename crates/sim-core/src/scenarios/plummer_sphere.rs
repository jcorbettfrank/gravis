use crate::particle::Particles;
use crate::scenario::Scenario;
use crate::units::G;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Plummer sphere: an analytical model of a self-gravitating stellar system
/// in virial equilibrium.
///
/// The density profile is:
///   ρ(r) = (3M / 4πa³) * (1 + r²/a²)^(-5/2)
///
/// where M is the total mass and a is the Plummer scale radius.
///
/// This model is useful because:
/// 1. It has a known analytical distribution function f(E), so we can
///    sample exact equilibrium initial conditions.
/// 2. In virial equilibrium, 2K + U = 0 (virial theorem), meaning
///    the virial ratio 2K/|U| = 1.
/// 3. Any secular drift in the virial ratio indicates a bug in the
///    integrator or force calculation.
///
/// The sampling method follows Aarseth, Henon & Wielen (1974):
/// - Positions: CDF inversion of the Plummer density.
/// - Velocities: rejection sampling from the distribution function.
pub struct PlummerSphere {
    /// Number of particles
    pub n: usize,
    /// Plummer scale radius
    pub scale_radius: f64,
    /// Total mass
    pub total_mass: f64,
    /// RNG seed for deterministic generation
    pub seed: u64,
}

impl Default for PlummerSphere {
    fn default() -> Self {
        Self {
            n: 1000,
            scale_radius: 1.0,
            total_mass: 1.0,
            seed: 42,
        }
    }
}

impl Scenario for PlummerSphere {
    fn name(&self) -> &str {
        "Plummer Sphere"
    }

    fn generate(&self) -> Particles {
        assert!(self.n > 0, "Particle count must be > 0, got {}", self.n);
        assert!(
            self.scale_radius > 0.0,
            "Scale radius must be positive, got {}",
            self.scale_radius
        );
        assert!(
            self.total_mass > 0.0,
            "Total mass must be positive, got {}",
            self.total_mass
        );

        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
        let mut particles = Particles::new(self.n);
        let m_per_particle = self.total_mass / self.n as f64;
        let a = self.scale_radius;

        for _ in 0..self.n {
            // --- Sample position ---
            // The cumulative mass profile is: M(<r) = M * r³ / (r² + a²)^(3/2)
            // Setting M(<r)/M = X (uniform random), solving for r:
            //   r = a / sqrt(X^(-2/3) - 1)
            let x_mass: f64 = rng.random_range(0.001..1.0); // avoid r=0
            let r = a / (x_mass.powf(-2.0 / 3.0) - 1.0).sqrt();

            // Uniform direction on sphere
            let cos_theta: f64 = rng.random_range(-1.0..1.0);
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let phi: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);

            let px = r * sin_theta * phi.cos();
            let py = r * sin_theta * phi.sin();
            let pz = r * cos_theta;

            // --- Sample velocity ---
            // The escape velocity at radius r is:
            //   v_esc = sqrt(2 * |Φ(r)|) = sqrt(2GM / sqrt(r² + a²))
            let v_esc = (2.0 * G * self.total_mass / (r * r + a * a).sqrt()).sqrt();

            // Rejection sampling from the distribution function.
            // The DF f(E) ∝ (-E)^(7/2) for the Plummer model.
            // We sample q = v/v_esc uniformly in [0, 1) and accept with
            // probability proportional to q² * (1 - q²)^(7/2).
            let q = loop {
                let q: f64 = rng.random_range(0.0..1.0);
                let g = q * q * (1.0 - q * q).powf(3.5);
                // Maximum of q²(1-q²)^(7/2) is at q² = 1/5, giving g_max ≈ 0.09177
                let g_max = 0.1;
                let u: f64 = rng.random();
                if u < g / g_max {
                    break q;
                }
            };

            let v = q * v_esc;

            // Uniform direction for velocity
            let cos_theta_v: f64 = rng.random_range(-1.0..1.0);
            let sin_theta_v = (1.0 - cos_theta_v * cos_theta_v).sqrt();
            let phi_v: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);

            let vx = v * sin_theta_v * phi_v.cos();
            let vy = v * sin_theta_v * phi_v.sin();
            let vz = v * cos_theta_v;

            particles.add(px, py, pz, vx, vy, vz, m_per_particle);
        }

        // Shift to center-of-mass frame to ensure COM = 0, P = 0
        particles.shift_to_com_frame();

        particles
    }

    fn suggested_dt(&self) -> f64 {
        // Dynamical time for Plummer sphere: t_dyn ~ (a³ / GM)^(1/2)
        // With G=1, M=1, a=1: t_dyn = 1.
        // Use ~100 steps per dynamical time.
        let t_dyn =
            (self.scale_radius.powi(3) / (G * self.total_mass)).sqrt();
        t_dyn / 100.0
    }

    fn suggested_softening(&self) -> f64 {
        // Softening ~ mean inter-particle spacing at the half-mass radius.
        // Half-mass radius of Plummer sphere ≈ 1.305 * a.
        // Mean spacing ~ (4/3 * π * r_h³ / N)^(1/3)
        let r_h = 1.305 * self.scale_radius;
        let mean_spacing = (4.0 / 3.0 * std::f64::consts::PI * r_h.powi(3) / self.n as f64)
            .powf(1.0 / 3.0);
        mean_spacing * 0.5
    }
}

