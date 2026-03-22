use crate::particle::Particles;
use crate::scenario::Scenario;
use crate::units::G;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Sample a single particle from the Plummer distribution function.
///
/// Returns `(position, velocity)` as `([f64; 3], [f64; 3])`.
/// Uses CDF inversion for positions and rejection sampling for velocities
/// following Aarseth, Henon & Wielen (1974).
pub fn sample_plummer_particle(
    rng: &mut ChaCha20Rng,
    total_mass: f64,
    scale_radius: f64,
) -> ([f64; 3], [f64; 3]) {
    let a = scale_radius;

    // Position: CDF inversion of M(<r)/M = r³/(r²+a²)^(3/2)
    let x_mass: f64 = rng.random_range(0.001..1.0);
    let r = a / (x_mass.powf(-2.0 / 3.0) - 1.0).sqrt();

    let cos_theta: f64 = rng.random_range(-1.0..1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);

    let pos = [
        r * sin_theta * phi.cos(),
        r * sin_theta * phi.sin(),
        r * cos_theta,
    ];

    // Velocity: rejection sampling from DF f(E) ∝ (-E)^(7/2)
    let v_esc = (2.0 * G * total_mass / (r * r + a * a).sqrt()).sqrt();
    let q = loop {
        let q: f64 = rng.random_range(0.0..1.0);
        let g = q * q * (1.0 - q * q).powf(3.5);
        let g_max = 0.1;
        let u: f64 = rng.random();
        if u < g / g_max {
            break q;
        }
    };
    let v = q * v_esc;

    let cos_theta_v: f64 = rng.random_range(-1.0..1.0);
    let sin_theta_v = (1.0 - cos_theta_v * cos_theta_v).sqrt();
    let phi_v: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);

    let vel = [
        v * sin_theta_v * phi_v.cos(),
        v * sin_theta_v * phi_v.sin(),
        v * cos_theta_v,
    ];

    (pos, vel)
}

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
            let (pos, vel) = sample_plummer_particle(&mut rng, self.total_mass, a);
            particles.add(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], m_per_particle);
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

