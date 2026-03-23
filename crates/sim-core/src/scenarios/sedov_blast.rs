//! Sedov-Taylor blast wave: point energy injection in uniform gas.
//!
//! The analytical solution gives blast radius R(t) = C · (E₀/ρ₀)^{1/5} · t^{2/5},
//! where C ≈ 1.15 for γ = 5/3 in 3D.
//!
//! Initial conditions: uniform density gas sphere with a small central kernel
//! receiving all the blast energy.

use crate::particle::Particles;
use crate::scenario::Scenario;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand::Rng;

/// Sedov-Taylor blast wave initial conditions.
pub struct SedovBlast {
    /// Total number of gas particles.
    pub n_particles: usize,
    /// Radius of the uniform gas sphere.
    pub radius: f64,
    /// Background density.
    pub rho_0: f64,
    /// Total blast energy deposited in the center.
    pub e_blast: f64,
    /// Number of particles that receive the blast energy.
    pub n_central: usize,
    /// Background internal energy (very small).
    pub u_background: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for SedovBlast {
    fn default() -> Self {
        Self {
            n_particles: 10_000,
            radius: 1.0,
            rho_0: 1.0,
            e_blast: 1.0,
            n_central: 10,
            u_background: 1e-5,
            seed: 42,
        }
    }
}

impl Scenario for SedovBlast {
    fn name(&self) -> &str {
        "Sedov-Taylor Blast"
    }

    fn generate(&self) -> Particles {
        let mut particles = Particles::new(self.n_particles);
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);

        // Volume of sphere
        let vol = 4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3);
        let mass = self.rho_0 * vol / self.n_particles as f64;

        // Mean inter-particle spacing
        let mean_spacing = (vol / self.n_particles as f64).cbrt();
        let h = 1.5 * mean_spacing;

        // Place particles uniformly in a sphere via rejection sampling
        let mut count = 0;
        while count < self.n_particles {
            let x: f64 = rng.random::<f64>() * 2.0 * self.radius - self.radius;
            let y: f64 = rng.random::<f64>() * 2.0 * self.radius - self.radius;
            let z: f64 = rng.random::<f64>() * 2.0 * self.radius - self.radius;

            if x * x + y * y + z * z <= self.radius * self.radius {
                particles.add_gas(x, y, z, 0.0, 0.0, 0.0, mass, self.u_background, h);
                count += 1;
            }
        }

        // Deposit blast energy in central particles
        // Find the n_central particles closest to origin
        let mut dists: Vec<(usize, f64)> = (0..particles.count)
            .map(|i| {
                let r2 = particles.x[i].powi(2) + particles.y[i].powi(2) + particles.z[i].powi(2);
                (i, r2)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let u_blast = self.e_blast / (self.n_central as f64 * mass);
        for k in 0..self.n_central.min(particles.count) {
            let i = dists[k].0;
            particles.internal_energy[i] = u_blast;
        }

        particles
    }

    fn suggested_dt(&self) -> f64 {
        1e-4 // Will be overridden by CFL
    }

    fn suggested_softening(&self) -> f64 {
        let vol = 4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3);
        let mean_spacing = (vol / self.n_particles as f64).cbrt();
        0.3 * mean_spacing
    }
}

/// Analytical Sedov-Taylor blast radius at time t.
///
/// R(t) = ξ₀ · (E₀ / ρ₀)^{1/5} · t^{2/5}
///
/// where ξ₀ ≈ 1.15 for γ = 5/3 in 3D (from dimensional analysis and the
/// Sedov self-similar solution).
pub fn sedov_radius(e_blast: f64, rho_0: f64, t: f64, gamma: f64) -> f64 {
    // ξ₀ depends on γ. For γ = 5/3: ξ₀ ≈ 1.1517
    // General formula: ξ₀ = (75(γ-1)(γ+1)² / (16π(3γ-1)))^{1/5}
    let gp1 = gamma + 1.0;
    let gm1 = gamma - 1.0;
    let xi_0 = (75.0 * gm1 * gp1 * gp1 / (16.0 * std::f64::consts::PI * (3.0 * gamma - 1.0)))
        .powf(0.2);
    xi_0 * (e_blast / rho_0).powf(0.2) * t.powf(0.4)
}
