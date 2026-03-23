//! Protoplanetary disk: central star + Keplerian gas disk.
//!
//! A massive central star surrounded by a self-gravitating gas disk in
//! near-Keplerian rotation. The disk-to-star mass ratio (default 10%) makes
//! it marginally Toomre-unstable, producing spiral structure from disk
//! self-gravity.
//!
//! Honest scope: without an embedded perturber there is no gap formation,
//! and without cooling or sink particles there is no planet formation.
//! What we model is gas compression and spiral arms from gravitational
//! instability.

use crate::particle::{ParticleType, Particles};
use crate::scenario::Scenario;
use crate::units::G;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Protoplanetary disk initial conditions.
pub struct Protoplanetary {
    /// Number of gas particles in the disk.
    pub n_gas: usize,
    /// Central star mass.
    pub m_star: f64,
    /// Disk mass as fraction of star mass.
    pub disk_mass_fraction: f64,
    /// Inner disk radius.
    pub r_in: f64,
    /// Outer disk radius.
    pub r_out: f64,
    /// Disk aspect ratio h/r at r_out (controls temperature / sound speed).
    pub aspect_ratio: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for Protoplanetary {
    fn default() -> Self {
        Self {
            n_gas: 5000,
            m_star: 1.0,
            disk_mass_fraction: 0.1,
            r_in: 0.1,
            r_out: 2.0,
            aspect_ratio: 0.05,
            seed: 42,
        }
    }
}

impl Scenario for Protoplanetary {
    fn name(&self) -> &str {
        "Protoplanetary Disk"
    }

    fn generate(&self) -> Particles {
        let n_total = self.n_gas + 1; // +1 for central star
        let mut particles = Particles::new(n_total);
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);

        let m_disk = self.disk_mass_fraction * self.m_star;
        let m_gas = m_disk / self.n_gas as f64;

        // Central star: massive, gravity-only particle at origin
        particles.add_typed(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.m_star, ParticleType::Default as u8);

        // Gas disk particles
        // Surface density Σ ∝ r^{-1} → CDF inversion: r = r_in * (r_out/r_in)^U
        let ln_ratio = (self.r_out / self.r_in).ln();

        for _ in 0..self.n_gas {
            // Radial sampling from Σ ∝ r^{-1}
            let u: f64 = rand::Rng::random::<f64>(&mut rng);
            let r = self.r_in * (u * ln_ratio).exp();

            // Azimuthal angle: uniform
            let phi: f64 = rand::Rng::random::<f64>(&mut rng) * 2.0 * std::f64::consts::PI;

            // Vertical structure: flared disk, z ~ h(r) * gaussian
            // h(r) = aspect_ratio * r (constant h/r for simplicity)
            let h_r = self.aspect_ratio * r;
            let z = h_r * rand_normal(&mut rng);

            let x = r * phi.cos();
            let y = r * phi.sin();

            // Keplerian velocity with sub-Keplerian pressure support correction
            let v_kep = (G * self.m_star / r).sqrt();
            // For Σ ∝ r^{-1}, T ∝ r^{-1}: d ln P / d ln r ≈ -3
            // v_φ = v_kep * sqrt(1 + (h/r)² * d ln P / d ln r)
            let hr = self.aspect_ratio;
            let v_phi = v_kep * (1.0 + hr * hr * (-3.0)).sqrt().max(0.5);

            let vx = -v_phi * phi.sin();
            let vy = v_phi * phi.cos();
            let vz = 0.0;

            // Internal energy: u = c_s² / (γ-1), with c_s = aspect_ratio * v_kep
            let gamma = 5.0 / 3.0;
            let c_s = self.aspect_ratio * v_kep;
            let internal_energy = c_s * c_s / (gamma - 1.0);

            // Smoothing length from local surface density
            // At radius r with Σ ∝ r^{-1}, local particle spacing ~ sqrt(2πr * Δr / N_local)
            // Approximate: h ~ 1.5 * (2π r² / n_gas)^{1/2} * (scale_height factor)
            let h_sml = 1.5 * (2.0 * std::f64::consts::PI * r * r / self.n_gas as f64).sqrt();

            particles.add_gas(x, y, z, vx, vy, vz, m_gas, internal_energy, h_sml);
        }

        particles
    }

    fn suggested_dt(&self) -> f64 {
        // Orbital period at inner edge: T = 2π * sqrt(r_in³ / (G * M_star))
        let t_inner = 2.0 * std::f64::consts::PI
            * (self.r_in.powi(3) / (G * self.m_star)).sqrt();
        // ~100 steps per inner orbit; CFL will refine
        t_inner / 100.0
    }

    fn suggested_softening(&self) -> f64 {
        // Softening ~ fraction of inner radius
        self.r_in * 0.2
    }
}

/// Box-Muller transform for a standard normal variate.
fn rand_normal(rng: &mut impl rand::Rng) -> f64 {
    let u1: f64 = rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
