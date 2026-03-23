//! Kelvin-Helmholtz instability: two shearing fluid layers.
//!
//! This is SPH's hardest classical test. Two layers of gas in relative shear
//! motion with a density contrast develop characteristic roll-up vortices at
//! the interface. Traditional SPH notoriously suppresses mixing at density
//! discontinuities; the viscosity switch and artificial conductivity help.
//!
//! Setup: periodic box, equal pressure, density ratio 2:1, velocity shear
//! with a sinusoidal perturbation to seed the instability.

use crate::particle::Particles;
use crate::scenario::Scenario;

/// Kelvin-Helmholtz instability initial conditions.
pub struct KelvinHelmholtz {
    /// Number of particles per side in x and z. Total ≈ nx * nx * 3 (y layers).
    pub nx: usize,
    /// Box size.
    pub box_size: f64,
    /// Density of the high-density layer (central strip).
    pub rho_high: f64,
    /// Density of the low-density layers (top/bottom).
    pub rho_low: f64,
    /// Shear velocity magnitude.
    pub v_shear: f64,
    /// Perturbation amplitude.
    pub perturbation: f64,
    /// Pressure (uniform).
    pub pressure: f64,
    /// Adiabatic index.
    pub gamma: f64,
}

impl Default for KelvinHelmholtz {
    fn default() -> Self {
        Self {
            nx: 64,
            box_size: 1.0,
            rho_high: 2.0,
            rho_low: 1.0,
            v_shear: 0.5,
            perturbation: 0.025,
            pressure: 2.5,
            gamma: 5.0 / 3.0,
        }
    }
}

impl Scenario for KelvinHelmholtz {
    fn name(&self) -> &str {
        "Kelvin-Helmholtz Instability"
    }

    fn generate(&self) -> Particles {
        let l = self.box_size;
        let dx_low = l / self.nx as f64;
        // Adjust spacing for high-density layer to match density ratio
        let dx_high = dx_low * (self.rho_low / self.rho_high).cbrt();

        let mass_low = self.rho_low * dx_low * dx_low * dx_low;
        let mass_high = self.rho_high * dx_high * dx_high * dx_high;

        let u = self.pressure / ((self.gamma - 1.0) * self.rho_low); // Same P everywhere
        let u_high = self.pressure / ((self.gamma - 1.0) * self.rho_high);

        // y boundaries: [0, L], high-density layer in [L/4, 3L/4]
        let y_lo = l / 4.0;
        let y_hi = 3.0 * l / 4.0;

        let h_low = 1.5 * dx_low;
        let h_high = 1.5 * dx_high;

        // Estimate total particles
        let n_est = (self.nx as f64).powi(3) as usize * 2;
        let mut particles = Particles::new(n_est);

        // Low-density region: y in [0, L/4) and [3L/4, L]
        let nz = self.nx; // Same count in z
        {
            let ny_bottom = (y_lo / dx_low) as usize;
            let ny_top = ((l - y_hi) / dx_low) as usize;

            for iz in 0..nz {
                for ix in 0..self.nx {
                    let x = (ix as f64 + 0.5) * dx_low;
                    let z = (iz as f64 + 0.5) * dx_low;

                    // Bottom region
                    for iy in 0..ny_bottom {
                        let y = (iy as f64 + 0.5) * dx_low;
                        let vy_pert = self.perturbation
                            * (2.0 * std::f64::consts::PI * x / l).sin();
                        particles.add_gas(
                            x, y, z,
                            -self.v_shear, vy_pert, 0.0,
                            mass_low, u, h_low,
                        );
                    }

                    // Top region
                    for iy in 0..ny_top {
                        let y = y_hi + (iy as f64 + 0.5) * dx_low;
                        let vy_pert = self.perturbation
                            * (2.0 * std::f64::consts::PI * x / l).sin();
                        particles.add_gas(
                            x, y, z,
                            -self.v_shear, vy_pert, 0.0,
                            mass_low, u, h_low,
                        );
                    }
                }
            }
        }

        // High-density region: y in [L/4, 3L/4]
        {
            let ny_mid = ((y_hi - y_lo) / dx_high) as usize;
            let nz_high = (l / dx_high) as usize;
            let nx_high = (l / dx_high) as usize;

            for iz in 0..nz_high {
                for ix in 0..nx_high {
                    let x = (ix as f64 + 0.5) * dx_high;
                    let z = (iz as f64 + 0.5) * dx_high;

                    for iy in 0..ny_mid {
                        let y = y_lo + (iy as f64 + 0.5) * dx_high;
                        let vy_pert = self.perturbation
                            * (2.0 * std::f64::consts::PI * x / l).sin();
                        particles.add_gas(
                            x, y, z,
                            self.v_shear, vy_pert, 0.0,
                            mass_high, u_high, h_high,
                        );
                    }
                }
            }
        }

        particles
    }

    fn suggested_dt(&self) -> f64 {
        1e-3 // Will be overridden by CFL
    }

    fn suggested_softening(&self) -> f64 {
        0.001 // Minimal gravity softening for hydro test
    }
}
