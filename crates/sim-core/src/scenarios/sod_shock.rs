//! Sod shock tube: the standard 1D verification test for SPH.
//!
//! A membrane separates two gas states:
//! - Left:  ρ=1.0, P=1.0, v=0
//! - Right: ρ=0.125, P=0.1, v=0
//!
//! At t=0 the membrane is removed. The analytical Riemann solution produces
//! a left-going rarefaction fan, a contact discontinuity, and a right-going shock.
//!
//! We map this to 3D by using a slab of particles with several layers in y and z.

use crate::particle::Particles;
use crate::scenario::Scenario;

/// Sod shock tube initial conditions.
pub struct SodShockTube {
    /// Number of particles along x on the left (high-density) side.
    pub nx_left: usize,
    /// Number of particles along x on the right (low-density) side.
    pub nx_right: usize,
    /// Number of particle layers in y and z (same for both).
    pub nyz: usize,
    /// Domain half-length in x (membrane at x=0, domain [-x_extent, x_extent]).
    pub x_extent: f64,
    /// Adiabatic index.
    pub gamma: f64,
}

impl Default for SodShockTube {
    fn default() -> Self {
        Self {
            nx_left: 120,
            nx_right: 30,
            nyz: 4,
            x_extent: 0.5,
            gamma: 5.0 / 3.0,
        }
    }
}

impl Scenario for SodShockTube {
    fn name(&self) -> &str {
        "Sod Shock Tube"
    }

    fn generate(&self) -> Particles {
        let n_left = self.nx_left * self.nyz * self.nyz;
        let n_right = self.nx_right * self.nyz * self.nyz;
        let mut particles = Particles::new(n_left + n_right);

        // Left side: ρ_L = 1.0, P_L = 1.0
        let rho_l = 1.0;
        let p_l = 1.0;
        let u_l = p_l / ((self.gamma - 1.0) * rho_l);
        let dx_l = self.x_extent / self.nx_left as f64;
        let dy_l = dx_l; // Equal spacing in y/z
        let mass_l = rho_l * dx_l * dy_l * dy_l;
        let h_l = 2.0 * dx_l;

        for ix in 0..self.nx_left {
            for iy in 0..self.nyz {
                for iz in 0..self.nyz {
                    let x = -self.x_extent + (ix as f64 + 0.5) * dx_l;
                    let y = (iy as f64 + 0.5) * dy_l;
                    let z = (iz as f64 + 0.5) * dy_l;
                    particles.add_gas(x, y, z, 0.0, 0.0, 0.0, mass_l, u_l, h_l);
                }
            }
        }

        // Right side: ρ_R = 0.125, P_R = 0.1
        let rho_r = 0.125;
        let p_r = 0.1;
        let u_r = p_r / ((self.gamma - 1.0) * rho_r);
        let dx_r = self.x_extent / self.nx_right as f64;
        let dy_r = dx_r;
        let mass_r = rho_r * dx_r * dy_r * dy_r;
        let h_r = 2.0 * dx_r;

        for ix in 0..self.nx_right {
            for iy in 0..self.nyz {
                for iz in 0..self.nyz {
                    let x = (ix as f64 + 0.5) * dx_r;
                    let y = (iy as f64 + 0.5) * dy_r;
                    let z = (iz as f64 + 0.5) * dy_r;
                    particles.add_gas(x, y, z, 0.0, 0.0, 0.0, mass_r, u_r, h_r);
                }
            }
        }

        particles
    }

    fn suggested_dt(&self) -> f64 {
        1e-4 // Will be overridden by CFL adaptive timestep
    }

    fn suggested_softening(&self) -> f64 {
        0.01
    }
}

/// Y/Z extent for boundary conditions (based on particle spacing).
impl SodShockTube {
    /// Compute the y/z extent for reflective boundaries.
    pub fn yz_extent(&self) -> f64 {
        let dx_l = self.x_extent / self.nx_left as f64;
        self.nyz as f64 * dx_l
    }
}

/// Analytical Riemann solution for the Sod problem at time t.
///
/// Returns (density, velocity, pressure) evaluated at the given x coordinates.
///
/// Reference: Toro (2009), "Riemann Solvers and Numerical Methods for Fluid Dynamics"
pub fn sod_analytical(x: &[f64], t: f64, gamma: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let rho_l = 1.0;
    let p_l = 1.0;
    let rho_r = 0.125;
    let p_r = 0.1;

    let c_l = (gamma * p_l / rho_l).sqrt();
    let c_r = (gamma * p_r / rho_r).sqrt();

    let gp1 = gamma + 1.0;
    let gm1 = gamma - 1.0;

    // Solve for post-shock pressure p_star iteratively (Newton-Raphson)
    let mut p_star = 0.5 * (p_l + p_r);
    for _ in 0..50 {
        let (fl, dfl) = riemann_f(p_star, rho_l, p_l, c_l, gamma);
        let (fr, dfr) = riemann_f(p_star, rho_r, p_r, c_r, gamma);
        let f_total = fl + fr;
        let df_total = dfl + dfr;
        if df_total.abs() < 1e-30 {
            break;
        }
        let dp = f_total / df_total;
        p_star -= dp;
        if p_star < 1e-10 {
            p_star = 1e-10;
        }
        if dp.abs() / p_star < 1e-10 {
            break;
        }
    }

    // Post-shock velocity
    let (fl, _) = riemann_f(p_star, rho_l, p_l, c_l, gamma);
    let (fr, _) = riemann_f(p_star, rho_r, p_r, c_r, gamma);
    let u_star = 0.5 * (-fl + fr);

    // Post-shock densities
    let rho_star_l = rho_l * (p_star / p_l).powf(1.0 / gamma);
    let c_star_l = (gamma * p_star / rho_star_l).sqrt();
    let rho_star_r = rho_r * (p_star / p_r + gm1 / gp1) / (gm1 / gp1 * p_star / p_r + 1.0);

    // Shock speed
    let s_r = (0.5 * gp1 / gamma * p_star / p_r + 0.5 * gm1 / gamma).sqrt() * c_r;

    // Wave speeds
    let s_hl = -c_l;
    let s_tl = u_star - c_star_l;
    let s_contact = u_star;

    let mut rho_out = Vec::with_capacity(x.len());
    let mut vel_out = Vec::with_capacity(x.len());
    let mut prs_out = Vec::with_capacity(x.len());

    for &xi in x {
        let s = if t > 0.0 { xi / t } else { 0.0 };

        let (rho, vel, prs) = if s < s_hl {
            (rho_l, 0.0, p_l)
        } else if s < s_tl {
            let u_fan = 2.0 / gp1 * (c_l + s);
            let c_fan = c_l - 0.5 * gm1 * u_fan;
            let rho_fan = rho_l * (c_fan / c_l).powf(2.0 / gm1);
            let p_fan = p_l * (c_fan / c_l).powf(2.0 * gamma / gm1);
            (rho_fan, u_fan, p_fan)
        } else if s < s_contact {
            (rho_star_l, u_star, p_star)
        } else if s < s_r {
            (rho_star_r, u_star, p_star)
        } else {
            (rho_r, 0.0, p_r)
        };

        rho_out.push(rho);
        vel_out.push(vel);
        prs_out.push(prs);
    }

    (rho_out, vel_out, prs_out)
}

fn riemann_f(p: f64, rho_k: f64, p_k: f64, c_k: f64, gamma: f64) -> (f64, f64) {
    let gp1 = gamma + 1.0;
    let gm1 = gamma - 1.0;

    if p > p_k {
        let a_k = 2.0 / (gp1 * rho_k);
        let b_k = gm1 / gp1 * p_k;
        let sqrt_term = (a_k / (p + b_k)).sqrt();
        let f = (p - p_k) * sqrt_term;
        let df = sqrt_term * (1.0 - 0.5 * (p - p_k) / (p + b_k));
        (f, df)
    } else {
        let f = 2.0 * c_k / gm1 * ((p / p_k).powf(gm1 / (2.0 * gamma)) - 1.0);
        let df = 1.0 / (rho_k * c_k) * (p / p_k).powf(-(gp1) / (2.0 * gamma));
        (f, df)
    }
}
