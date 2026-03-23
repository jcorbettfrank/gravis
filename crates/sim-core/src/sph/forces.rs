//! SPH force computation: pressure gradient, artificial viscosity, and thermal conductivity.
//!
//! Implements the grad-h corrected momentum equation (Springel & Hernquist 2002),
//! Monaghan artificial viscosity with the Morris & Monaghan (1997) per-particle
//! viscosity switch, and Price (2008) artificial thermal conductivity.

use crate::particle::Particles;
use crate::sph::kernel;
use crate::sph::neighbors::NeighborList;

/// Floor for density and internal energy to prevent division-by-zero divergence.
const DENSITY_FLOOR: f64 = 1e-10;

/// Parameters for SPH force computation.
pub struct ForceParams {
    /// Adiabatic index (5/3 for monatomic ideal gas).
    pub gamma: f64,
    /// Artificial viscosity β parameter (quadratic term). Standard: 2.0.
    pub beta_visc: f64,
    /// Minimum viscosity α (Morris & Monaghan decay floor). Standard: 0.1.
    pub alpha_min: f64,
    /// Maximum viscosity α (shock spike ceiling). Standard: 2.0.
    pub alpha_max: f64,
    /// Artificial thermal conductivity coefficient. Standard: 1.0.
    pub alpha_cond: f64,
    /// Viscosity switch decay rate parameter σ. Standard: 0.1.
    pub sigma_visc: f64,
}

impl Default for ForceParams {
    fn default() -> Self {
        Self {
            gamma: 5.0 / 3.0,
            beta_visc: 2.0,
            alpha_min: 0.1,
            alpha_max: 2.0,
            alpha_cond: 1.0,
            sigma_visc: 0.1,
        }
    }
}

/// Per-particle intermediate values needed for the force loop.
/// Stored in a flat array indexed by particle.
pub struct ForceIntermediates {
    /// Velocity divergence ∇·v for each particle.
    pub div_v: Vec<f64>,
    /// Maximum |μ_ij| encountered for each particle (for CFL).
    pub max_mu: Vec<f64>,
    /// Time derivative of viscosity parameter dα/dt.
    pub dalpha_dt: Vec<f64>,
}

/// Compute SPH accelerations, du/dt, and viscosity switch for all gas particles.
///
/// This is a **gather-only** implementation: each particle i loops over its
/// neighbors and accumulates forces onto itself. This doubles the work compared
/// to a scatter approach (each pair computed twice) but is trivially parallel
/// and works on WASM.
///
/// # Arguments
/// * `particles` - Particle data. `ax, ay, az` and `du_dt` are accumulated (not cleared here).
/// * `nlist` - Neighbor lists from density computation.
/// * `omega` - Grad-h correction factors Ω_i from density computation.
/// * `params` - Force parameters (viscosity, conductivity coefficients).
///
/// # Returns
/// `ForceIntermediates` with div_v, max_mu, dalpha_dt for CFL and viscosity switch.
pub fn compute_forces(
    particles: &mut Particles,
    nlist: &NeighborList,
    omega: &[f64],
    params: &ForceParams,
) -> ForceIntermediates {
    let n = particles.count;
    let mut div_v = vec![0.0; n];
    let mut max_mu = vec![0.0f64; n];
    let mut dalpha_dt = vec![0.0; n];

    for i in 0..n {
        if !particles.is_gas(i) {
            continue;
        }

        let xi = particles.x[i];
        let yi = particles.y[i];
        let zi = particles.z[i];
        let vxi = particles.vx[i];
        let vyi = particles.vy[i];
        let vzi = particles.vz[i];
        let hi = particles.smoothing_length[i];
        let rho_i = particles.density[i].max(DENSITY_FLOOR);
        let p_i = particles.pressure[i];
        let u_i = particles.internal_energy[i];
        let c_i = particles.sound_speed[i];
        let alpha_i = particles.alpha_visc[i];
        let omega_i = omega[i];

        // Pressure term for particle i: P_i / (Ω_i ρ_i²)
        let p_over_rho2_i = if rho_i > 0.0 {
            p_i / (omega_i * rho_i * rho_i)
        } else {
            0.0
        };

        let mut ax_acc = 0.0;
        let mut ay_acc = 0.0;
        let mut az_acc = 0.0;
        let mut du_dt_acc = 0.0;
        let mut div_v_acc = 0.0;
        let mut max_mu_i = 0.0f64;

        for &j_idx in nlist.neighbors(i) {
            let j = j_idx as usize;
            if j == i {
                continue;
            }

            let dx = xi - particles.x[j];
            let dy = yi - particles.y[j];
            let dz = zi - particles.z[j];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();

            let hj = particles.smoothing_length[j];
            let h_avg = 0.5 * (hi + hj);

            // Skip if outside both kernel supports (h_avg <= max(hi,hj), so redundant)
            if r >= 2.0 * hi && r >= 2.0 * hj {
                continue;
            }

            let mj = particles.mass[j];
            let rho_j = particles.density[j].max(DENSITY_FLOOR);
            let p_j = particles.pressure[j];
            let c_j = particles.sound_speed[j];
            let alpha_j = particles.alpha_visc[j];
            let omega_j = omega[j];

            // Relative velocity v_ij = v_i - v_j
            let dvx = vxi - particles.vx[j];
            let dvy = vyi - particles.vy[j];
            let dvz = vzi - particles.vz[j];
            let vr = dvx * dx + dvy * dy + dvz * dz; // v_ij · r_ij

            // Kernel gradients
            let grad_wi = kernel::grad_w(dx, dy, dz, r, hi);
            let grad_wj = kernel::grad_w(dx, dy, dz, r, hj);
            let grad_w_avg = kernel::grad_w(dx, dy, dz, r, h_avg);

            // Pressure term for particle j: P_j / (Ω_j ρ_j²)
            let p_over_rho2_j = if rho_j > 0.0 {
                p_j / (omega_j * rho_j * rho_j)
            } else {
                0.0
            };

            // --- Artificial viscosity (Monaghan) ---
            let pi_ij = if vr < 0.0 {
                let alpha_avg = 0.5 * (alpha_i + alpha_j);
                let c_avg = 0.5 * (c_i + c_j);
                let rho_avg = 0.5 * (rho_i + rho_j);
                let eta2 = 0.01 * h_avg * h_avg;
                let mu = h_avg * vr / (r * r + eta2);
                max_mu_i = max_mu_i.max(mu.abs());

                if rho_avg > 0.0 {
                    (-alpha_avg * c_avg * mu + params.beta_visc * mu * mu) / rho_avg
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // --- Momentum equation ---
            // dv/dt_i = -Σ_j m_j [P_i/(Ω_i ρ_i²) ∇W(h_i) + P_j/(Ω_j ρ_j²) ∇W(h_j) + Π_ij ∇W̄]
            ax_acc -= mj * (p_over_rho2_i * grad_wi[0] + p_over_rho2_j * grad_wj[0] + pi_ij * grad_w_avg[0]);
            ay_acc -= mj * (p_over_rho2_i * grad_wi[1] + p_over_rho2_j * grad_wj[1] + pi_ij * grad_w_avg[1]);
            az_acc -= mj * (p_over_rho2_i * grad_wi[2] + p_over_rho2_j * grad_wj[2] + pi_ij * grad_w_avg[2]);

            // --- Energy equation ---
            // du/dt = P_i/(Ω_i ρ_i²) Σ_j m_j v_ij · ∇W(h_i) + ½ Σ_j m_j Π_ij v_ij · ∇W̄
            let vdot_grad_wi = dvx * grad_wi[0] + dvy * grad_wi[1] + dvz * grad_wi[2];
            let vdot_grad_wavg = dvx * grad_w_avg[0] + dvy * grad_w_avg[1] + dvz * grad_w_avg[2];
            du_dt_acc += mj * (p_over_rho2_i * vdot_grad_wi + 0.5 * pi_ij * vdot_grad_wavg);

            // --- Velocity divergence (for viscosity switch) ---
            // ∇·v_i = -(1/ρ_i) Σ_j m_j v_ij · ∇W_ij
            div_v_acc -= mj * vdot_grad_wi;

            // --- Artificial thermal conductivity (Price 2008) ---
            if params.alpha_cond > 0.0 {
                let u_j = particles.internal_energy[j];
                let rho_sum = rho_i + rho_j;
                if rho_sum > 0.0 {
                    let rho_avg = 0.5 * rho_sum;
                    let dp = (p_i - p_j).abs();
                    let v_sig_u = (dp / rho_avg).sqrt();
                    let abs_gw = kernel::abs_grad_w(r, h_avg);
                    du_dt_acc += mj * params.alpha_cond * v_sig_u * (u_i - u_j) / rho_sum * abs_gw;
                }
            }
        }

        particles.ax[i] += ax_acc;
        particles.ay[i] += ay_acc;
        particles.az[i] += az_acc;
        particles.du_dt[i] += du_dt_acc;

        if rho_i > 0.0 {
            div_v[i] = div_v_acc / rho_i;
        }
        max_mu[i] = max_mu_i;

        // --- Morris & Monaghan viscosity switch ---
        // dα/dt = -(α - α_min)/τ + S
        // τ = h/(2σc), S = max(-∇·v, 0) · (α_max - α)
        let tau = if c_i > 0.0 {
            hi / (2.0 * params.sigma_visc * c_i)
        } else {
            1e10 // Large decay time if c=0
        };
        let source = (-div_v[i]).max(0.0) * (params.alpha_max - alpha_i);
        dalpha_dt[i] = -(alpha_i - params.alpha_min) / tau + source;
    }

    ForceIntermediates {
        div_v,
        max_mu,
        dalpha_dt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::Octree;
    use crate::particle::Particles;
    use crate::sph::density;

    #[test]
    fn uniform_pressure_zero_acceleration() {
        // A uniform density/pressure field should produce zero net SPH acceleration
        // for interior particles.
        let n_side = 8;
        let spacing = 0.5;
        let u = 1.0;
        let h = 1.0;
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

        let tree = Octree::build(&particles);
        let gamma = 5.0 / 3.0;
        let result = density::compute_density(&mut particles, &tree, gamma, 1.2);
        particles.clear_accelerations();
        particles.clear_sph_rates();

        let params = ForceParams::default();
        let _intermediates = compute_forces(&mut particles, &result.neighbors, &result.omega, &params);

        // Check interior particles have near-zero acceleration
        let edge = (n_side - 1) as f64 * spacing;
        for i in 0..particles.count {
            if !particles.is_gas(i) {
                continue;
            }
            let margin = 2.5 * particles.smoothing_length[i];
            let x = particles.x[i];
            let y = particles.y[i];
            let z = particles.z[i];
            if x < margin || y < margin || z < margin
                || x > edge - margin || y > edge - margin || z > edge - margin
            {
                continue;
            }
            let a_mag = (particles.ax[i].powi(2) + particles.ay[i].powi(2) + particles.az[i].powi(2)).sqrt();
            assert!(
                a_mag < 1.0,
                "Interior particle {} has |a| = {} (expected ~0)",
                i, a_mag
            );
        }
    }

    #[test]
    fn viscosity_only_for_approaching() {
        // Two gas particles: viscosity should activate only when approaching
        let mut particles = Particles::new(2);
        let h = 1.0;
        // Particle 0 at origin, particle 1 at x=1, approaching
        particles.add_gas(0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0, h);
        particles.add_gas(1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 1.0, 1.0, h);

        let tree = Octree::build(&particles);
        let gamma = 5.0 / 3.0;
        let result = density::compute_density(&mut particles, &tree, gamma, 1.2);
        particles.clear_accelerations();
        particles.clear_sph_rates();

        let params = ForceParams::default();
        let intermediates = compute_forces(&mut particles, &result.neighbors, &result.omega, &params);

        // v_ij · r_ij < 0 for approaching particles, so viscosity should be active
        assert!(intermediates.max_mu[0] > 0.0, "Viscosity should activate for approaching particles");

        // Now test receding particles
        let mut particles2 = Particles::new(2);
        particles2.add_gas(0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 1.0, 1.0, h);
        particles2.add_gas(1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0, h);

        let tree2 = Octree::build(&particles2);
        let result2 = density::compute_density(&mut particles2, &tree2, gamma, 1.2);
        particles2.clear_accelerations();
        particles2.clear_sph_rates();

        let intermediates2 = compute_forces(&mut particles2, &result2.neighbors, &result2.omega, &params);

        // v_ij · r_ij > 0 for receding particles, so viscosity should not activate
        assert_eq!(intermediates2.max_mu[0], 0.0, "Viscosity should not activate for receding particles");
    }
}
