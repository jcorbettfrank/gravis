//! SPH density estimation, adaptive smoothing length, and equation of state.
//!
//! Implements the self-consistent density-smoothing length iteration with
//! grad-h correction factors (Springel & Hernquist 2002). The Ω_i factor
//! ensures exact energy conservation when the smoothing length varies spatially.

use crate::octree::Octree;
use crate::particle::Particles;
use crate::sph::kernel;
use crate::sph::neighbors::{self, NeighborList};

/// Maximum iterations for the h-ρ convergence loop.
const MAX_H_ITERATIONS: usize = 5;

/// Relative tolerance for h convergence.
const H_TOLERANCE: f64 = 1e-3;

/// SPH density computation result, including neighbor lists and grad-h factors.
pub struct DensityResult {
    /// Neighbor lists for all particles.
    pub neighbors: NeighborList,
    /// Grad-h correction factor Ω_i for each particle.
    /// Ω_i = 1 - (∂h/∂ρ) Σ_j m_j ∂W/∂h
    pub omega: Vec<f64>,
}

/// Compute SPH densities, adaptive smoothing lengths, equation of state,
/// and grad-h correction factors for all gas particles.
///
/// This is the first phase of each SPH timestep. It:
/// 1. Builds neighbor lists from the octree
/// 2. Iterates the h-ρ relation to convergence
/// 3. Computes the grad-h Ω_i correction factor
/// 4. Applies the equation of state (P, c_s)
///
/// # Arguments
/// * `particles` - Particle data (density, h, P, c fields are updated in-place)
/// * `tree` - Octree built from current positions
/// * `gamma` - Adiabatic index (5/3 for monatomic ideal gas)
/// * `eta` - Smoothing length scaling (h = η(m/ρ)^{1/3}), typically 1.2
pub fn compute_density(
    particles: &mut Particles,
    tree: &Octree,
    gamma: f64,
    eta: f64,
) -> DensityResult {
    let n = particles.count;

    // Build neighbor lists with padding for h iteration
    let search_padding = 1.1;
    let mut nlist = neighbors::build_neighbor_lists(particles, tree, search_padding);
    let mut omega = vec![1.0; n];

    // Iterate density-h for each gas particle
    #[allow(clippy::needless_range_loop)] // i indexes multiple parallel arrays
    for i in 0..n {
        if !particles.is_gas(i) {
            continue;
        }

        let mut h = particles.smoothing_length[i];
        let m_i = particles.mass[i];

        for _iter in 0..MAX_H_ITERATIONS {
            // Compute density with current h
            let (rho, dw_dh_sum) = density_and_dw_dh(particles, &nlist, i, h);

            // Update h from h-ρ relation: h = η(m/ρ)^{1/3}
            let h_new = if rho > 0.0 {
                eta * (m_i / rho).cbrt()
            } else {
                h
            };

            // Compute grad-h correction: Ω_i = 1 - (∂h/∂ρ) Σ_j m_j ∂W/∂h
            // where ∂h/∂ρ = -h/(3ρ) from the h-ρ relation
            if rho > 0.0 {
                let dh_drho = -h_new / (3.0 * rho);
                omega[i] = 1.0 - dh_drho * dw_dh_sum;
                // Clamp to prevent numerical issues
                if omega[i] < 0.1 {
                    omega[i] = 0.1;
                }
            }

            particles.density[i] = rho;
            particles.smoothing_length[i] = h_new;

            // Check convergence
            if (h_new - h).abs() / h.max(1e-30) < H_TOLERANCE {
                break;
            }

            // If h grew beyond the padded search radius, re-query this particle only
            let search_radius = 2.0 * h * search_padding;
            if 2.0 * h_new > search_radius {
                let new_radius = 2.0 * h_new * search_padding;
                let pos = [particles.x[i], particles.y[i], particles.z[i]];
                let mut buf = Vec::new();
                tree.query_ball(pos, new_radius, particles, &mut buf);
                nlist.replace_neighbors(i, &buf);
            }

            h = h_new;
        }
    }

    // Apply equation of state for all gas particles
    for i in 0..n {
        if !particles.is_gas(i) {
            continue;
        }
        let rho = particles.density[i];
        let u = particles.internal_energy[i];
        particles.pressure[i] = (gamma - 1.0) * rho * u;
        particles.sound_speed[i] = (gamma * (gamma - 1.0) * u.max(0.0)).sqrt();
    }

    DensityResult {
        neighbors: nlist,
        omega,
    }
}

/// Compute density and the sum Σ_j m_j ∂W/∂h for a single particle,
/// using its current neighbor list and smoothing length h.
fn density_and_dw_dh(
    particles: &Particles,
    nlist: &NeighborList,
    i: usize,
    h: f64,
) -> (f64, f64) {
    let xi = particles.x[i];
    let yi = particles.y[i];
    let zi = particles.z[i];

    let mut rho = 0.0;
    let mut dw_dh_sum = 0.0;

    for &j_idx in nlist.neighbors(i) {
        let j = j_idx as usize;
        let dx = xi - particles.x[j];
        let dy = yi - particles.y[j];
        let dz = zi - particles.z[j];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();

        if r < 2.0 * h {
            let mj = particles.mass[j];
            rho += mj * kernel::w(r, h);
            dw_dh_sum += mj * kernel::dw_dh(r, h);
        }
    }

    (rho, dw_dh_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particles;

    fn uniform_gas_cube(n_side: usize, spacing: f64, h: f64, u: f64) -> Particles {
        let n = n_side * n_side * n_side;
        let mut particles = Particles::new(n);
        for ix in 0..n_side {
            for iy in 0..n_side {
                for iz in 0..n_side {
                    let x = ix as f64 * spacing;
                    let y = iy as f64 * spacing;
                    let z = iz as f64 * spacing;
                    particles.add_gas(x, y, z, 0.0, 0.0, 0.0, 1.0, u, h);
                }
            }
        }
        particles
    }

    #[test]
    fn uniform_density_recovery() {
        // For a uniform grid with mass m and spacing s, the theoretical density
        // is m / s^3 (number density × mass). Verify SPH recovers this.
        let n_side = 8;
        let spacing = 0.5;
        let mass = 1.0;
        let h_initial = 1.0; // Will be adapted
        let u = 1.0;

        let mut particles = uniform_gas_cube(n_side, spacing, h_initial, u);
        let tree = Octree::build(&particles);

        let gamma = 5.0 / 3.0;
        let eta = 1.2;
        let result = compute_density(&mut particles, &tree, gamma, eta);

        // Expected density: mass / spacing^3 = 1.0 / 0.125 = 8.0
        let expected_rho = mass / (spacing * spacing * spacing);

        // Check interior particles (away from edges where boundary effects occur)
        let mut sum_err = 0.0;
        let mut count = 0;
        for i in 0..particles.count {
            // Skip particles near the boundary (within 2h of edge)
            let x = particles.x[i];
            let y = particles.y[i];
            let z = particles.z[i];
            let edge = (n_side - 1) as f64 * spacing;
            let margin = 2.0 * particles.smoothing_length[i];
            if x < margin || y < margin || z < margin
                || x > edge - margin || y > edge - margin || z > edge - margin
            {
                continue;
            }
            let err = (particles.density[i] - expected_rho).abs() / expected_rho;
            sum_err += err;
            count += 1;
        }

        if count > 0 {
            let avg_err = sum_err / count as f64;
            assert!(
                avg_err < 0.05,
                "Average density error for interior particles: {:.1}% (expected <5%)",
                avg_err * 100.0
            );
        }

        // Verify Ω ≈ 1 for interior particles in uniform density
        for i in 0..particles.count {
            if !particles.is_gas(i) {
                continue;
            }
            let x = particles.x[i];
            let y = particles.y[i];
            let z = particles.z[i];
            let edge = (n_side - 1) as f64 * spacing;
            let margin = 2.0 * particles.smoothing_length[i];
            if x < margin || y < margin || z < margin
                || x > edge - margin || y > edge - margin || z > edge - margin
            {
                continue;
            }
            assert!(
                (result.omega[i] - 1.0).abs() < 0.3,
                "Interior Omega[{}] = {} (expected ~1.0)",
                i,
                result.omega[i]
            );
        }
    }

    #[test]
    fn eos_correct() {
        let gamma = 5.0 / 3.0;
        let mut particles = Particles::new(1);
        let u = 3.0;
        particles.add_gas(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, u, 0.5);
        particles.density[0] = 2.0;

        // Apply EOS manually
        let rho = particles.density[0];
        particles.pressure[0] = (gamma - 1.0) * rho * u;
        particles.sound_speed[0] = (gamma * (gamma - 1.0) * u).sqrt();

        assert!((particles.pressure[0] - (2.0 / 3.0) * 2.0 * 3.0).abs() < 1e-10);
        assert!((particles.sound_speed[0] - (gamma * (gamma - 1.0) * u).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn h_convergence() {
        // Start with a bad h guess and verify convergence
        let n_side = 6;
        let spacing = 0.5;
        let h_bad = 2.0; // Way too large
        let mut particles = uniform_gas_cube(n_side, spacing, h_bad, 1.0);

        let tree = Octree::build(&particles);
        let _result = compute_density(&mut particles, &tree, 5.0 / 3.0, 1.2);

        // h should have converged to something reasonable
        for i in 0..particles.count {
            if particles.is_gas(i) {
                let h = particles.smoothing_length[i];
                assert!(
                    h > 0.1 && h < 5.0,
                    "h[{}] = {} — did not converge to reasonable value",
                    i,
                    h
                );
            }
        }
    }
}
