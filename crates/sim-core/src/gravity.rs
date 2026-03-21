use crate::particle::Particles;
use crate::units::G;

/// Gravity solver trait. Implementations compute gravitational accelerations
/// for all particles and accumulate them into the acceleration arrays.
pub trait GravitySolver {
    /// Compute and accumulate gravitational accelerations.
    /// Assumes accelerations have been zeroed before calling.
    fn compute_accelerations(&self, particles: &mut Particles);
}

/// Direct summation O(N²) gravity with Plummer softening.
///
/// The force between particles i and j is:
///
///   F_ij = -G * m_i * m_j * r_ij / (|r_ij|² + ε²)^(3/2)
///
/// where ε is the softening length. This prevents the force from
/// diverging as particles approach each other (which would require
/// infinitely small timesteps). Physically, softening represents
/// treating point masses as extended Plummer spheres with scale
/// radius ε.
///
/// The force is antisymmetric (F_ij = -F_ji), so we compute each
/// pair once and apply to both particles. This halves the work and
/// ensures exact momentum conservation.
pub struct BruteForce {
    /// Plummer softening length
    pub softening: f64,
}

impl BruteForce {
    pub fn new(softening: f64) -> Self {
        Self { softening }
    }
}

impl GravitySolver for BruteForce {
    fn compute_accelerations(&self, p: &mut Particles) {
        let eps2 = self.softening * self.softening;
        let n = p.count;

        // Pairwise force computation. For each pair (i, j) with j > i,
        // compute the force once and apply Newton's third law.
        // This guarantees exact momentum conservation (to machine precision).
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = p.x[j] - p.x[i];
                let dy = p.y[j] - p.y[i];
                let dz = p.z[j] - p.z[i];

                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                // r2^(3/2) = r2 * sqrt(r2)
                let inv_r3 = 1.0 / (r2 * r2.sqrt());

                // Acceleration on i due to j: a_i += G * m_j * r_ij / |r_ij|³
                // Acceleration on j due to i: a_j -= G * m_i * r_ij / |r_ij|³
                let ai_factor = G * p.mass[j] * inv_r3;
                let aj_factor = G * p.mass[i] * inv_r3;

                p.ax[i] += ai_factor * dx;
                p.ay[i] += ai_factor * dy;
                p.az[i] += ai_factor * dz;

                p.ax[j] -= aj_factor * dx;
                p.ay[j] -= aj_factor * dy;
                p.az[j] -= aj_factor * dz;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_body_force_symmetry() {
        let mut p = Particles::new(2);
        p.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        p.add(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let solver = BruteForce::new(0.0);
        solver.compute_accelerations(&mut p);

        // Forces should be equal and opposite (Newton's third law)
        let tol = 1e-14;
        assert!((p.ax[0] + p.ax[1]).abs() < tol);
        assert!((p.ay[0] + p.ay[1]).abs() < tol);
        assert!((p.az[0] + p.az[1]).abs() < tol);

        // Both should be attracted toward each other
        assert!(p.ax[0] > 0.0); // particle 0 pulled toward +x
        assert!(p.ax[1] < 0.0); // particle 1 pulled toward -x
    }

    #[test]
    fn force_magnitude_inverse_square() {
        // Two unit masses at distance r=2, no softening
        // F = G*m1*m2/r² = 1/4 = 0.25
        let mut p = Particles::new(2);
        p.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        p.add(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let solver = BruteForce::new(0.0);
        solver.compute_accelerations(&mut p);

        let expected = G * 1.0 / 4.0; // G*m/r²
        assert!((p.ax[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn softening_prevents_singularity() {
        // Two particles at same position — without softening this would diverge
        let mut p = Particles::new(2);
        p.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        p.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let solver = BruteForce::new(0.1);
        solver.compute_accelerations(&mut p);

        // Accelerations should be finite (zero by symmetry)
        assert!(p.ax[0].is_finite());
        assert!(p.ax[1].is_finite());
    }
}
