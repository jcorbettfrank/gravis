use std::sync::Mutex;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::gravity::GravitySolver;
use crate::octree::{Octree, NONE};
use crate::particle::Particles;
use crate::units::G;

/// Barnes-Hut O(N log N) gravity solver.
///
/// Builds an octree from particle positions, then computes forces by
/// walking the tree for each particle. Distant clusters of particles
/// are approximated as single bodies when the opening angle criterion
/// is satisfied: `side_length / distance < theta`.
///
/// Lower theta = more accurate (theta=0 reduces to brute-force).
/// Higher theta = faster but less accurate.
/// Default theta=0.5 gives <1% RMS force error for typical distributions.
pub struct BarnesHut {
    /// Plummer softening length
    pub softening: f64,
    /// Opening angle parameter (default 0.5)
    pub theta: f64,
    /// Cached tree arena to avoid re-allocating each step.
    tree_cache: Mutex<Option<Octree>>,
    /// Squared opening angle, precomputed.
    theta2: f64,
}

impl BarnesHut {
    pub fn new(softening: f64, theta: f64) -> Self {
        Self {
            softening,
            theta,
            tree_cache: Mutex::new(None),
            theta2: theta * theta,
        }
    }

    /// Compute the acceleration on particle `i` by walking the tree.
    /// Uses an iterative stack-based walk to avoid recursive call overhead.
    fn tree_walk_accel(
        &self,
        tree: &Octree,
        px: f64,
        py: f64,
        pz: f64,
        particle_idx: u32,
        eps2: f64,
    ) -> [f64; 3] {
        let mut acc = [0.0f64; 3];
        let theta2 = self.theta2;
        let nodes = &tree.nodes;

        // Explicit stack to avoid recursive function calls.
        // Size 512 = MAX_DEPTH(64) * 8 children, sufficient for worst-case
        // traversal where every level is fully opened (theta=0).
        let mut stack_buf = [0u32; 512];
        let mut stack_len = 1usize;
        stack_buf[0] = 0; // root

        while stack_len > 0 {
            stack_len -= 1;
            let node_idx = stack_buf[stack_len];
            let node = &nodes[node_idx as usize];

            if node.total_mass == 0.0 {
                continue;
            }

            if node.particle_index != NONE {
                // Leaf node
                if node.particle_index == particle_idx {
                    continue;
                }
                Self::add_force(px, py, pz,
                    node.center_of_mass[0], node.center_of_mass[1], node.center_of_mass[2],
                    node.total_mass, eps2, &mut acc);
                continue;
            }

            // Internal node: check opening angle
            let dx = node.center_of_mass[0] - px;
            let dy = node.center_of_mass[1] - py;
            let dz = node.center_of_mass[2] - pz;
            let d2 = dx * dx + dy * dy + dz * dz;
            let s = node.bounds.half_width * 2.0;

            if s * s < theta2 * d2 {
                Self::add_force(px, py, pz,
                    node.center_of_mass[0], node.center_of_mass[1], node.center_of_mass[2],
                    node.total_mass, eps2, &mut acc);
                continue;
            }

            // Push children onto stack
            for &child_idx in &node.children {
                if child_idx != NONE {
                    stack_buf[stack_len] = child_idx;
                    stack_len += 1;
                }
            }
        }

        acc
    }

    /// Add gravitational acceleration from a source (mass at position) onto
    /// the target particle at (px, py, pz).
    #[inline(always)]
    fn add_force(
        px: f64,
        py: f64,
        pz: f64,
        sx: f64,
        sy: f64,
        sz: f64,
        mass: f64,
        eps2: f64,
        acc: &mut [f64; 3],
    ) {
        let dx = sx - px;
        let dy = sy - py;
        let dz = sz - pz;
        let r2 = dx * dx + dy * dy + dz * dz + eps2;
        let inv_r3 = 1.0 / (r2 * r2.sqrt());
        let f = G * mass * inv_r3;

        acc[0] += f * dx;
        acc[1] += f * dy;
        acc[2] += f * dz;
    }
}

impl GravitySolver for BarnesHut {
    fn compute_accelerations(&self, p: &mut Particles) {
        let n = p.count;
        if n == 0 {
            return;
        }

        // Reuse the tree arena from previous calls to avoid re-allocating.
        let mut cache = self.tree_cache.lock().unwrap();
        let tree = match cache.as_mut() {
            Some(tree) => {
                tree.rebuild(p);
                tree
            }
            None => {
                *cache = Some(Octree::build(p));
                cache.as_mut().unwrap()
            }
        };
        let eps2 = self.softening * self.softening;

        // Compute accelerations — each particle walks the tree independently.
        #[cfg(feature = "parallel")]
        let accels: Vec<[f64; 3]> = {
            // Use parallel iteration for large N, sequential for small N where
            // rayon overhead dominates.
            const PAR_THRESHOLD: usize = 1000;
            if n >= PAR_THRESHOLD {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        self.tree_walk_accel(&tree, p.x[i], p.y[i], p.z[i], i as u32, eps2)
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|i| {
                        self.tree_walk_accel(&tree, p.x[i], p.y[i], p.z[i], i as u32, eps2)
                    })
                    .collect()
            }
        };

        #[cfg(not(feature = "parallel"))]
        let accels: Vec<[f64; 3]> = (0..n)
            .map(|i| self.tree_walk_accel(&tree, p.x[i], p.y[i], p.z[i], i as u32, eps2))
            .collect();

        // Accumulate into particle arrays (matching GravitySolver trait contract)
        for i in 0..n {
            p.ax[i] += accels[i][0];
            p.ay[i] += accels[i][1];
            p.az[i] += accels[i][2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity::BruteForce;

    #[test]
    fn two_body_force_approximate_symmetry() {
        let mut p = Particles::new(2);
        p.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        p.add(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let solver = BarnesHut::new(0.0, 0.5);
        solver.compute_accelerations(&mut p);

        // Forces should be approximately equal and opposite
        let tol = 1e-10;
        assert!(
            (p.ax[0] + p.ax[1]).abs() < tol,
            "ax asymmetry: {}",
            (p.ax[0] + p.ax[1]).abs()
        );
    }

    #[test]
    fn theta_zero_matches_brute_force() {
        let mut p_bh = Particles::new(5);
        let mut p_bf = Particles::new(5);

        let positions = [
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 1.5),
            (-1.0, -1.0, 0.5, 0.5),
            (0.5, 0.5, -0.5, 3.0),
        ];

        for &(x, y, z, m) in &positions {
            p_bh.add(x, y, z, 0.0, 0.0, 0.0, m);
            p_bf.add(x, y, z, 0.0, 0.0, 0.0, m);
        }

        let softening = 0.05;
        let bh = BarnesHut::new(softening, 0.0);
        let bf = BruteForce::new(softening);

        bh.compute_accelerations(&mut p_bh);
        bf.compute_accelerations(&mut p_bf);

        for i in 0..5 {
            let tol = 1e-12;
            assert!(
                (p_bh.ax[i] - p_bf.ax[i]).abs() < tol,
                "ax[{i}] mismatch: bh={}, bf={}, diff={}",
                p_bh.ax[i],
                p_bf.ax[i],
                (p_bh.ax[i] - p_bf.ax[i]).abs()
            );
            assert!(
                (p_bh.ay[i] - p_bf.ay[i]).abs() < tol,
                "ay[{i}] mismatch: bh={}, bf={}",
                p_bh.ay[i],
                p_bf.ay[i]
            );
            assert!(
                (p_bh.az[i] - p_bf.az[i]).abs() < tol,
                "az[{i}] mismatch: bh={}, bf={}",
                p_bh.az[i],
                p_bf.az[i]
            );
        }
    }

    #[test]
    fn force_magnitude_correct() {
        let mut p = Particles::new(2);
        p.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        p.add(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let solver = BarnesHut::new(0.0, 0.5);
        solver.compute_accelerations(&mut p);

        let expected = G * 1.0 / 4.0;
        assert!(
            (p.ax[0] - expected).abs() < 1e-14,
            "Force magnitude: got {}, expected {}",
            p.ax[0],
            expected
        );
    }
}
