//! Neighbor list construction for SPH using the octree ball query.
//!
//! Stores neighbor lists as a flat array with per-particle offset ranges,
//! avoiding per-particle heap allocations and improving cache locality.

use crate::octree::Octree;
use crate::particle::Particles;

/// Flat neighbor list storage with per-particle (start, end) offsets.
///
/// For gas particle i, its neighbors are `indices[offsets[i].0 .. offsets[i].1]`.
/// Non-gas particles have `offsets[i] = (0, 0)` (empty range).
pub struct NeighborList {
    /// Flat array of neighbor particle indices.
    pub indices: Vec<u32>,
    /// Per-particle (start, end) offsets into `indices`.
    /// Length == particles.count.
    pub offsets: Vec<(u32, u32)>,
}

impl NeighborList {
    /// Get the neighbor indices for particle i.
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        let (start, end) = self.offsets[i];
        &self.indices[start as usize..end as usize]
    }

    /// Number of neighbors for particle i.
    #[inline]
    pub fn count(&self, i: usize) -> usize {
        let (start, end) = self.offsets[i];
        (end - start) as usize
    }

    /// Replace the neighbor list for a single particle by appending new
    /// neighbors to the end of the flat storage. The old entries become
    /// orphaned but are harmless — the list is rebuilt every timestep.
    pub fn replace_neighbors(&mut self, i: usize, new_neighbors: &[u32]) {
        let start = self.indices.len() as u32;
        self.indices.extend_from_slice(new_neighbors);
        let end = self.indices.len() as u32;
        self.offsets[i] = (start, end);
    }
}

/// Build neighbor lists for all gas particles.
///
/// For each gas particle, finds all particles (gas and non-gas) within
/// `search_radius_factor * 2 * h_i`. The factor > 1 provides padding so
/// that if h changes slightly during the density iteration, we don't need
/// to re-query.
///
/// On native (with rayon), the searches run in parallel. On WASM (single-threaded),
/// they run sequentially.
pub fn build_neighbor_lists(
    particles: &Particles,
    tree: &Octree,
    search_radius_factor: f64,
) -> NeighborList {
    let n = particles.count;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        // Each thread collects its own (index, Vec<u32>) pairs
        let thread_results: Vec<(usize, Vec<u32>)> = (0..n)
            .into_par_iter()
            .filter(|&i| particles.is_gas(i))
            .map(|i| {
                let h = particles.smoothing_length[i];
                let radius = 2.0 * h * search_radius_factor;
                let pos = [particles.x[i], particles.y[i], particles.z[i]];
                let mut buf = Vec::new();
                tree.query_ball(pos, radius, particles, &mut buf);
                (i, buf)
            })
            .collect();

        // Flatten into flat storage
        let total: usize = thread_results.iter().map(|(_, v)| v.len()).sum();
        let mut indices = Vec::with_capacity(total);
        let mut offsets = vec![(0u32, 0u32); n];

        for (i, neighbors) in thread_results {
            let start = indices.len() as u32;
            indices.extend_from_slice(&neighbors);
            let end = indices.len() as u32;
            offsets[i] = (start, end);
        }

        NeighborList { indices, offsets }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut indices = Vec::new();
        let mut offsets = vec![(0u32, 0u32); n];
        let mut buf = Vec::new();

        for i in 0..n {
            if !particles.is_gas(i) {
                continue;
            }
            let h = particles.smoothing_length[i];
            let radius = 2.0 * h * search_radius_factor;
            let pos = [particles.x[i], particles.y[i], particles.z[i]];
            buf.clear();
            tree.query_ball(pos, radius, particles, &mut buf);
            let start = indices.len() as u32;
            indices.extend_from_slice(&buf);
            let end = indices.len() as u32;
            offsets[i] = (start, end);
        }

        NeighborList { indices, offsets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particles;

    #[test]
    fn brute_force_matches_tree_search() {
        // Create a small grid of gas particles and verify tree-based
        // neighbor search matches brute-force.
        let n_side = 5;
        let spacing = 0.5;
        let h = 0.8;
        let mut particles = Particles::new(n_side * n_side * n_side);

        for ix in 0..n_side {
            for iy in 0..n_side {
                for iz in 0..n_side {
                    let x = ix as f64 * spacing;
                    let y = iy as f64 * spacing;
                    let z = iz as f64 * spacing;
                    particles.add_gas(x, y, z, 0.0, 0.0, 0.0, 1.0, 1.0, h);
                }
            }
        }

        let tree = Octree::build(&particles);
        let nlist = build_neighbor_lists(&particles, &tree, 1.0);

        // Verify against brute force for each particle
        let n = particles.count;
        for i in 0..n {
            let radius = 2.0 * h;
            let radius_sq = radius * radius;

            // Brute-force neighbors
            let mut bf_neighbors: Vec<u32> = Vec::new();
            for j in 0..n {
                let dx = particles.x[j] - particles.x[i];
                let dy = particles.y[j] - particles.y[i];
                let dz = particles.z[j] - particles.z[i];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq <= radius_sq {
                    bf_neighbors.push(j as u32);
                }
            }
            bf_neighbors.sort();

            // Tree neighbors
            let mut tree_neighbors: Vec<u32> = nlist.neighbors(i).to_vec();
            tree_neighbors.sort();

            assert_eq!(
                bf_neighbors, tree_neighbors,
                "Mismatch at particle {}: brute_force={:?}, tree={:?}",
                i, bf_neighbors, tree_neighbors
            );
        }
    }

    #[test]
    fn non_gas_particles_have_empty_neighbors() {
        let mut particles = Particles::new(10);
        // Add some non-gas particles
        for i in 0..5 {
            particles.add(i as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }
        // Add some gas particles
        for i in 0..5 {
            particles.add_gas(i as f64, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.8);
        }

        let tree = Octree::build(&particles);
        let nlist = build_neighbor_lists(&particles, &tree, 1.0);

        // Non-gas particles should have empty neighbor lists
        for i in 0..5 {
            assert_eq!(nlist.count(i), 0, "Non-gas particle {} has neighbors", i);
        }
        // Gas particles should have neighbors
        for i in 5..10 {
            assert!(nlist.count(i) > 0, "Gas particle {} has no neighbors", i);
        }
    }
}
