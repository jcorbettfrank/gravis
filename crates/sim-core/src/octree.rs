use crate::particle::Particles;

/// Axis-aligned cubic bounding box for octree nodes.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// Center of the cube
    pub center: [f64; 3],
    /// Half the side length
    pub half_width: f64,
}

impl BoundingBox {
    /// Compute the smallest cube containing all particles, with a small margin.
    pub fn containing(particles: &Particles) -> Self {
        let n = particles.count;
        assert!(n > 0, "Cannot build bounding box for zero particles");

        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];

        for i in 0..n {
            let pos = [particles.x[i], particles.y[i], particles.z[i]];
            for d in 0..3 {
                min[d] = min[d].min(pos[d]);
                max[d] = max[d].max(pos[d]);
            }
        }

        let center = [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ];

        // Use the largest extent across any axis to make a cube
        let extent = (max[0] - min[0])
            .max(max[1] - min[1])
            .max(max[2] - min[2]);

        // Add small margin to avoid particles sitting exactly on boundaries
        let half_width = 0.5 * extent * 1.001 + 1e-10;

        Self { center, half_width }
    }

    /// Compute the bounding box for a child octant (0..7).
    ///
    /// Octant indexing:
    ///   bit 0 (x): 0 = left,  1 = right
    ///   bit 1 (y): 0 = below, 1 = above
    ///   bit 2 (z): 0 = back,  1 = front
    pub fn child_bounds(&self, octant: usize) -> Self {
        let q = self.half_width * 0.5;
        let center = [
            self.center[0] + if octant & 1 != 0 { q } else { -q },
            self.center[1] + if octant & 2 != 0 { q } else { -q },
            self.center[2] + if octant & 4 != 0 { q } else { -q },
        ];
        Self {
            center,
            half_width: q,
        }
    }

    /// Determine which octant a point falls into relative to this box's center.
    #[inline]
    pub fn octant(&self, x: f64, y: f64, z: f64) -> usize {
        let mut oct = 0;
        if x >= self.center[0] {
            oct |= 1;
        }
        if y >= self.center[1] {
            oct |= 2;
        }
        if z >= self.center[2] {
            oct |= 4;
        }
        oct
    }

    /// Side length of the cube.
    #[inline]
    pub fn side_length(&self) -> f64 {
        2.0 * self.half_width
    }
}

/// Maximum tree depth to prevent infinite recursion when particles overlap.
const MAX_DEPTH: usize = 64;

/// Sentinel value indicating no child/node.
pub const NONE: u32 = u32::MAX;

/// A node in the arena-allocated octree.
///
/// Each node is one of:
/// - **Empty**: total_mass == 0, children all NONE, particle_index == NONE
/// - **Leaf**: particle_index != NONE, children all NONE
/// - **Internal**: has at least one child != NONE, stores aggregate mass and COM
#[derive(Clone)]
pub struct OctreeNode {
    pub bounds: BoundingBox,
    pub total_mass: f64,
    pub center_of_mass: [f64; 3],
    /// Indices into the arena for each of 8 children. NONE means no child.
    pub children: [u32; 8],
    /// Index of the particle stored in this leaf node. NONE if not a leaf.
    pub particle_index: u32,
}

impl OctreeNode {
    fn empty(bounds: BoundingBox) -> Self {
        Self {
            bounds,
            total_mass: 0.0,
            center_of_mass: [0.0; 3],
            children: [NONE; 8],
            particle_index: NONE,
        }
    }

    /// Whether this node contains no particles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_mass == 0.0
    }

    /// Whether this node is a leaf (contains exactly one particle).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.particle_index != NONE
    }
}

/// Arena-allocated octree for Barnes-Hut force calculation.
///
/// All nodes are stored in a contiguous `Vec`, avoiding per-node heap
/// allocations. Children are referenced by index into the arena.
pub struct Octree {
    pub nodes: Vec<OctreeNode>,
}

impl Octree {
    /// Build an octree from the current particle positions and masses.
    pub fn build(particles: &Particles) -> Self {
        let n = particles.count;
        let mut tree = Octree {
            nodes: Vec::with_capacity(2 * n + 1),
        };
        tree.rebuild(particles);
        tree
    }

    /// Clear and rebuild the tree from new particle positions.
    /// Reuses the existing allocation to avoid repeated large allocations.
    pub fn rebuild(&mut self, particles: &Particles) {
        self.nodes.clear();
        let bounds = BoundingBox::containing(particles);
        self.nodes.push(OctreeNode::empty(bounds));

        for i in 0..particles.count {
            self.insert(0, i as u32, particles.x[i], particles.y[i], particles.z[i], particles.mass[i], 0);
        }
    }

    /// Root node (always at index 0).
    #[inline]
    pub fn root(&self) -> &OctreeNode {
        &self.nodes[0]
    }

    /// Get a node by index.
    #[inline]
    pub fn node(&self, idx: u32) -> &OctreeNode {
        &self.nodes[idx as usize]
    }

    /// Allocate a new empty node and return its index.
    #[inline]
    fn alloc(&mut self, bounds: BoundingBox) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(OctreeNode::empty(bounds));
        idx
    }

    /// Insert a particle into the subtree rooted at `node_idx`.
    #[allow(clippy::too_many_arguments)]
    fn insert(&mut self, node_idx: u32, particle_idx: u32, px: f64, py: f64, pz: f64, mass: f64, depth: usize) {
        if self.nodes[node_idx as usize].is_empty() {
            let node = &mut self.nodes[node_idx as usize];
            node.particle_index = particle_idx;
            node.total_mass = mass;
            node.center_of_mass = [px, py, pz];
            return;
        }

        if depth >= MAX_DEPTH {
            Self::update_com(&mut self.nodes[node_idx as usize], px, py, pz, mass);
            return;
        }

        let existing = self.nodes[node_idx as usize].particle_index;
        if existing != NONE {
            // Leaf node: subdivide
            let old_com = self.nodes[node_idx as usize].center_of_mass;
            let old_mass = self.nodes[node_idx as usize].total_mass;
            self.nodes[node_idx as usize].particle_index = NONE;

            // Re-insert existing particle into child
            let old_oct = self.nodes[node_idx as usize].bounds.octant(old_com[0], old_com[1], old_com[2]);
            let child_bounds = self.nodes[node_idx as usize].bounds.child_bounds(old_oct);
            let child_idx = self.alloc(child_bounds);
            self.nodes[node_idx as usize].children[old_oct] = child_idx;
            self.insert(child_idx, existing, old_com[0], old_com[1], old_com[2], old_mass, depth + 1);

            // Insert new particle
            let new_oct = self.nodes[node_idx as usize].bounds.octant(px, py, pz);
            if self.nodes[node_idx as usize].children[new_oct] == NONE {
                let child_bounds = self.nodes[node_idx as usize].bounds.child_bounds(new_oct);
                let child_idx = self.alloc(child_bounds);
                self.nodes[node_idx as usize].children[new_oct] = child_idx;
            }
            let child_idx = self.nodes[node_idx as usize].children[new_oct];
            Self::update_com(&mut self.nodes[node_idx as usize], px, py, pz, mass);
            self.insert(child_idx, particle_idx, px, py, pz, mass, depth + 1);
        } else {
            // Internal node: insert into appropriate child
            Self::update_com(&mut self.nodes[node_idx as usize], px, py, pz, mass);

            let oct = self.nodes[node_idx as usize].bounds.octant(px, py, pz);
            if self.nodes[node_idx as usize].children[oct] == NONE {
                let child_bounds = self.nodes[node_idx as usize].bounds.child_bounds(oct);
                let child_idx = self.alloc(child_bounds);
                self.nodes[node_idx as usize].children[oct] = child_idx;
            }
            let child_idx = self.nodes[node_idx as usize].children[oct];
            self.insert(child_idx, particle_idx, px, py, pz, mass, depth + 1);
        }
    }

    /// Update center of mass when adding a new particle.
    #[inline]
    fn update_com(node: &mut OctreeNode, px: f64, py: f64, pz: f64, mass: f64) {
        let new_mass = node.total_mass + mass;
        if new_mass > 0.0 {
            let inv_new = 1.0 / new_mass;
            node.center_of_mass[0] =
                (node.total_mass * node.center_of_mass[0] + mass * px) * inv_new;
            node.center_of_mass[1] =
                (node.total_mass * node.center_of_mass[1] + mass * py) * inv_new;
            node.center_of_mass[2] =
                (node.total_mass * node.center_of_mass[2] + mass * pz) * inv_new;
        }
        node.total_mass = new_mass;
    }
}

// Octree is Send + Sync since it contains only Vec<OctreeNode> with primitive fields.
const _: () = {
    const fn _assert_send_sync<T: Send + Sync>() {}
    let _ = _assert_send_sync::<Octree>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particles;

    #[test]
    fn single_particle_tree() {
        let mut p = Particles::new(1);
        p.add(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 5.0);

        let tree = Octree::build(&p);
        let root = tree.root();

        assert!(root.is_leaf());
        assert_eq!(root.particle_index, 0);
        assert!((root.total_mass - 5.0).abs() < 1e-14);
        assert!((root.center_of_mass[0] - 1.0).abs() < 1e-14);
        assert!((root.center_of_mass[1] - 2.0).abs() < 1e-14);
        assert!((root.center_of_mass[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn two_particles_subdivide() {
        let mut p = Particles::new(2);
        p.add(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        p.add(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let tree = Octree::build(&p);
        let root = tree.root();

        // Should be an internal node (not a leaf)
        assert!(!root.is_leaf());
        assert!(root.children.iter().any(|&c| c != NONE));

        // Total mass should be 2.0
        assert!((root.total_mass - 2.0).abs() < 1e-14);

        // Center of mass should be at origin (midpoint of equal masses)
        assert!((root.center_of_mass[0]).abs() < 1e-14);
        assert!((root.center_of_mass[1]).abs() < 1e-14);
        assert!((root.center_of_mass[2]).abs() < 1e-14);
    }

    #[test]
    fn bounding_box_contains_all() {
        let mut p = Particles::new(4);
        p.add(-5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        p.add(2.0, -4.0, 7.0, 0.0, 0.0, 0.0, 1.0);
        p.add(0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 1.0);
        p.add(4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let bb = BoundingBox::containing(&p);

        for i in 0..p.count {
            let pos = [p.x[i], p.y[i], p.z[i]];
            for d in 0..3 {
                let lo = bb.center[d] - bb.half_width;
                let hi = bb.center[d] + bb.half_width;
                assert!(
                    pos[d] >= lo && pos[d] <= hi,
                    "Particle {i} dim {d}: pos={} not in [{lo}, {hi}]",
                    pos[d]
                );
            }
        }
    }

    #[test]
    fn octant_indexing() {
        let bb = BoundingBox {
            center: [0.0, 0.0, 0.0],
            half_width: 1.0,
        };

        assert_eq!(bb.octant(-0.5, -0.5, -0.5), 0);
        assert_eq!(bb.octant(0.5, -0.5, -0.5), 1);
        assert_eq!(bb.octant(-0.5, 0.5, -0.5), 2);
        assert_eq!(bb.octant(0.5, 0.5, -0.5), 3);
        assert_eq!(bb.octant(-0.5, -0.5, 0.5), 4);
        assert_eq!(bb.octant(0.5, 0.5, 0.5), 7);
    }

    #[test]
    fn degenerate_same_position() {
        let mut p = Particles::new(2);
        p.add(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        p.add(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0);

        let tree = Octree::build(&p);

        assert!((tree.root().total_mass - 3.0).abs() < 1e-14);
        assert!((tree.root().center_of_mass[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn weighted_center_of_mass() {
        let mut p = Particles::new(2);
        p.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        p.add(4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0);

        let tree = Octree::build(&p);

        assert!((tree.root().center_of_mass[0] - 3.0).abs() < 1e-14);
        assert!((tree.root().total_mass - 4.0).abs() < 1e-14);
    }

    #[test]
    fn many_particles_tree() {
        let n = 100;
        let mut p = Particles::new(n);
        for i in 0..n {
            let x = (i % 10) as f64;
            let y = ((i / 10) % 10) as f64;
            let z = 0.0;
            p.add(x, y, z, 0.0, 0.0, 0.0, 1.0);
        }

        let tree = Octree::build(&p);

        assert!((tree.root().total_mass - n as f64).abs() < 1e-14);
        assert!(!tree.root().is_leaf());
    }
}
