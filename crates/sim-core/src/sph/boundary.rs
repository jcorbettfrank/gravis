//! Simple boundary conditions for SPH.

use crate::particle::Particles;

/// Boundary condition type.
pub enum Boundary {
    /// No boundaries (particles can escape freely).
    None,
    /// Reflective boundaries: particles bounce off walls.
    /// bounds[d] = (min, max) for dimension d (x=0, y=1, z=2).
    Reflective {
        bounds: [(f64, f64); 3],
    },
}

impl Boundary {
    /// Apply boundary conditions after the drift step.
    /// Reflects position and reverses velocity component for particles that
    /// crossed a boundary wall.
    pub fn apply(&self, particles: &mut Particles) {
        match self {
            Boundary::None => {}
            Boundary::Reflective { bounds } => {
                for i in 0..particles.count {
                    let pos = [&mut particles.x, &mut particles.y, &mut particles.z];
                    let vel = [&mut particles.vx, &mut particles.vy, &mut particles.vz];

                    for d in 0..3 {
                        let (lo, hi) = bounds[d];
                        if pos[d][i] < lo {
                            pos[d][i] = 2.0 * lo - pos[d][i];
                            vel[d][i] = -vel[d][i];
                        } else if pos[d][i] > hi {
                            pos[d][i] = 2.0 * hi - pos[d][i];
                            vel[d][i] = -vel[d][i];
                        }
                    }
                }
            }
        }
    }
}
