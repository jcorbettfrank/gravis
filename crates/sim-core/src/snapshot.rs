use crate::particle::Particles;
use std::io::{self, Read, Write};

/// Magic bytes to identify snapshot files.
const MAGIC: &[u8; 8] = b"NBODY001";

/// A complete simulation state that can be saved and restored.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Simulation time at this snapshot
    pub time: f64,
    /// Step number
    pub step: u64,
    /// Softening parameter used
    pub softening: f64,
    /// Timestep used
    pub dt: f64,
    /// Particle state
    pub particles: Particles,
}

impl Snapshot {
    /// Create a snapshot from current simulation state.
    pub fn capture(particles: &Particles, time: f64, step: u64, softening: f64, dt: f64) -> Self {
        Self {
            time,
            step,
            softening,
            dt,
            particles: particles.clone(),
        }
    }

    /// Write snapshot to a binary stream.
    ///
    /// Format: magic (8B) | time (f64) | step (u64) | softening (f64) | dt (f64)
    ///         | count (u64) | x[] | y[] | z[] | vx[] | vy[] | vz[] | mass[]
    ///
    /// All arrays are count × f64 (8 bytes each), little-endian.
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&self.time.to_le_bytes())?;
        w.write_all(&self.step.to_le_bytes())?;
        w.write_all(&self.softening.to_le_bytes())?;
        w.write_all(&self.dt.to_le_bytes())?;
        w.write_all(&(self.particles.count as u64).to_le_bytes())?;

        let p = &self.particles;
        for arr in [&p.x, &p.y, &p.z, &p.vx, &p.vy, &p.vz, &p.mass] {
            for &val in arr.iter() {
                w.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Read snapshot from a binary stream.
    pub fn read_from(r: &mut dyn Read) -> io::Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid snapshot magic bytes",
            ));
        }

        let time = read_f64(r)?;
        let step = read_u64(r)?;
        let softening = read_f64(r)?;
        let dt = read_f64(r)?;
        let count = read_u64(r)? as usize;

        let mut particles = Particles::new(count);
        particles.count = count;

        read_f64_vec(r, &mut particles.x, count)?;
        read_f64_vec(r, &mut particles.y, count)?;
        read_f64_vec(r, &mut particles.z, count)?;
        read_f64_vec(r, &mut particles.vx, count)?;
        read_f64_vec(r, &mut particles.vy, count)?;
        read_f64_vec(r, &mut particles.vz, count)?;
        read_f64_vec(r, &mut particles.mass, count)?;

        // Accelerations are not stored — they MUST be recomputed from
        // positions before stepping. See `Snapshot::initialize()`.
        particles.ax.resize(count, 0.0);
        particles.ay.resize(count, 0.0);
        particles.az.resize(count, 0.0);

        Ok(Self {
            time,
            step,
            softening,
            dt,
            particles,
        })
    }

    /// Recompute accelerations from current positions, making the snapshot
    /// ready for time integration.
    ///
    /// **You must call this before stepping** after loading a snapshot.
    /// The KDK leapfrog's first operation is a half-kick (v += a * dt/2),
    /// so valid accelerations are required. Without this call, the first
    /// half-kick uses zeroed accelerations and produces a wrong trajectory.
    pub fn initialize(&mut self, gravity: &dyn crate::gravity::GravitySolver) {
        self.particles.clear_accelerations();
        gravity.compute_accelerations(&mut self.particles);
    }
}

fn read_f64(r: &mut dyn Read) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_u64(r: &mut dyn Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64_vec(r: &mut dyn Read, vec: &mut Vec<f64>, count: usize) -> io::Result<()> {
    vec.clear();
    vec.reserve(count);
    for _ in 0..count {
        vec.push(read_f64(r)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn round_trip() {
        let mut particles = Particles::new(2);
        particles.add(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.5);
        particles.add(-1.0, -2.0, -3.0, -0.1, -0.2, -0.3, 2.5);

        let snap = Snapshot::capture(&particles, 1.234, 42, 0.05, 0.001);

        let mut buf = Vec::new();
        snap.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let restored = Snapshot::read_from(&mut cursor).unwrap();

        assert_eq!(restored.time, 1.234);
        assert_eq!(restored.step, 42);
        assert_eq!(restored.softening, 0.05);
        assert_eq!(restored.dt, 0.001);
        assert_eq!(restored.particles.count, 2);
        assert_eq!(restored.particles.x[0], 1.0);
        assert_eq!(restored.particles.mass[1], 2.5);

        // Accelerations should be zeroed on load (caller must call initialize)
        assert_eq!(restored.particles.ax[0], 0.0);
        assert_eq!(restored.particles.ax[1], 0.0);
    }

    #[test]
    fn restore_then_initialize_matches_fresh() {
        use crate::gravity::{BruteForce, GravitySolver};
        use crate::integrator::{Integrator, LeapfrogKDK};
        use crate::scenario::Scenario;
        use crate::scenarios::two_body::TwoBody;

        let scenario = TwoBody::default();
        let softening = scenario.suggested_softening();
        let dt = scenario.suggested_dt();
        let gravity = BruteForce::new(softening);
        let integrator = LeapfrogKDK;

        // Run 100 steps, snapshot, run 100 more
        let mut p = scenario.generate();
        p.clear_accelerations();
        gravity.compute_accelerations(&mut p);
        for _ in 0..100 {
            integrator.step(&mut p, &gravity, dt);
        }

        let snap = Snapshot::capture(&p, 100.0 * dt, 100, softening, dt);
        let mut buf = Vec::new();
        snap.write_to(&mut buf).unwrap();

        // Continue original for 100 more steps
        for _ in 0..100 {
            integrator.step(&mut p, &gravity, dt);
        }
        let expected_x = p.x[0];

        // Restore snapshot and run 100 steps
        let mut cursor = Cursor::new(&buf);
        let mut restored = Snapshot::read_from(&mut cursor).unwrap();
        restored.initialize(&gravity);
        for _ in 0..100 {
            integrator.step(&mut restored.particles, &gravity, dt);
        }
        let restored_x = restored.particles.x[0];

        // Should match to machine precision
        assert!(
            (expected_x - restored_x).abs() < 1e-14,
            "Restored trajectory diverged: expected={expected_x}, got={restored_x}"
        );
    }
}
