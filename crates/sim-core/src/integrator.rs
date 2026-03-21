use crate::gravity::GravitySolver;
use crate::particle::Particles;

/// Time integration method. Advances particles forward by one timestep.
pub trait Integrator {
    /// Advance the system by one timestep dt.
    fn step(&self, particles: &mut Particles, gravity: &dyn GravitySolver, dt: f64);
}

/// Kick-Drift-Kick (KDK) leapfrog integrator.
///
/// This is the velocity Verlet form of the leapfrog method:
///
///   v(t + dt/2) = v(t)     + a(t)     * dt/2    [half-kick]
///   x(t + dt)   = x(t)     + v(t+dt/2) * dt     [drift]
///   a(t + dt)   = F(x(t+dt)) / m                 [recompute forces]
///   v(t + dt)   = v(t+dt/2) + a(t+dt)  * dt/2    [half-kick]
///
/// Key properties:
///
/// 1. **Symplectic**: The map (x,v) → (x',v') preserves phase space volume
///    (Liouville's theorem). This means it exactly conserves a "shadow
///    Hamiltonian" H_shadow = H + O(dt²), so energy errors are bounded
///    and oscillating rather than growing secularly. RK4, despite being
///    4th-order accurate, is not symplectic and shows linear energy drift.
///
/// 2. **Time-reversible**: Running the integrator with -dt exactly undoes
///    the forward step. This is a consequence of the symmetric splitting
///    and further constrains error growth.
///
/// 3. **Second-order**: The global error is O(dt²). For higher accuracy
///    without sacrificing symplecticity, the Yoshida 4th-order method
///    composes three leapfrog steps with specific coefficients.
///
/// 4. **Self-starting**: Unlike multi-step methods, leapfrog only needs
///    the current state. The KDK form (vs DKD) keeps velocities and
///    positions synchronized at integer timesteps, which simplifies
///    diagnostics and snapshot output.
pub struct LeapfrogKDK;

impl Integrator for LeapfrogKDK {
    fn step(&self, p: &mut Particles, gravity: &dyn GravitySolver, dt: f64) {
        let half_dt = 0.5 * dt;
        let n = p.count;

        // Half-kick: v(t + dt/2) = v(t) + a(t) * dt/2
        for i in 0..n {
            p.vx[i] += p.ax[i] * half_dt;
            p.vy[i] += p.ay[i] * half_dt;
            p.vz[i] += p.az[i] * half_dt;
        }

        // Drift: x(t + dt) = x(t) + v(t + dt/2) * dt
        for i in 0..n {
            p.x[i] += p.vx[i] * dt;
            p.y[i] += p.vy[i] * dt;
            p.z[i] += p.vz[i] * dt;
        }

        // Recompute forces at new positions
        p.clear_accelerations();
        gravity.compute_accelerations(p);

        // Half-kick: v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        for i in 0..n {
            p.vx[i] += p.ax[i] * half_dt;
            p.vy[i] += p.ay[i] * half_dt;
            p.vz[i] += p.az[i] * half_dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity::BruteForce;

    #[test]
    fn circular_orbit_one_period() {
        // Two equal masses in circular orbit.
        // With G=1, m1=m2=0.5, separation d=1:
        //   Reduced mass problem: a = d/2 = 0.5 from center
        //   v_circular = sqrt(G * M_total / (4 * a)) = sqrt(1 / 2) for each
        //   Actually for two-body: v = sqrt(G * m_other / (4 * a))
        //
        // Simpler: place m=1 at origin, m_test at r=1.
        // v_circular = sqrt(G*M/r) = 1.0
        // Period T = 2*pi*r/v = 2*pi
        let mut p = Particles::new(2);
        p.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e6); // heavy central mass
        p.add(1.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 1.0); // orbiting particle

        // For m_central >> m_test: v_c = sqrt(G * m_central / r)
        let v_c = (1e6_f64).sqrt(); // = 1000
        p.vy[1] = v_c;

        let gravity = BruteForce::new(0.0);
        let integrator = LeapfrogKDK;

        // Initialize accelerations
        gravity.compute_accelerations(&mut p);

        // Period T = 2*pi*r/v = 2*pi/1000 ≈ 0.006283
        let period = 2.0 * std::f64::consts::PI / v_c;
        let n_steps = 10000;
        let dt = period / n_steps as f64;

        for _ in 0..n_steps {
            integrator.step(&mut p, &gravity, dt);
        }

        // After one full period, particle should return near starting position
        let dx = p.x[1] - 1.0;
        let dy = p.y[1] - 0.0;
        let dist_err = (dx * dx + dy * dy).sqrt();

        // With 10K steps per orbit, leapfrog should give ~1e-6 positional error
        assert!(
            dist_err < 1e-4,
            "Position error after one orbit: {dist_err:.2e}"
        );
    }
}
