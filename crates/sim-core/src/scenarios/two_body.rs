use crate::particle::Particles;
use crate::scenario::Scenario;
use crate::units::G;

/// Two-body Kepler orbit for verification.
///
/// Places two masses in an elliptical orbit with specified eccentricity.
/// Uses the center-of-mass frame so total momentum is exactly zero and
/// COM is at the origin.
///
/// For G=1, m1+m2=1, semi-major axis a=1:
///   Period T = 2π (Kepler's third law: T² = 4π²a³/(G*M))
///
/// The analytical solution is known exactly, making this the primary
/// test for integrator correctness: energy conservation, angular momentum
/// conservation, orbital period accuracy, and absence of apsidal precession.
pub struct TwoBody {
    /// Mass of first body (fraction of total mass)
    pub mass_ratio: f64,
    /// Orbital eccentricity (0 = circular, <1 = elliptical)
    pub eccentricity: f64,
    /// Semi-major axis
    pub semi_major_axis: f64,
}

impl Default for TwoBody {
    fn default() -> Self {
        Self {
            mass_ratio: 0.5,
            eccentricity: 0.5,
            semi_major_axis: 1.0,
        }
    }
}

impl TwoBody {
    /// Orbital period from Kepler's third law.
    pub fn period(&self) -> f64 {
        let m_total = 1.0; // by convention in N-body units
        2.0 * std::f64::consts::PI * (self.semi_major_axis.powi(3) / (G * m_total)).sqrt()
    }
}

impl Scenario for TwoBody {
    fn name(&self) -> &str {
        "Two-Body Kepler Orbit"
    }

    fn generate(&self) -> Particles {
        assert!(
            self.eccentricity >= 0.0 && self.eccentricity < 1.0,
            "Eccentricity must be in [0, 1) for bound orbits, got {}",
            self.eccentricity
        );
        assert!(
            self.semi_major_axis > 0.0,
            "Semi-major axis must be positive, got {}",
            self.semi_major_axis
        );
        assert!(
            self.mass_ratio > 0.0 && self.mass_ratio < 1.0,
            "Mass ratio must be in (0, 1), got {}",
            self.mass_ratio
        );

        let m_total = 1.0;
        let m1 = m_total * self.mass_ratio;
        let m2 = m_total * (1.0 - self.mass_ratio);
        let a = self.semi_major_axis;
        let e = self.eccentricity;

        // Start at periapsis (closest approach).
        // Periapsis distance: r_p = a * (1 - e)
        let r_peri = a * (1.0 - e);

        // Velocity at periapsis from the vis-viva equation:
        //   v² = G*M * (2/r - 1/a)
        let v_peri = (G * m_total * (2.0 / r_peri - 1.0 / a)).sqrt();

        // Place bodies in COM frame along x-axis, velocity along y-axis.
        // Body 1 at -m2/M * r_peri, body 2 at +m1/M * r_peri
        let x1 = -m2 / m_total * r_peri;
        let x2 = m1 / m_total * r_peri;

        // Velocities: v1 = -m2/M * v_peri, v2 = +m1/M * v_peri (COM frame)
        let vy1 = -m2 / m_total * v_peri;
        let vy2 = m1 / m_total * v_peri;

        let mut particles = Particles::new(2);
        particles.add(x1, 0.0, 0.0, 0.0, vy1, 0.0, m1);
        particles.add(x2, 0.0, 0.0, 0.0, vy2, 0.0, m2);
        particles
    }

    fn suggested_dt(&self) -> f64 {
        // ~10,000 steps per orbit for good accuracy
        self.period() / 10_000.0
    }

    fn suggested_softening(&self) -> f64 {
        // Very small softening — we want accurate inverse-square law
        // for the verification test. The periapsis distance is a*(1-e),
        // so softening should be much smaller than that.
        self.semi_major_axis * (1.0 - self.eccentricity) * 0.001
    }
}
