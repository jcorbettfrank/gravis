#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::barnes_hut::BarnesHut;
use crate::particle::Particles;
use crate::units::G;

/// Complete diagnostic state of the simulation at a point in time.
#[derive(Debug, Clone)]
pub struct Diagnostics {
    /// Simulation time
    pub time: f64,
    /// Step number
    pub step: u64,
    /// Total kinetic energy: Σ 0.5 * m * v²
    pub kinetic_energy: f64,
    /// Total gravitational potential energy: Σᵢ<ⱼ -G*mᵢ*mⱼ / |rᵢⱼ|
    /// Computed with softening for consistency with the force calculation.
    pub potential_energy: f64,
    /// Total energy: K + U (should be conserved by symplectic integrator)
    pub total_energy: f64,
    /// Total linear momentum vector (should be conserved exactly)
    pub momentum: [f64; 3],
    /// Total angular momentum vector (should be conserved exactly)
    pub angular_momentum: [f64; 3],
    /// Center of mass position (should remain stationary)
    pub center_of_mass: [f64; 3],
    /// Center of mass velocity (should remain zero)
    pub center_of_mass_velocity: [f64; 3],
    /// Virial ratio 2K/|U| (should be ~1 for virial equilibrium)
    pub virial_ratio: f64,
    /// Total thermal energy: Σ mᵢ uᵢ (gas particles only)
    pub thermal_energy: f64,
}

/// How `compute_inner` gets the potential energy.
enum Potential {
    /// Skip it: U, E and the virial ratio come out without U.
    Skip,
    /// Exact O(N²) pair sum.
    Exact,
    /// Barnes-Hut tree estimate, O(N log N), with this opening angle.
    Tree(f64),
}

/// Compute all diagnostics for the current state.
///
/// The potential energy calculation is O(N²) — same cost as brute-force
/// gravity. For large N, `compute_tree` gives an approximate potential in
/// O(N log N); for verification purposes exact is correct.
pub fn compute(particles: &Particles, softening: f64, time: f64, step: u64) -> Diagnostics {
    compute_inner(particles, softening, time, step, Potential::Exact)
}

/// Compute all diagnostics with a Barnes-Hut estimate of the potential energy.
///
/// O(N log N). The potential carries the tree's monopole error, so use a
/// small `theta` (≲0.3) when the result feeds an energy-drift readout, and
/// compare drifts only between values computed the same way.
pub fn compute_tree(
    particles: &Particles,
    softening: f64,
    theta: f64,
    time: f64,
    step: u64,
) -> Diagnostics {
    compute_inner(particles, softening, time, step, Potential::Tree(theta))
}

/// Compute diagnostics without the O(N²) potential energy calculation.
///
/// Returns kinetic energy, momentum, angular momentum, and COM — all O(N).
/// Potential energy, total energy, and virial ratio are set to 0.
/// Use this for large N when you don't need energy conservation tracking.
pub fn compute_fast(particles: &Particles, time: f64, step: u64) -> Diagnostics {
    compute_inner(particles, 0.0, time, step, Potential::Skip)
}

fn compute_inner(
    particles: &Particles,
    softening: f64,
    time: f64,
    step: u64,
    potential: Potential,
) -> Diagnostics {
    let n = particles.count;
    let eps2 = softening * softening;
    let p = particles;

    // Kinetic energy: K = Σ 0.5 * m * (vx² + vy² + vz²)
    let mut kinetic_energy = 0.0;
    for i in 0..n {
        let v2 = p.vx[i] * p.vx[i] + p.vy[i] * p.vy[i] + p.vz[i] * p.vz[i];
        kinetic_energy += 0.5 * p.mass[i] * v2;
    }

    // Potential energy: U = Σᵢ<ⱼ -G * mᵢ * mⱼ / sqrt(rᵢⱼ² + ε²)
    let mut potential_energy = 0.0;
    if let Potential::Tree(theta) = potential {
        potential_energy = BarnesHut::new(softening, theta).potential_energy(p);
    }
    if let Potential::Exact = potential {
        potential_energy = exact_potential(p, eps2);
    }

    // Thermal energy: E_th = Σ mᵢ uᵢ (gas particles only)
    let mut thermal_energy = 0.0;
    for i in 0..n {
        if p.is_gas(i) {
            thermal_energy += p.mass[i] * p.internal_energy[i];
        }
    }

    let total_energy = kinetic_energy + potential_energy + thermal_energy;

    // Linear momentum: p = Σ m * v
    let mut momentum = [0.0; 3];
    for i in 0..n {
        momentum[0] += p.mass[i] * p.vx[i];
        momentum[1] += p.mass[i] * p.vy[i];
        momentum[2] += p.mass[i] * p.vz[i];
    }

    // Angular momentum: L = Σ m * (r × v)
    let mut angular_momentum = [0.0; 3];
    for i in 0..n {
        angular_momentum[0] += p.mass[i] * (p.y[i] * p.vz[i] - p.z[i] * p.vy[i]);
        angular_momentum[1] += p.mass[i] * (p.z[i] * p.vx[i] - p.x[i] * p.vz[i]);
        angular_momentum[2] += p.mass[i] * (p.x[i] * p.vy[i] - p.y[i] * p.vx[i]);
    }

    // Center of mass
    let m_total = p.total_mass();
    let mut center_of_mass = [0.0; 3];
    let mut center_of_mass_velocity = [0.0; 3];
    if m_total > 0.0 {
        for i in 0..n {
            center_of_mass[0] += p.mass[i] * p.x[i];
            center_of_mass[1] += p.mass[i] * p.y[i];
            center_of_mass[2] += p.mass[i] * p.z[i];
            center_of_mass_velocity[0] += p.mass[i] * p.vx[i];
            center_of_mass_velocity[1] += p.mass[i] * p.vy[i];
            center_of_mass_velocity[2] += p.mass[i] * p.vz[i];
        }
        for c in &mut center_of_mass {
            *c /= m_total;
        }
        for c in &mut center_of_mass_velocity {
            *c /= m_total;
        }
    }

    let virial_ratio = if potential_energy.abs() > 0.0 {
        2.0 * kinetic_energy / potential_energy.abs()
    } else {
        0.0
    };

    Diagnostics {
        time,
        step,
        kinetic_energy,
        potential_energy,
        total_energy,
        momentum,
        angular_momentum,
        center_of_mass,
        center_of_mass_velocity,
        virial_ratio,
        thermal_energy,
    }
}

/// Exact softened pair potential U = Σᵢ<ⱼ -G mᵢ mⱼ / sqrt(rᵢⱼ² + ε²).
///
/// Rows (one i against all j > i) run in parallel with the `parallel`
/// feature; row sums are then added in index order, so the result does not
/// depend on thread count.
fn exact_potential(p: &Particles, eps2: f64) -> f64 {
    let n = p.count;
    let row = |i: usize| {
        let mut u = 0.0;
        for j in (i + 1)..n {
            let dx = p.x[j] - p.x[i];
            let dy = p.y[j] - p.y[i];
            let dz = p.z[j] - p.z[i];
            let r2 = dx * dx + dy * dy + dz * dz + eps2;
            u -= G * p.mass[i] * p.mass[j] / r2.sqrt();
        }
        u
    };

    #[cfg(feature = "parallel")]
    let rows: Vec<f64> = (0..n).into_par_iter().map(row).collect();
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<f64> = (0..n).map(row).collect();

    rows.iter().sum()
}

impl Diagnostics {
    /// Magnitude of the linear momentum vector.
    pub fn momentum_magnitude(&self) -> f64 {
        let [px, py, pz] = self.momentum;
        (px * px + py * py + pz * pz).sqrt()
    }

    /// Magnitude of the angular momentum vector.
    pub fn angular_momentum_magnitude(&self) -> f64 {
        let [lx, ly, lz] = self.angular_momentum;
        (lx * lx + ly * ly + lz * lz).sqrt()
    }

    /// Distance of center of mass from origin.
    pub fn com_drift(&self) -> f64 {
        let [x, y, z] = self.center_of_mass;
        (x * x + y * y + z * z).sqrt()
    }

    /// CSV header for diagnostic output.
    pub fn csv_header() -> &'static str {
        "step,time,kinetic_energy,potential_energy,thermal_energy,total_energy,momentum_mag,angular_momentum_mag,com_drift,virial_ratio"
    }

    /// Format as a CSV row.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{:.6e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.6}",
            self.step,
            self.time,
            self.kinetic_energy,
            self.potential_energy,
            self.thermal_energy,
            self.total_energy,
            self.momentum_magnitude(),
            self.angular_momentum_magnitude(),
            self.com_drift(),
            self.virial_ratio,
        )
    }
}
