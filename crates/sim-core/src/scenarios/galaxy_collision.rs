use crate::particle::{ParticleType, Particles};
use crate::scenario::Scenario;
use crate::units::G;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Two-galaxy collision scenario.
///
/// Each galaxy consists of three components:
/// - **Disk**: exponential radial profile with sech² vertical structure (stars)
/// - **Bulge**: Plummer sphere (stars)
/// - **Halo**: Hernquist dark matter profile (collisionless)
///
/// The two galaxies are placed on a hyperbolic approach trajectory.
///
/// All quantities are in N-body units (G=1). The natural mapping to
/// physical units is: 1 length unit = 1 kpc, 1 mass unit = 10^10 M_sun,
/// 1 time unit ≈ 14.9 Myr.
pub struct GalaxyCollision {
    /// Total particles per galaxy (split among disk, bulge, halo by mass fraction)
    pub n_per_galaxy: usize,
    /// Fraction of baryonic mass in the disk (remainder goes to bulge)
    pub disk_fraction: f64,
    /// Halo mass as a multiple of baryonic mass
    pub halo_mass_ratio: f64,
    /// Total baryonic mass per galaxy (disk + bulge)
    pub baryon_mass: f64,
    /// Exponential disk scale radius
    pub disk_scale_radius: f64,
    /// Disk vertical scale height (sech² profile)
    pub disk_scale_height: f64,
    /// Plummer bulge scale radius
    pub bulge_scale_radius: f64,
    /// Hernquist halo scale radius
    pub halo_scale_radius: f64,
    /// Maximum disk radius (truncation)
    pub disk_max_radius: f64,
    /// Maximum halo radius (truncation)
    pub halo_max_radius: f64,
    /// Initial separation between galaxy centers
    pub separation: f64,
    /// Perpendicular offset (impact parameter)
    pub impact_parameter: f64,
    /// Relative approach velocity at initial separation
    pub approach_velocity: f64,
    /// RNG seed for deterministic generation
    pub seed: u64,
}

impl Default for GalaxyCollision {
    fn default() -> Self {
        Self {
            n_per_galaxy: 50_000,
            disk_fraction: 0.7,
            halo_mass_ratio: 10.0,
            baryon_mass: 1.0,
            disk_scale_radius: 1.0,
            disk_scale_height: 0.1,
            bulge_scale_radius: 0.3,
            halo_scale_radius: 5.0,
            disk_max_radius: 6.0,  // 6 scale radii
            halo_max_radius: 30.0, // 6 halo scale radii
            separation: 15.0,
            impact_parameter: 2.0,
            approach_velocity: 0.5,
            seed: 42,
        }
    }
}

impl Scenario for GalaxyCollision {
    fn name(&self) -> &str {
        "Galaxy Collision"
    }

    fn generate(&self) -> Particles {
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);

        // Mass budget per galaxy
        let m_disk = self.baryon_mass * self.disk_fraction;
        let m_bulge = self.baryon_mass * (1.0 - self.disk_fraction);
        let m_halo = self.baryon_mass * self.halo_mass_ratio;
        let m_total = m_disk + m_bulge + m_halo;

        // Particle counts proportional to mass
        let n_disk = ((self.n_per_galaxy as f64) * m_disk / m_total).round() as usize;
        let n_bulge = ((self.n_per_galaxy as f64) * m_bulge / m_total).round() as usize;
        let n_halo = self.n_per_galaxy - n_disk - n_bulge;

        // Pre-compute disk enclosed mass lookup table for V_c calculation
        let disk_mass_table = DiskMassTable::new(m_disk, self.disk_scale_radius, self.disk_max_radius);

        let total_particles = self.n_per_galaxy * 2;
        let mut particles = Particles::new(total_particles);

        // Generate galaxy 1 at origin
        let g1_start = particles.count;
        generate_galaxy(
            &mut particles,
            &mut rng,
            n_disk,
            n_bulge,
            n_halo,
            m_disk,
            m_bulge,
            m_halo,
            self.disk_scale_radius,
            self.disk_scale_height,
            self.disk_max_radius,
            self.bulge_scale_radius,
            self.halo_scale_radius,
            self.halo_max_radius,
            &disk_mass_table,
        );
        let g1_end = particles.count;

        // Generate galaxy 2 at origin (will offset after)
        let g2_start = particles.count;
        generate_galaxy(
            &mut particles,
            &mut rng,
            n_disk,
            n_bulge,
            n_halo,
            m_disk,
            m_bulge,
            m_halo,
            self.disk_scale_radius,
            self.disk_scale_height,
            self.disk_max_radius,
            self.bulge_scale_radius,
            self.halo_scale_radius,
            self.halo_max_radius,
            &disk_mass_table,
        );
        let g2_end = particles.count;

        // Place galaxies on collision trajectory
        let half_sep = self.separation / 2.0;
        let half_impact = self.impact_parameter / 2.0;
        let half_vel = self.approach_velocity / 2.0;

        // Galaxy 1: offset to (-half_sep, -half_impact, 0), moving in +x
        for i in g1_start..g1_end {
            particles.x[i] -= half_sep;
            particles.y[i] -= half_impact;
            particles.vx[i] += half_vel;
        }

        // Galaxy 2: offset to (+half_sep, +half_impact, 0), moving in -x
        for i in g2_start..g2_end {
            particles.x[i] += half_sep;
            particles.y[i] += half_impact;
            particles.vx[i] -= half_vel;
        }

        // Shift to center-of-mass frame
        particles.shift_to_com_frame();

        particles
    }

    fn suggested_dt(&self) -> f64 {
        // Orbital period at disk scale radius:
        // T = 2π * R / V_c(R)
        // V_c ≈ sqrt(GM_enclosed / R) at R_d
        // For a rough estimate, use enclosed mass ≈ 0.26 * M_disk (1 scale radius)
        // plus bulge and halo contributions
        let m_enclosed_disk = 0.26 * self.baryon_mass * self.disk_fraction;
        let r = self.disk_scale_radius;
        let a_b = self.bulge_scale_radius;
        let m_bulge = self.baryon_mass * (1.0 - self.disk_fraction);
        let m_enclosed_bulge = m_bulge * r.powi(3) / (r * r + a_b * a_b).powf(1.5);
        let a_h = self.halo_scale_radius;
        let m_halo = self.baryon_mass * self.halo_mass_ratio;
        let x = r / a_h;
        let m_enclosed_halo = m_halo * x * x / ((1.0 + x) * (1.0 + x));
        let m_enc = m_enclosed_disk + m_enclosed_bulge + m_enclosed_halo;
        let v_c = (G * m_enc / r).sqrt();
        let t_orb = 2.0 * std::f64::consts::PI * r / v_c;
        // ~500 steps per inner orbital period
        t_orb / 500.0
    }

    fn suggested_softening(&self) -> f64 {
        // Softening ≈ 5% of disk scale radius
        self.disk_scale_radius * 0.05
    }
}

// ---------------------------------------------------------------------------
// Disk enclosed mass lookup table
// ---------------------------------------------------------------------------

/// Pre-computed enclosed mass M_disk(<R) for the exponential disk.
///
/// The exponential disk has surface density Σ(R) = Σ_0 exp(-R/R_d).
/// The enclosed mass is M(<R) = 2π ∫₀ᴿ Σ(R') R' dR'
///   = M_total * [1 - (1 + R/R_d) * exp(-R/R_d)]
///
/// This has a closed form! But we tabulate it for fast interpolation
/// alongside the other components in the circular velocity calculation.
struct DiskMassTable {
    radii: Vec<f64>,
    enclosed: Vec<f64>,
}

impl DiskMassTable {
    fn new(m_disk: f64, scale_radius: f64, max_radius: f64) -> Self {
        let n_bins = 200;
        let mut radii = Vec::with_capacity(n_bins);
        let mut enclosed = Vec::with_capacity(n_bins);
        let r_d = scale_radius;

        for i in 0..n_bins {
            let r = max_radius * (i as f64 + 0.5) / n_bins as f64;
            let x = r / r_d;
            // M(<R) = M_total * [1 - (1 + x) * exp(-x)]
            let m = m_disk * (1.0 - (1.0 + x) * (-x).exp());
            radii.push(r);
            enclosed.push(m);
        }

        Self { radii, enclosed }
    }

    fn enclosed_mass(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        if r >= *self.radii.last().unwrap() {
            return *self.enclosed.last().unwrap();
        }
        // Linear interpolation
        let dr = self.radii[1] - self.radii[0];
        let idx = ((r - self.radii[0]) / dr).max(0.0) as usize;
        let idx = idx.min(self.radii.len() - 2);
        let frac = (r - self.radii[idx]) / dr;
        self.enclosed[idx] + frac * (self.enclosed[idx + 1] - self.enclosed[idx])
    }
}

// ---------------------------------------------------------------------------
// Galaxy generation
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn generate_galaxy(
    particles: &mut Particles,
    rng: &mut ChaCha20Rng,
    n_disk: usize,
    n_bulge: usize,
    n_halo: usize,
    m_disk: f64,
    m_bulge: f64,
    m_halo: f64,
    disk_scale_radius: f64,
    disk_scale_height: f64,
    disk_max_radius: f64,
    bulge_scale_radius: f64,
    halo_scale_radius: f64,
    halo_max_radius: f64,
    disk_mass_table: &DiskMassTable,
) {
    generate_disk(
        particles,
        rng,
        n_disk,
        m_disk,
        m_bulge,
        m_halo,
        disk_scale_radius,
        disk_scale_height,
        disk_max_radius,
        bulge_scale_radius,
        halo_scale_radius,
        disk_mass_table,
    );
    generate_bulge(particles, rng, n_bulge, m_bulge, bulge_scale_radius);
    generate_halo(
        particles,
        rng,
        n_halo,
        m_bulge,
        m_halo,
        halo_scale_radius,
        halo_max_radius,
        bulge_scale_radius,
        disk_mass_table,
    );
}

/// Compute circular velocity from the combined potential at radius R in the disk plane.
fn circular_velocity(
    r: f64,
    m_bulge: f64,
    bulge_a: f64,
    m_halo: f64,
    halo_a: f64,
    disk_mass_table: &DiskMassTable,
) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }

    // Disk contribution: V_c² = G * M_disk(<R) / R
    let v2_disk = G * disk_mass_table.enclosed_mass(r) / r;

    // Plummer bulge: V_c² = G * M_b * R² / (R² + a²)^(3/2)
    let v2_bulge = G * m_bulge * r * r / (r * r + bulge_a * bulge_a).powf(1.5);

    // Hernquist halo: M(<r) = M_h * r² / (r + a)²
    // V_c² = G * M(<r) / r = G * M_h * r / (r + a)²
    let v2_halo = G * m_halo * r / ((r + halo_a) * (r + halo_a));

    (v2_disk + v2_bulge + v2_halo).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn generate_disk(
    particles: &mut Particles,
    rng: &mut ChaCha20Rng,
    n_disk: usize,
    m_disk: f64,
    m_bulge: f64,
    m_halo: f64,
    scale_radius: f64,
    scale_height: f64,
    max_radius: f64,
    bulge_scale_radius: f64,
    halo_scale_radius: f64,
    disk_mass_table: &DiskMassTable,
) {
    let m_per_particle = m_disk / n_disk as f64;
    let r_d = scale_radius;

    // CDF of exponential disk: F(R) = 1 - (1 + R/R_d) * exp(-R/R_d)
    // No simple inversion, use rejection sampling with envelope
    let max_x = max_radius / r_d;

    for _ in 0..n_disk {
        // Sample radial position from exponential disk via rejection sampling
        let r = loop {
            // Sample from exponential envelope: f(x) = x * exp(-x)
            // CDF: F(x) = 1 - (1+x)*exp(-x), which we already know
            // Use the CDF inversion of the normalized surface density
            let u: f64 = rng.random();
            // We want R from Σ(R) ∝ R * exp(-R/R_d) (include the R from area element)
            // CDF is F(R) = 1 - (1 + R/R_d) * exp(-R/R_d)
            // Can't invert analytically — use rejection against uniform in [0, max_R]
            // with f_max = (1/R_d) * exp(-1) at R = R_d
            let r_try = u * max_radius;
            let x = r_try / r_d;
            let f = x * (-x).exp(); // proportional to R * Σ(R)
            let f_max = (-1.0_f64).exp(); // max of x*exp(-x) at x=1
            let accept: f64 = rng.random();
            if accept < f / f_max && x < max_x {
                break r_try;
            }
        };

        // Azimuthal position: uniform
        let phi: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);
        let x = r * phi.cos();
        let y = r * phi.sin();

        // Vertical position: sech²(z/z_0) distribution
        // CDF inversion: z = z_0 * atanh(2U - 1)
        let u: f64 = rng.random_range(0.01..0.99); // avoid infinity at edges
        let z = scale_height * (2.0 * u - 1.0).atanh();

        // Circular velocity from combined potential
        let v_c = circular_velocity(
            r,
            m_bulge,
            bulge_scale_radius,
            m_halo,
            halo_scale_radius,
            disk_mass_table,
        );

        // Velocity: circular + small radial dispersion for stability
        // Radial dispersion σ_R ≈ 0.15 * V_c (Toomre stability)
        let sigma_r = 0.15 * v_c;
        let v_r: f64 = rng.random_range(-1.0..1.0) * sigma_r;
        let v_phi = v_c + rng.random_range(-1.0..1.0) * sigma_r * 0.5;

        // Vertical dispersion
        let sigma_z = 0.1 * v_c;
        let v_z: f64 = rng.random_range(-1.0..1.0) * sigma_z;

        // Convert cylindrical velocity to Cartesian
        let vx = v_r * phi.cos() - v_phi * phi.sin();
        let vy = v_r * phi.sin() + v_phi * phi.cos();

        particles.add_typed(x, y, z, vx, vy, v_z, m_per_particle, ParticleType::DiskStar as u8);
    }
}

fn generate_bulge(
    particles: &mut Particles,
    rng: &mut ChaCha20Rng,
    n_bulge: usize,
    m_bulge: f64,
    bulge_scale_radius: f64,
) {
    let m_per_particle = m_bulge / n_bulge as f64;
    let a = bulge_scale_radius;

    for _ in 0..n_bulge {
        // Plummer sphere sampling (same as PlummerSphere scenario)
        let x_mass: f64 = rng.random_range(0.001..1.0);
        let r = a / (x_mass.powf(-2.0 / 3.0) - 1.0).sqrt();

        let cos_theta: f64 = rng.random_range(-1.0..1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);

        let px = r * sin_theta * phi.cos();
        let py = r * sin_theta * phi.sin();
        let pz = r * cos_theta;

        // Velocity from Plummer DF: v_esc = sqrt(2GM / sqrt(r² + a²))
        let v_esc = (2.0 * G * m_bulge / (r * r + a * a).sqrt()).sqrt();
        let q = loop {
            let q: f64 = rng.random_range(0.0..1.0);
            let g = q * q * (1.0 - q * q).powf(3.5);
            let g_max = 0.1;
            let u: f64 = rng.random();
            if u < g / g_max {
                break q;
            }
        };
        let v = q * v_esc;

        let cos_theta_v: f64 = rng.random_range(-1.0..1.0);
        let sin_theta_v = (1.0 - cos_theta_v * cos_theta_v).sqrt();
        let phi_v: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);

        let vx = v * sin_theta_v * phi_v.cos();
        let vy = v * sin_theta_v * phi_v.sin();
        let vz = v * cos_theta_v;

        particles.add_typed(px, py, pz, vx, vy, vz, m_per_particle, ParticleType::BulgeStar as u8);
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_halo(
    particles: &mut Particles,
    rng: &mut ChaCha20Rng,
    n_halo: usize,
    m_bulge: f64,
    m_halo: f64,
    halo_scale_radius: f64,
    halo_max_radius: f64,
    bulge_scale_radius: f64,
    disk_mass_table: &DiskMassTable,
) {
    let m_per_particle = m_halo / n_halo as f64;
    let a = halo_scale_radius;

    // Hernquist CDF: M(<r)/M = r²/(r+a)²
    // Inversion: r = a * √U / (1 - √U)  where U = M(<r)/M
    // Truncate at halo_max_radius
    let u_max = {
        let x = halo_max_radius / (halo_max_radius + a);
        x * x // = r²/(r+a)² evaluated at max_radius
    };

    for _ in 0..n_halo {
        // Sample radius from Hernquist profile
        let u: f64 = rng.random_range(0.001..u_max);
        let sqrt_u = u.sqrt();
        let r = a * sqrt_u / (1.0 - sqrt_u);

        // Uniform direction on sphere
        let cos_theta: f64 = rng.random_range(-1.0..1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);

        let px = r * sin_theta * phi.cos();
        let py = r * sin_theta * phi.sin();
        let pz = r * cos_theta;

        // Velocity: isotropic Gaussian with σ = V_c(r) / √2
        // This is the virial equilibrium approximation
        let v_c = circular_velocity(
            r,
            m_bulge,
            bulge_scale_radius,
            m_halo,
            a,
            disk_mass_table,
        );
        let sigma = v_c / 2.0_f64.sqrt();

        // Sample 3 independent Gaussian components via Box-Muller
        let (g1, g2) = box_muller(rng);
        let (g3, _) = box_muller(rng);

        let vx = sigma * g1;
        let vy = sigma * g2;
        let vz = sigma * g3;

        particles.add_typed(
            px,
            py,
            pz,
            vx,
            vy,
            vz,
            m_per_particle,
            ParticleType::DarkMatter as u8,
        );
    }
}

/// Box-Muller transform: generate two independent standard normal samples.
fn box_muller(rng: &mut ChaCha20Rng) -> (f64, f64) {
    let u1: f64 = rng.random_range(0.001..1.0);
    let u2: f64 = rng.random_range(0.0..2.0 * std::f64::consts::PI);
    let r = (-2.0 * u1.ln()).sqrt();
    (r * u2.cos(), r * u2.sin())
}
