/// Particle type identifiers for multi-component scenarios.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleType {
    Default = 0,
    DiskStar = 1,
    BulgeStar = 2,
    DarkMatter = 3,
}

/// Structure-of-Arrays particle storage for cache-friendly iteration.
///
/// All simulation quantities are f64 for numerical accuracy in long-run
/// integrations. f32 is used only when packing data for GPU rendering.
///
/// The SoA layout ensures that when iterating over all positions (the hot
/// path in force calculation), the data is contiguous in memory. On the
/// M5 Pro with 64-byte cache lines, each line holds 8 f64 values, so
/// sequential position reads get full cache utilization.
#[derive(Debug, Clone)]
pub struct Particles {
    /// Number of active particles
    pub count: usize,

    // Positions (N-body length units)
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,

    // Velocities (N-body velocity units)
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub vz: Vec<f64>,

    // Accelerations (computed each timestep, N-body acceleration units)
    pub ax: Vec<f64>,
    pub ay: Vec<f64>,
    pub az: Vec<f64>,

    // Mass (N-body mass units)
    pub mass: Vec<f64>,

    // Particle type (not used in force calculations, used for rendering color)
    pub particle_type: Vec<u8>,
}

impl Particles {
    /// Create a particle system with pre-allocated capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            count: 0,
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            vx: Vec::with_capacity(capacity),
            vy: Vec::with_capacity(capacity),
            vz: Vec::with_capacity(capacity),
            ax: Vec::with_capacity(capacity),
            ay: Vec::with_capacity(capacity),
            az: Vec::with_capacity(capacity),
            mass: Vec::with_capacity(capacity),
            particle_type: Vec::with_capacity(capacity),
        }
    }

    /// Add a single particle with default type. Returns its index.
    #[allow(clippy::too_many_arguments)]
    pub fn add(&mut self, x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64) -> usize {
        self.add_typed(x, y, z, vx, vy, vz, mass, ParticleType::Default as u8)
    }

    /// Add a single particle with explicit type. Returns its index.
    #[allow(clippy::too_many_arguments)]
    pub fn add_typed(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        vx: f64,
        vy: f64,
        vz: f64,
        mass: f64,
        particle_type: u8,
    ) -> usize {
        let idx = self.count;
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
        self.vx.push(vx);
        self.vy.push(vy);
        self.vz.push(vz);
        self.ax.push(0.0);
        self.ay.push(0.0);
        self.az.push(0.0);
        self.mass.push(mass);
        self.particle_type.push(particle_type);
        self.count += 1;
        idx
    }

    /// Total mass of the system.
    pub fn total_mass(&self) -> f64 {
        self.mass.iter().sum()
    }

    /// Zero all accelerations. Called before force accumulation.
    pub fn clear_accelerations(&mut self) {
        for i in 0..self.count {
            self.ax[i] = 0.0;
            self.ay[i] = 0.0;
            self.az[i] = 0.0;
        }
    }

    /// Shift particles so center of mass is at origin with zero bulk velocity.
    pub fn shift_to_com_frame(&mut self) {
        let m_total = self.total_mass();
        if m_total == 0.0 {
            return;
        }

        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        let mut cvx = 0.0;
        let mut cvy = 0.0;
        let mut cvz = 0.0;

        for i in 0..self.count {
            cx += self.mass[i] * self.x[i];
            cy += self.mass[i] * self.y[i];
            cz += self.mass[i] * self.z[i];
            cvx += self.mass[i] * self.vx[i];
            cvy += self.mass[i] * self.vy[i];
            cvz += self.mass[i] * self.vz[i];
        }

        cx /= m_total;
        cy /= m_total;
        cz /= m_total;
        cvx /= m_total;
        cvy /= m_total;
        cvz /= m_total;

        for i in 0..self.count {
            self.x[i] -= cx;
            self.y[i] -= cy;
            self.z[i] -= cz;
            self.vx[i] -= cvx;
            self.vy[i] -= cvy;
            self.vz[i] -= cvz;
        }
    }
}
