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
        }
    }

    /// Add a single particle. Returns its index.
    #[allow(clippy::too_many_arguments)]
    pub fn add(&mut self, x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64) -> usize {
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
}
