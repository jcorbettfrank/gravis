/// N-body unit system: G = 1, M_total = 1, R_virial = 1.
///
/// In these units, the dynamical time t_dyn = (R³ / GM)^(1/2) = 1,
/// meaning one time unit is approximately one crossing time.
///
/// For galactic simulations, a common mapping is:
///   Mass unit = 10^10 M_sun
///   Length unit = 1 kpc
///   Time unit ≈ 14.9 Myr (derived from G=1 constraint)
///   Velocity unit ≈ 65.6 km/s
pub const G: f64 = 1.0;

/// Default Plummer softening length. Prevents singularity at r=0.
/// Should be a small fraction of the system's characteristic size.
/// For a Plummer sphere with scale radius a=1, epsilon ~ 0.01-0.05 is typical.
pub const DEFAULT_SOFTENING: f64 = 0.05;
