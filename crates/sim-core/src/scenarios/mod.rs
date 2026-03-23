pub mod cold_collapse;
pub mod evrard_collapse;
pub mod galaxy_collision;
pub mod kelvin_helmholtz;
pub mod plummer_sphere;
pub mod protoplanetary;
pub mod sedov_blast;
pub mod sod_shock;
pub mod two_body;

use rand::Rng;

use crate::particle::Particles;

/// Compute volume, mean inter-particle spacing, and initial smoothing length
/// for `n_particles` uniformly distributed in a sphere of given `radius`.
pub fn sphere_spacing(radius: f64, n_particles: usize) -> (f64, f64, f64) {
    let vol = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
    let mean_spacing = (vol / n_particles as f64).cbrt();
    let h = 1.5 * mean_spacing;
    (vol, mean_spacing, h)
}

/// Place `n` gas particles uniformly inside a sphere via rejection sampling.
pub fn fill_uniform_gas_sphere(
    particles: &mut Particles,
    rng: &mut impl Rng,
    n: usize,
    radius: f64,
    mass_per_particle: f64,
    internal_energy: f64,
    smoothing_length: f64,
) {
    let mut count = 0;
    while count < n {
        let x: f64 = rng.random::<f64>() * 2.0 * radius - radius;
        let y: f64 = rng.random::<f64>() * 2.0 * radius - radius;
        let z: f64 = rng.random::<f64>() * 2.0 * radius - radius;
        if x * x + y * y + z * z <= radius * radius {
            particles.add_gas(x, y, z, 0.0, 0.0, 0.0, mass_per_particle, internal_energy, smoothing_length);
            count += 1;
        }
    }
}
