//! Telescope-like particle styling.
//!
//! What a particle looks like depends on its type, never its mass. Mass is a
//! simulation detail: a galaxy may use heavy dark-matter particles and light
//! star particles, and a black hole outweighs a star cluster, but neither
//! should change how big or bright a star looks on screen.
//!
//! - Stars (`Default`, `DiskStar`, `BulgeStar`) and gas share one size and
//!   brightness, set by how many of them there are, so a 1K and a 40K
//!   rendering of the same system have similar total light.
//! - Dark matter is invisible to a telescope and is hidden unless
//!   [`ParticleStyler::new`] is asked to show it.
//! - Point masses (black holes, central stars) are small, very bright points
//!   that bloom into a glow.

/// Particle type ids, mirroring `sim_core::particle::ParticleType`.
/// (render-core doesn't depend on sim-core.)
pub mod types {
    pub const DEFAULT: u8 = 0;
    pub const DISK_STAR: u8 = 1;
    pub const BULGE_STAR: u8 = 2;
    pub const DARK_MATTER: u8 = 3;
    pub const GAS: u8 = 4;
    pub const POINT_MASS: u8 = 5;
}

/// World-space billboard size of one particle when there are 1,000 of its
/// kind; it shrinks as N^(-1/3), like the mean interparticle spacing.
const BASE_SIZE: f32 = 0.03;
const BASE_COUNT: f32 = 1000.0;

/// HDR brightness of a point mass. Anything above the bloom threshold (0.8)
/// spills into a glow, so this reads as a bright point source.
const POINT_MASS_BRIGHTNESS: f32 = 25.0;
/// Point masses are drawn a little larger than a star, but never as a blob.
const POINT_MASS_SIZE_FACTOR: f32 = 1.5;
const POINT_MASS_MAX_SIZE: f32 = 0.1;

/// How one particle is drawn. `color.rgb` already includes brightness (and
/// may exceed 1 in the HDR target); `size` is the billboard's world-space
/// width. A size of zero means the particle isn't drawn.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParticleStyle {
    pub color: [f32; 4],
    pub size: f32,
}

impl ParticleStyle {
    pub const HIDDEN: Self = Self { color: [0.0; 4], size: 0.0 };

    pub fn is_visible(&self) -> bool {
        self.size > 0.0
    }
}

/// Size and brightness for a population of `count` equal-weight particles.
fn population_scale(count: usize) -> (f32, f32) {
    let ratio = BASE_COUNT / count.max(1) as f32;
    let size = BASE_SIZE * ratio.cbrt();
    let brightness = ratio.sqrt().clamp(0.3, 1.0);
    (size, brightness)
}

fn is_luminous(particle_type: u8) -> bool {
    !matches!(particle_type, types::DARK_MATTER | types::POINT_MASS)
}

/// Styles every particle in a scene. Built once per scenario (it counts the
/// particles of each kind), then queried per particle.
#[derive(Copy, Clone, Debug)]
pub struct ParticleStyler {
    luminous_size: f32,
    luminous_brightness: f32,
    dark_matter_size: f32,
    dark_matter_brightness: f32,
    show_dark_matter: bool,
}

impl ParticleStyler {
    pub fn new(particle_types: &[u8], show_dark_matter: bool) -> Self {
        let luminous = particle_types.iter().filter(|&&t| is_luminous(t)).count();
        let dark = particle_types
            .iter()
            .filter(|&&t| t == types::DARK_MATTER)
            .count();
        let (luminous_size, luminous_brightness) = population_scale(luminous);
        let (dark_matter_size, dark_matter_brightness) = population_scale(dark);
        Self {
            luminous_size,
            luminous_brightness,
            dark_matter_size,
            dark_matter_brightness,
            show_dark_matter,
        }
    }

    pub fn show_dark_matter(&self) -> bool {
        self.show_dark_matter
    }

    pub fn style(&self, particle_type: u8, internal_energy: f32) -> ParticleStyle {
        match particle_type {
            types::DARK_MATTER if !self.show_dark_matter => ParticleStyle::HIDDEN,
            types::DARK_MATTER => ParticleStyle {
                color: scaled(particle_type_to_color(particle_type), self.dark_matter_brightness),
                size: self.dark_matter_size,
            },
            types::POINT_MASS => ParticleStyle {
                color: scaled(particle_type_to_color(particle_type), POINT_MASS_BRIGHTNESS),
                size: (self.luminous_size * POINT_MASS_SIZE_FACTOR).min(POINT_MASS_MAX_SIZE),
            },
            _ => ParticleStyle {
                color: scaled(
                    particle_color(particle_type, internal_energy),
                    self.luminous_brightness,
                ),
                size: self.luminous_size,
            },
        }
    }

    /// Style a whole scene; `internal_energies` only matters for gas.
    pub fn style_all(&self, particle_types: &[u8], internal_energies: &[f32]) -> Vec<ParticleStyle> {
        particle_types
            .iter()
            .zip(internal_energies)
            .map(|(&t, &u)| self.style(t, u))
            .collect()
    }
}

fn scaled(color: [f32; 4], brightness: f32) -> [f32; 4] {
    [color[0] * brightness, color[1] * brightness, color[2] * brightness, color[3]]
}

/// Map particle type to RGBA color for rendering.
///
/// Color scheme:
/// - Default (0): blue-white (backward compatible with original color)
/// - DiskStar (1): blue-white (cool stellar population)
/// - BulgeStar (2): yellow-orange (old stellar population)
/// - DarkMatter (3): faint violet haze (only drawn on request)
/// - Gas (4): use `gas_temperature_color` instead for dynamic coloring
/// - PointMass (5): warm white
pub fn particle_type_to_color(particle_type: u8) -> [f32; 4] {
    match particle_type {
        types::DEFAULT => [0.7, 0.85, 1.0, 1.0],     // blue-white
        types::DISK_STAR => [0.6, 0.75, 1.0, 1.0],   // blue-white
        types::BULGE_STAR => [1.0, 0.8, 0.4, 1.0],   // yellow-orange
        types::DARK_MATTER => [0.45, 0.3, 0.8, 0.4], // faint violet haze
        types::GAS => [0.3, 0.6, 1.0, 0.8],          // overridden by temperature
        types::POINT_MASS => [1.0, 0.95, 0.85, 1.0], // warm white
        _ => [0.7, 0.85, 1.0, 1.0],                  // fallback: blue-white
    }
}

/// Map a particle to its RGBA color, dispatching gas particles to temperature
/// coloring and all others to the static type-based palette.
pub fn particle_color(particle_type: u8, internal_energy: f32) -> [f32; 4] {
    if particle_type == types::GAS {
        gas_temperature_color(internal_energy)
    } else {
        particle_type_to_color(particle_type)
    }
}

/// Map gas internal energy to an RGBA color using a temperature colormap.
///
/// Uses log(u) as a temperature proxy:
/// - Cold (low u): blue
/// - Warm (moderate u): yellow-orange
/// - Hot (high u): white
///
/// The range is calibrated to u ∈ [0.01, 100] covering typical SPH problems.
pub fn gas_temperature_color(internal_energy: f32) -> [f32; 4] {
    let log_u = internal_energy.max(1e-6).log10();
    // Map log10(u) from [-2, 2] to [0, 1]
    let t = ((log_u + 2.0) / 4.0).clamp(0.0, 1.0);

    // Cool (blue) → warm (yellow) → hot (white)
    let r = t.powf(0.5);
    let g = t.powf(1.2);
    let b = (1.0 - t).powf(0.7) * 0.9 + t * 0.5;
    let a = 0.7 + 0.3 * t; // Hotter particles more opaque

    [r, g, b, a]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_ids_match_sim_core() {
        use sim_core::particle::ParticleType;
        assert_eq!(types::DEFAULT, ParticleType::Default as u8);
        assert_eq!(types::DISK_STAR, ParticleType::DiskStar as u8);
        assert_eq!(types::BULGE_STAR, ParticleType::BulgeStar as u8);
        assert_eq!(types::DARK_MATTER, ParticleType::DarkMatter as u8);
        assert_eq!(types::GAS, ParticleType::Gas as u8);
        assert_eq!(types::POINT_MASS, ParticleType::PointMass as u8);
    }

    #[test]
    fn dark_matter_hidden_by_default_and_shown_on_request() {
        let kinds = [types::DISK_STAR, types::DARK_MATTER];
        let hidden = ParticleStyler::new(&kinds, false);
        assert!(!hidden.style(types::DARK_MATTER, 0.0).is_visible());
        assert!(hidden.style(types::DISK_STAR, 0.0).is_visible());
        let shown = ParticleStyler::new(&kinds, true);
        assert!(shown.style(types::DARK_MATTER, 0.0).is_visible());
    }

    #[test]
    fn star_look_ignores_dark_matter_count() {
        // A galaxy with a heavy halo draws its stars exactly like the same
        // stars on their own.
        let stars = vec![types::DISK_STAR; 1000];
        let mut galaxy = stars.clone();
        galaxy.extend(std::iter::repeat_n(types::DARK_MATTER, 9000));
        let a = ParticleStyler::new(&stars, false).style(types::DISK_STAR, 0.0);
        let b = ParticleStyler::new(&galaxy, true).style(types::DISK_STAR, 0.0);
        assert_eq!(a, b);
    }

    #[test]
    fn equal_mass_scenes_match_the_old_mass_based_look() {
        // Before type-based styling, size was 0.03·(1000 m)^(1/3) and
        // brightness clamp((1000 m)^½, 0.3, 1) with m = 1/N.
        for n in [2usize, 1000, 5000, 40_000] {
            let s = ParticleStyler::new(&vec![types::DEFAULT; n], false).style(types::DEFAULT, 0.0);
            let m = 1000.0 / n as f32;
            assert!((s.size - 0.03 * m.cbrt()).abs() < 1e-6);
            assert!((s.color[0] - 0.7 * m.sqrt().clamp(0.3, 1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn point_mass_is_small_and_bright() {
        let mut kinds = vec![types::DEFAULT; 1000];
        kinds.push(types::POINT_MASS);
        let styler = ParticleStyler::new(&kinds, false);
        let star = styler.style(types::DEFAULT, 0.0);
        let hole = styler.style(types::POINT_MASS, 0.0);
        assert!(hole.size <= POINT_MASS_MAX_SIZE && hole.size < 2.0 * star.size);
        assert!(hole.color[0] > 0.8, "point mass should exceed the bloom threshold");

        // Alone, a point mass still isn't a blob.
        let lone = ParticleStyler::new(&[types::POINT_MASS], false).style(types::POINT_MASS, 0.0);
        assert!(lone.size <= POINT_MASS_MAX_SIZE);
    }
}
