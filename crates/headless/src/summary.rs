//! The machine-readable run summary.
//!
//! Component tracking (centers, sizes, encounters) lives in
//! `sim_core::tracking`, shared with the browser's preview check.

use serde::Serialize;
use sim_core::diagnostics::Diagnostics;
use sim_core::particle::{ParticleType, Particles};
use sim_core::spec::{Component, ScenarioSpec};
pub use sim_core::tracking::{dist, from_spec, whole, ComponentState, Encounter, EncounterLog, Tracked};


/// Particle type id of a lone massive body (black hole, central star), for
/// scenarios that tag one. Workstream B adds it to `ParticleType` as
/// `PointMass = 5`; until then no particle carries it.
pub const POINT_MASS_TYPE: u8 = 5;

/// Indices of point masses (drawn as markers): the spec's `point_mass`
/// components, plus any particle typed as a point mass.
pub fn point_masses(particles: &Particles, spec: Option<&ScenarioSpec>) -> Vec<usize> {
    let mut out: Vec<usize> = (0..particles.count)
        .filter(|&i| particles.particle_type[i] == POINT_MASS_TYPE)
        .collect();
    let mut start = 0;
    for c in spec.map(|s| s.components.as_slice()).unwrap_or_default() {
        if matches!(c, Component::PointMass(_)) && !out.contains(&start) {
            out.push(start);
        }
        start += c.particle_count();
    }
    out.sort_unstable();
    out
}
#[derive(Serialize)]
pub struct Energetics {
    pub kinetic: f64,
    pub potential: f64,
    pub thermal: f64,
    pub total: f64,
    /// (E − E₀) / |E₀|
    pub relative_drift: f64,
    /// 2K / |U|; ≈ 1 in equilibrium
    pub virial_ratio: f64,
}

impl Energetics {
    pub fn new(d: &Diagnostics, e0: f64) -> Self {
        Energetics {
            kinetic: d.kinetic_energy,
            potential: d.potential_energy,
            thermal: d.thermal_energy,
            total: d.total_energy,
            relative_drift: (d.total_energy - e0) / e0.abs(),
            virial_ratio: d.virial_ratio,
        }
    }
}

#[derive(Serialize)]
pub struct Frame {
    pub time: f64,
    pub step: u64,
    /// PNG paths, one per view.
    pub images: Vec<String>,
    /// Linear brightness mapped to white, per view.
    pub exposure: Vec<f32>,
    pub energy: Energetics,
    pub components: Vec<ComponentState>,
}

#[derive(Serialize)]
pub struct RenderInfo {
    pub size: usize,
    /// Half-width of the field of view, centered on the system's center of mass.
    pub radius: f64,
    pub views: Vec<String>,
    pub dark_matter_shown: bool,
    /// "fixed": every frame uses the first frame's exposure, so brightness is
    /// comparable across frames. "per_frame": each frame exposed on its own.
    pub exposure: String,
}

#[derive(Serialize, Default)]
pub struct TypeCounts {
    pub disk_star: usize,
    pub bulge_star: usize,
    pub dark_matter: usize,
    pub gas: usize,
    pub point_mass: usize,
    pub other: usize,
}

impl TypeCounts {
    pub fn of(p: &Particles) -> Self {
        let mut c = TypeCounts::default();
        for &t in &p.particle_type {
            match t {
                t if t == ParticleType::DiskStar as u8 => c.disk_star += 1,
                t if t == ParticleType::BulgeStar as u8 => c.bulge_star += 1,
                t if t == ParticleType::DarkMatter as u8 => c.dark_matter += 1,
                t if t == ParticleType::Gas as u8 => c.gas += 1,
                POINT_MASS_TYPE => c.point_mass += 1,
                _ => c.other += 1,
            }
        }
        c
    }
}

#[derive(Serialize)]
pub struct Summary {
    pub scenario: String,
    pub spec: Option<String>,
    pub particles: usize,
    pub particle_types: TypeCounts,
    pub algorithm: String,
    pub theta: Option<f64>,
    pub dt: f64,
    pub softening: f64,
    pub steps: u64,
    pub sim_time: f64,
    pub wall_seconds: f64,
    pub initial: Energetics,
    pub final_state: Energetics,
    /// Largest |dE/E| among the frames, diagnostic rows, and the end state.
    pub max_abs_energy_drift: f64,
    /// |ΔL| / |L₀| over the run
    pub angular_momentum_drift: f64,
    pub components: Vec<ComponentState>,
    pub encounters: Vec<Encounter>,
    pub render: Option<RenderInfo>,
    pub frames: Vec<Frame>,
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_components_map_to_index_ranges() {
        let spec = ScenarioSpec::from_json(
            r#"{"components": [
                {"type": "plummer", "particles": 200, "position": [-5, 0, 0]},
                {"type": "point_mass", "mass": 0.1, "position": [5, 0, 0]}
            ]}"#,
        )
        .unwrap();
        let built = sim_core::scenario_builder::build_spec(&spec, None, None, None).unwrap();
        let tracked = from_spec(&built.particles, &spec);
        assert_eq!(tracked[0].range, 0..200);
        assert_eq!(tracked[1].range, 200..201);
        assert_eq!(point_masses(&built.particles, Some(&spec)), vec![200]);

        let log = EncounterLog::new(&tracked);
        let e = &log.encounters[0];
        assert!(
            (e.initial_separation - 10.0).abs() < 0.1,
            "separation {}",
            e.initial_separation
        );
    }

}
