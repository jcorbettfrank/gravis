//! Per-component tracking and the machine-readable run summary.
//!
//! A spec's components are generated in order into contiguous index ranges, so
//! each one can be followed through the run: its center, bulk velocity, size,
//! and the closest approach between every pair, which says whether an
//! encounter actually happened.

use serde::Serialize;
use sim_core::diagnostics::Diagnostics;
use sim_core::particle::{ParticleType, Particles};
use sim_core::spec::{Component, ScenarioSpec};
use std::ops::Range;

/// A group of particles followed through the run.
pub struct Tracked {
    pub kind: String,
    pub range: Range<usize>,
    /// Members that emit light (not dark matter); all members if none do.
    visible: Vec<usize>,
    /// Radius around the center used to re-find it each step: the visible
    /// half-mass radius at t = 0, so tidal tails don't drag the center.
    track_radius: f64,
    pub center: [f64; 3],
}

impl Tracked {
    fn new(particles: &Particles, kind: String, range: Range<usize>) -> Self {
        let dm = ParticleType::DarkMatter as u8;
        let mut visible: Vec<usize> = range
            .clone()
            .filter(|&i| particles.particle_type[i] != dm)
            .collect();
        if visible.is_empty() {
            visible = range.clone().collect();
        }
        let center = mass_center(particles, &visible, None).0;
        let track_radius = half_mass_radius(particles, &visible, center);
        Tracked {
            kind,
            range,
            visible,
            track_radius,
            center,
        }
    }

    /// Re-center on the visible members within `track_radius` of the previous
    /// center (a few fixed-point iterations follow the density peak).
    pub fn update(&mut self, particles: &Particles) {
        if self.visible.len() == 1 || self.track_radius <= 0.0 {
            self.center = mass_center(particles, &self.visible, None).0;
            return;
        }
        for _ in 0..3 {
            let (c, m) = mass_center(
                particles,
                &self.visible,
                Some((self.center, self.track_radius)),
            );
            if m == 0.0 {
                // Dispersed past recognition: fall back to the plain mean
                self.center = mass_center(particles, &self.visible, None).0;
                return;
            }
            self.center = c;
        }
    }

    pub fn state(&self, particles: &Particles, index: usize) -> ComponentState {
        let within = (self.track_radius > 0.0).then_some((self.center, self.track_radius));
        let mut velocity = mass_velocity(particles, &self.visible, within);
        if velocity.is_none() {
            velocity = mass_velocity(particles, &self.visible, None);
        }
        ComponentState {
            index,
            kind: self.kind.clone(),
            particles: self.range.len(),
            center: self.center,
            velocity: velocity.unwrap_or([0.0; 3]),
            half_mass_radius: half_mass_radius(particles, &self.visible, self.center),
        }
    }
}

/// One tracked group per spec component, in index order. `spec` must be the
/// spec as built (after any particle-budget rescale).
pub fn from_spec(particles: &Particles, spec: &ScenarioSpec) -> Vec<Tracked> {
    let mut start = 0;
    let tracked: Vec<Tracked> = spec
        .components
        .iter()
        .map(|c| {
            let range = start..start + c.particle_count();
            start = range.end;
            Tracked::new(particles, c.kind().to_string(), range)
        })
        .collect();
    assert_eq!(
        start, particles.count,
        "spec component counts must cover every particle"
    );
    tracked
}

/// The whole system as a single group, for named scenarios.
pub fn whole(particles: &Particles, name: &str) -> Vec<Tracked> {
    vec![Tracked::new(
        particles,
        name.to_string(),
        0..particles.count,
    )]
}

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

/// Closest approach between two tracked centers over the run.
#[derive(Serialize, Clone)]
pub struct Encounter {
    pub a: usize,
    pub b: usize,
    pub initial_separation: f64,
    pub min_separation: f64,
    pub time_of_min_separation: f64,
    pub final_separation: f64,
}

pub struct EncounterLog {
    pub encounters: Vec<Encounter>,
}

impl EncounterLog {
    pub fn new(tracked: &[Tracked]) -> Self {
        let mut encounters = Vec::new();
        for a in 0..tracked.len() {
            for b in a + 1..tracked.len() {
                let d = dist(tracked[a].center, tracked[b].center);
                encounters.push(Encounter {
                    a,
                    b,
                    initial_separation: d,
                    min_separation: d,
                    time_of_min_separation: 0.0,
                    final_separation: d,
                });
            }
        }
        EncounterLog { encounters }
    }

    pub fn record(&mut self, tracked: &[Tracked], time: f64) {
        for e in &mut self.encounters {
            let d = dist(tracked[e.a].center, tracked[e.b].center);
            e.final_separation = d;
            if d < e.min_separation {
                e.min_separation = d;
                e.time_of_min_separation = time;
            }
        }
    }
}

#[derive(Serialize)]
pub struct ComponentState {
    pub index: usize,
    pub kind: String,
    pub particles: usize,
    /// Density center of the visible matter.
    pub center: [f64; 3],
    /// Bulk velocity of the visible matter near the center.
    pub velocity: [f64; 3],
    /// Radius around `center` holding half the visible mass (grows as the
    /// component is stripped or heated).
    pub half_mass_radius: f64,
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

fn mass_center(
    p: &Particles,
    members: &[usize],
    within: Option<([f64; 3], f64)>,
) -> ([f64; 3], f64) {
    let mut c = [0.0; 3];
    let mut m = 0.0;
    for &i in members {
        let r = [p.x[i], p.y[i], p.z[i]];
        if let Some((c0, rad)) = within
            && dist(r, c0) > rad
        {
            continue;
        }
        m += p.mass[i];
        for k in 0..3 {
            c[k] += p.mass[i] * r[k];
        }
    }
    if m > 0.0 {
        c.iter_mut().for_each(|v| *v /= m);
    }
    (c, m)
}

fn mass_velocity(
    p: &Particles,
    members: &[usize],
    within: Option<([f64; 3], f64)>,
) -> Option<[f64; 3]> {
    let mut v = [0.0; 3];
    let mut m = 0.0;
    for &i in members {
        if let Some((c0, rad)) = within
            && dist([p.x[i], p.y[i], p.z[i]], c0) > rad
        {
            continue;
        }
        m += p.mass[i];
        v[0] += p.mass[i] * p.vx[i];
        v[1] += p.mass[i] * p.vy[i];
        v[2] += p.mass[i] * p.vz[i];
    }
    (m > 0.0).then(|| v.map(|x| x / m))
}

fn half_mass_radius(p: &Particles, members: &[usize], center: [f64; 3]) -> f64 {
    let mut r: Vec<(f64, f64)> = members
        .iter()
        .map(|&i| (dist([p.x[i], p.y[i], p.z[i]], center), p.mass[i]))
        .collect();
    r.sort_by(|a, b| a.0.total_cmp(&b.0));
    let half = 0.5 * r.iter().map(|x| x.1).sum::<f64>();
    let mut acc = 0.0;
    for (d, m) in r {
        acc += m;
        if acc >= half {
            return d;
        }
    }
    0.0
}

pub fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
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

    #[test]
    fn center_ignores_far_tail() {
        // Ten particles at the origin and one far away: the tracked center
        // stays at the clump.
        let mut p = Particles::new(11);
        for k in 0..10 {
            let s = 0.01 * k as f64;
            p.add(s, -s, 0.0, 1.0, 0.0, 0.0, 1.0);
        }
        p.add(100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let mut t = Tracked::new(&p, "x".into(), 0..11);
        t.center = [0.05, -0.05, 0.0];
        t.update(&p);
        assert!(
            dist(t.center, [0.045, -0.045, 0.0]) < 1e-9,
            "center {:?}",
            t.center
        );
        let s = t.state(&p, 0);
        assert!((s.velocity[0] - 1.0).abs() < 1e-12);
    }
}
