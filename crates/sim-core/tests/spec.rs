//! Declarative scenario specs: parsing, validation, placement, and the
//! example specs shipped in `specs/`.

use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::particle::{ParticleType, Particles};
use sim_core::scenario::Scenario;
use sim_core::scenario_builder;
use sim_core::spec::{Component, MAX_PARTICLES, ScenarioSpec};

const EXAMPLES: [(&str, &str); 3] = [
    ("tilted-flyby", include_str!("../../../specs/tilted-flyby.json")),
    ("minor-merger", include_str!("../../../specs/minor-merger.json")),
    ("cluster-and-black-hole", include_str!("../../../specs/cluster-and-black-hole.json")),
];

fn parse(json: &str) -> ScenarioSpec {
    ScenarioSpec::from_json(json).unwrap_or_else(|e| panic!("{e}"))
}

fn parse_err(json: &str) -> String {
    ScenarioSpec::from_json(json).expect_err("spec should be rejected")
}

/// Mass-weighted center of position and velocity for particles in `range`.
fn com(p: &Particles, range: std::ops::Range<usize>) -> ([f64; 3], [f64; 3]) {
    let m: f64 = p.mass[range.clone()].iter().sum();
    let (mut x, mut v) = ([0.0; 3], [0.0; 3]);
    for i in range {
        let w = p.mass[i] / m;
        x[0] += w * p.x[i];
        x[1] += w * p.y[i];
        x[2] += w * p.z[i];
        v[0] += w * p.vx[i];
        v[1] += w * p.vy[i];
        v[2] += w * p.vz[i];
    }
    (x, v)
}

#[test]
fn example_specs_build() {
    for (name, json) in EXAMPLES {
        let spec = parse(json);
        let built = scenario_builder::build_spec(&spec, None, None, None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(built.particles.count, spec.particle_count(), "{name}");
        assert!(built.dt > 0.0 && built.dt.is_finite(), "{name}: dt = {}", built.dt);
        assert!(built.softening > 0.0, "{name}: softening = {}", built.softening);
        assert!(
            built.particles.x.iter().all(|x| x.is_finite()),
            "{name}: non-finite positions"
        );
    }
}

#[test]
fn defaults_fill_in_omitted_fields() {
    let spec = parse(r#"{"components": [{"type": "plummer", "particles": 100}]}"#);
    assert_eq!(spec.name, "Custom Scenario");
    assert_eq!(spec.seed, 42);
    let Component::Plummer(p) = &spec.components[0] else { panic!("expected plummer") };
    assert_eq!((p.mass, p.scale_radius, p.position), (1.0, 1.0, [0.0; 3]));
}

#[test]
fn json_round_trip() {
    let spec = parse(EXAMPLES[1].1);
    assert_eq!(parse(&spec.to_json_pretty()), spec);
}

#[test]
fn rejects_unknown_fields_and_types() {
    let e = parse_err(r#"{"components": [{"type": "plummer", "particles": 10, "sclae_radius": 2}]}"#);
    assert!(e.contains("sclae_radius"), "{e}");

    let e = parse_err(r#"{"components": [{"type": "quasar", "particles": 10}]}"#);
    assert!(e.contains("quasar"), "{e}");

    let e = parse_err(r#"{"gravity": 2, "components": []}"#);
    assert!(e.contains("gravity"), "{e}");
}

#[test]
fn validation_names_the_offending_field() {
    let cases = [
        (r#"{"components": []}"#, "at least one component"),
        (r#"{"components": [{"type": "plummer"}]}"#, "components[0] (plummer): particles is required"),
        (
            r#"{"components": [{"type": "point_mass", "mass": 1}, {"type": "gas_cloud", "particles": 9, "radius": -1}]}"#,
            "components[1] (gas_cloud): radius must be a positive number",
        ),
        (
            r#"{"components": [{"type": "disk_galaxy", "particles": 9, "spin_axis": [0, 0, 0]}]}"#,
            "spin_axis must be a non-zero vector",
        ),
        (
            r#"{"components": [{"type": "disk_galaxy", "particles": 9, "disk_fraction": 1.5}]}"#,
            "disk_fraction must be in [0, 1]",
        ),
        (r#"{"dt": 0, "components": [{"type": "point_mass", "mass": 1}]}"#, "spec: dt must be"),
    ];
    for (json, expected) in cases {
        let e = parse_err(json);
        assert!(e.contains(expected), "expected {expected:?} in {e:?}");
    }

    let too_many = format!(
        r#"{{"components": [{{"type": "plummer", "particles": {}}}]}}"#,
        MAX_PARTICLES + 1
    );
    assert!(parse_err(&too_many).contains("limit"));
}

#[test]
fn generation_is_deterministic() {
    let spec = parse(EXAMPLES[0].1).with_particle_budget(600);
    let (a, b) = (spec.generate(), spec.generate());
    assert_eq!(a.x, b.x);
    assert_eq!(a.vz, b.vz);
}

#[test]
fn editing_one_component_leaves_others_alone() {
    let before = parse(
        r#"{"components": [
            {"type": "plummer", "particles": 200, "position": [-5, 0, 0]},
            {"type": "plummer", "particles": 200, "position": [5, 0, 0]}
        ]}"#,
    );
    // Same mass and position, different internal structure
    let mut after = before.clone();
    let Component::Plummer(p) = &mut after.components[1] else { unreachable!() };
    p.scale_radius = 0.3;

    let (a, b) = (before.generate(), after.generate());
    for i in 0..200 {
        assert!((a.x[i] - b.x[i]).abs() < 1e-12 && (a.vy[i] - b.vy[i]).abs() < 1e-12);
    }
    assert_ne!(a.x[200..], b.x[200..]);
}

#[test]
fn components_sit_at_their_position_and_velocity() {
    let spec = parse(
        r#"{"components": [
            {"type": "plummer", "particles": 500, "position": [-5, 1, 0], "velocity": [0, -0.1, 0.2]},
            {"type": "plummer", "particles": 500, "position": [5, -1, 0], "velocity": [0, 0.1, -0.2]}
        ]}"#,
    );
    let p = spec.generate();
    let (x, v) = com(&p, 0..500);
    let expected = ([-5.0, 1.0, 0.0], [0.0, -0.1, 0.2]);
    for k in 0..3 {
        assert!((x[k] - expected.0[k]).abs() < 1e-10, "position {x:?}");
        assert!((v[k] - expected.1[k]).abs() < 1e-10, "velocity {v:?}");
    }
}

#[test]
fn spin_axis_orients_the_disk() {
    for axis in [[1.0_f64, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.866, 0.5], [-0.3, 0.2, -0.9]] {
        let spec = ScenarioSpec::from_json(&format!(
            r#"{{"components": [{{"type": "disk_galaxy", "particles": 2000, "spin_axis": {axis:?}}}]}}"#
        ))
        .unwrap();
        let p = spec.generate();

        // Angular momentum of the disk stars only
        let mut l = [0.0; 3];
        for i in 0..p.count {
            if p.particle_type[i] != ParticleType::DiskStar as u8 {
                continue;
            }
            let m = p.mass[i];
            l[0] += m * (p.y[i] * p.vz[i] - p.z[i] * p.vy[i]);
            l[1] += m * (p.z[i] * p.vx[i] - p.x[i] * p.vz[i]);
            l[2] += m * (p.x[i] * p.vy[i] - p.y[i] * p.vx[i]);
        }
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let l_mag = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        let cos = (l[0] * axis[0] + l[1] * axis[1] + l[2] * axis[2]) / (l_mag * n);
        assert!(cos > 0.99, "axis {axis:?}: L = {l:?}, cos = {cos}");
    }
}

#[test]
fn particle_budget_rescales_extended_components_only() {
    let spec = parse(EXAMPLES[2].1); // point mass + 3000-particle cluster
    let small = spec.with_particle_budget(300);
    assert_eq!(small.particle_count(), 301);

    let flyby = parse(EXAMPLES[0].1).with_particle_budget(1000);
    assert_eq!(flyby.particle_count(), 1000);
}

#[test]
fn timestep_resolves_point_mass_orbits() {
    // Circular binary, separation 1, total mass 1: period 2π
    let binary = parse(
        r#"{"components": [
            {"type": "point_mass", "mass": 0.5, "position": [-0.5, 0, 0], "velocity": [0, -0.5, 0]},
            {"type": "point_mass", "mass": 0.5, "position": [0.5, 0, 0], "velocity": [0, 0.5, 0]}
        ]}"#,
    );
    let period = 2.0 * std::f64::consts::PI;
    let dt = binary.suggested_dt();
    assert!((dt - period / 1000.0).abs() < 1e-9, "dt = {dt}");

    // A dense cluster's tiny softening must not drive dt toward zero: the
    // cluster never comes within its softening length of the black hole.
    let cluster = parse(EXAMPLES[2].1);
    let dt = cluster.suggested_dt();
    assert!((1e-3..1e-2).contains(&dt), "dt = {dt}");
}

#[test]
fn energy_is_conserved_with_suggested_parameters() {
    let spec = parse(EXAMPLES[2].1).with_particle_budget(300);
    let built = scenario_builder::build_spec(&spec, None, None, None).unwrap();
    let mut p = built.particles;
    let gravity = BruteForce::new(built.softening);
    gravity.compute_accelerations(&mut p);
    let e0 = diagnostics::compute(&p, built.softening, 0.0, 0).total_energy;
    for _ in 0..500 {
        LeapfrogKDK.step(&mut p, &gravity, built.dt);
    }
    let e1 = diagnostics::compute(&p, built.softening, 0.0, 500).total_energy;
    let de = ((e1 - e0) / e0).abs();
    assert!(de < 1e-3, "dE/E = {de:e}");
}

#[test]
fn view_radius_frames_every_component() {
    let flyby = parse(EXAMPLES[0].1);
    // Each disk is ~7.57 from the center, plus 4 disk scale radii
    let r = flyby.view_radius();
    assert!((11.0..12.5).contains(&r), "view radius {r}");

    let mut fixed = flyby.clone();
    fixed.view_radius = Some(30.0);
    assert_eq!(fixed.view_radius(), 30.0);
}

#[test]
fn gas_cloud_matches_evrard_collapse() {
    use sim_core::scenarios::evrard_collapse::EvrardCollapse;
    let spec = parse(r#"{"components": [{"type": "gas_cloud", "particles": 500}]}"#);
    let evrard = EvrardCollapse { n_particles: 500, ..Default::default() };
    let (a, b) = (spec.generate(), evrard.generate());
    assert!(spec.has_gas());
    assert_eq!(a.count, b.count);
    // Same sampler and seed; only the COM recentering differs
    for i in 0..a.count {
        assert!((a.x[i] - b.x[i]).abs() < 0.05, "particle {i}");
    }
    assert_eq!(a.internal_energy, b.internal_energy);
    assert_eq!(a.smoothing_length, b.smoothing_length);
    assert_eq!(spec.suggested_softening(), evrard.suggested_softening());
}
