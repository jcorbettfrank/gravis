//! The quick check run behind the browser's text-to-universe page.

use sim_core::preview::{Preview, PreviewConfig};
use sim_core::spec::ScenarioSpec;

fn spec(json: &str) -> ScenarioSpec {
    ScenarioSpec::from_json(json).unwrap_or_else(|e| panic!("{e}"))
}

fn cluster() -> ScenarioSpec {
    spec(include_str!("../../../specs/cluster-and-black-hole.json"))
}

fn config(particles: usize, until: f64) -> PreviewConfig {
    PreviewConfig {
        particles,
        until,
        ..Default::default()
    }
}

/// The cluster passes the black hole at t ≈ 7.2, 0.47 away: Kepler is exact
/// for a point mass and a compact cluster (ADR 013's calibration table).
#[test]
fn finds_the_cluster_pericenter() {
    let mut p = Preview::new(&cluster(), &config(500, 10.0)).unwrap();
    // The cluster is rescaled to the budget; the black hole stays one particle.
    assert_eq!(p.particle_count(), 501);
    while !p.advance(200) {}
    let r = p.report();
    assert!(r.done && !r.truncated);
    assert_eq!(r.requested_particles, 3001);
    assert!((r.sim_time - 10.0).abs() < r.dt, "ran to {}", r.sim_time);
    assert_eq!(r.energy_verdict, "pass", "drift {}", r.max_abs_energy_drift);

    let e = &r.encounters[0];
    let passage = e.pericenters.first().expect("one completed passage");
    assert!(
        (passage.time - 7.2).abs() < 0.3,
        "passage at t = {}",
        passage.time
    );
    assert!(
        (passage.separation - 0.47).abs() < 0.05,
        "passage at {}",
        passage.separation
    );
    assert!(e.min_separation < e.initial_separation);
    assert_eq!(r.initial.len(), 2);
    assert_eq!(r.components[1].kind, "plummer");
}

/// Chunking is invisible: the same spec stepped in different chunk sizes ends
/// in the same state, so the page's worker can pace it however it likes.
#[test]
fn chunking_does_not_change_the_result() {
    let cfg = config(300, 2.0);
    let mut a = Preview::new(&cluster(), &cfg).unwrap();
    let mut b = Preview::new(&cluster(), &cfg).unwrap();
    while !a.advance(1) {}
    while !b.advance(37) {}
    let (ra, rb) = (a.report(), b.report());
    assert_eq!(ra.steps, rb.steps);
    assert_eq!(ra.max_abs_energy_drift, rb.max_abs_energy_drift);
    assert_eq!(ra.components[1].center, rb.components[1].center);
    assert!((a.progress() - 1.0).abs() < 1e-12);
}

#[test]
fn step_cap_truncates_and_says_so() {
    let cfg = PreviewConfig {
        max_steps: 50,
        ..config(300, 10.0)
    };
    let mut p = Preview::new(&cluster(), &cfg).unwrap();
    assert!(p.advance(1000), "stops at the cap");
    let r = p.report();
    assert_eq!(r.steps, 50);
    assert!(r.truncated && r.sim_time < 10.0);
}

#[test]
fn rejects_gas_and_bad_times() {
    let gas = spec(r#"{"components": [{"type": "gas_cloud", "particles": 200}]}"#);
    let err = Preview::new(&gas, &config(200, 1.0)).err().unwrap();
    assert!(err.contains("gas_cloud"), "{err}");
    let err = Preview::new(&cluster(), &config(200, 0.0)).err().unwrap();
    assert!(err.contains("positive"), "{err}");
}
