use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::scenario::Scenario;
use sim_core::scenarios::plummer_sphere::PlummerSphere;

/// Verify that a Plummer sphere remains in virial equilibrium.
///
/// The virial theorem states that for a self-gravitating system in
/// equilibrium: 2K + U = 0, or equivalently 2K/|U| = 1.
///
/// Starting from proper Plummer sphere ICs, the virial ratio should
/// fluctuate around 1.0 without secular drift.
#[test]
fn virial_equilibrium() {
    let scenario = PlummerSphere {
        n: 500,
        seed: 42,
        ..Default::default()
    };
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BruteForce::new(softening);
    let integrator = LeapfrogKDK;

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    // Run for 20 dynamical times, sampling the virial ratio
    let t_dyn = 1.0; // In N-body units with a=1, M=1, G=1
    let total_steps = (20.0 * t_dyn / dt).round() as u64;
    let sample_interval = (t_dyn / dt).round() as u64; // sample once per t_dyn

    let mut virial_samples = Vec::new();
    let mut time = 0.0;

    for step in 1..=total_steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;

        if step % sample_interval == 0 {
            let diag = diagnostics::compute(&particles, softening, time, step);
            virial_samples.push(diag.virial_ratio);
        }
    }

    // The mean virial ratio should be close to 1.0
    let mean_virial: f64 = virial_samples.iter().sum::<f64>() / virial_samples.len() as f64;

    assert!(
        (mean_virial - 1.0).abs() < 0.15,
        "Mean virial ratio {mean_virial:.4} deviates from 1.0 by more than 0.15"
    );

    // No sample should be wildly off (indicates instability)
    for (i, &v) in virial_samples.iter().enumerate() {
        assert!(
            v > 0.5 && v < 1.8,
            "Virial ratio at sample {i} is {v:.4} — system may be unstable"
        );
    }
}

/// Verify energy conservation for Plummer sphere over many dynamical times.
#[test]
fn energy_conservation() {
    let scenario = PlummerSphere {
        n: 300,
        seed: 123,
        ..Default::default()
    };
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BruteForce::new(softening);
    let integrator = LeapfrogKDK;

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let initial = diagnostics::compute(&particles, softening, 0.0, 0);

    // Run for 50 dynamical times
    let total_steps = (50.0 / dt).round() as u64;
    let mut time = 0.0;

    for _ in 0..total_steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;
    }

    let final_diag = diagnostics::compute(&particles, softening, time, total_steps);
    let de_rel = ((final_diag.total_energy - initial.total_energy) / initial.total_energy.abs()).abs();

    // Energy should be conserved within 0.1% for a Plummer sphere with N=300
    assert!(
        de_rel < 1e-3,
        "Energy drift in Plummer sphere: dE/E = {de_rel:.6e} (limit: 1e-3)"
    );
}
