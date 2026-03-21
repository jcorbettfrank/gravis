use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::scenario::Scenario;
use sim_core::scenarios::plummer_sphere::PlummerSphere;
use sim_core::scenarios::two_body::TwoBody;

/// Verify linear momentum conservation for a multi-body system.
///
/// With the pairwise-symmetric force calculation (F_ij = -F_ji),
/// total momentum should be conserved to machine precision regardless
/// of integrator choice. The Plummer sphere ICs are centered on the
/// COM frame, so total momentum starts at zero.
#[test]
fn linear_momentum_conservation_plummer() {
    let scenario = PlummerSphere {
        n: 200,
        seed: 77,
        ..Default::default()
    };
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BruteForce::new(softening);
    let integrator = LeapfrogKDK;

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    // Run 1000 steps
    let mut time = 0.0;
    for _ in 0..1000 {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;
    }

    let diag = diagnostics::compute(&particles, softening, time, 1000);

    // Momentum magnitude should be near machine precision
    assert!(
        diag.momentum_magnitude() < 1e-12,
        "Momentum magnitude: {:.6e} (should be near machine precision)",
        diag.momentum_magnitude()
    );
}

/// Verify center-of-mass remains stationary.
#[test]
fn com_drift_plummer() {
    let scenario = PlummerSphere {
        n: 200,
        seed: 99,
        ..Default::default()
    };
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BruteForce::new(softening);
    let integrator = LeapfrogKDK;

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let mut time = 0.0;
    for _ in 0..1000 {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;
    }

    let diag = diagnostics::compute(&particles, softening, time, 1000);

    // COM should remain at origin to machine precision
    assert!(
        diag.com_drift() < 1e-12,
        "COM drift: {:.6e} (should be near machine precision)",
        diag.com_drift()
    );
}

/// Verify angular momentum conservation for two-body orbit.
///
/// For a central force, angular momentum is exactly conserved.
/// With Plummer softening, the force is still central (depends only
/// on |r|), so angular momentum should be conserved to machine precision.
#[test]
fn angular_momentum_conservation_two_body() {
    let scenario = TwoBody {
        eccentricity: 0.8,
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
    let l0 = initial.angular_momentum_magnitude();

    // 50 orbits
    let period = scenario.period();
    let steps = ((50.0 * period) / dt).round() as u64;
    let mut time = 0.0;

    for _ in 0..steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;
    }

    let final_diag = diagnostics::compute(&particles, softening, time, steps);
    let l_final = final_diag.angular_momentum_magnitude();
    let dl_rel = ((l_final - l0) / l0).abs();

    assert!(
        dl_rel < 1e-8,
        "Angular momentum drift: dL/L = {dl_rel:.6e} (limit: 1e-8)"
    );
}

/// Verify linear momentum conservation for two-body (should be exactly zero).
#[test]
fn linear_momentum_conservation_two_body() {
    let scenario = TwoBody::default();
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BruteForce::new(softening);
    let integrator = LeapfrogKDK;

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let period = scenario.period();
    let steps = ((10.0 * period) / dt).round() as u64;
    let mut time = 0.0;

    for _ in 0..steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;
    }

    let diag = diagnostics::compute(&particles, softening, time, steps);

    assert!(
        diag.momentum_magnitude() < 1e-14,
        "Two-body momentum: {:.6e} (should be near machine precision)",
        diag.momentum_magnitude()
    );
}
