use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::scenario::Scenario;
use sim_core::scenarios::two_body::TwoBody;

/// Verify that a two-body Kepler orbit conserves energy to <0.01%
/// over 1000 orbital periods with the leapfrog integrator.
///
/// This is the fundamental test of integrator correctness.
/// The leapfrog (symplectic) integrator should show bounded, oscillating
/// energy errors — not secular drift.
#[test]
fn energy_conservation_1000_orbits() {
    let scenario = TwoBody {
        eccentricity: 0.5,
        ..Default::default()
    };
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BruteForce::new(softening);
    let integrator = LeapfrogKDK;

    // Initialize accelerations
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let initial = diagnostics::compute(&particles, softening, 0.0, 0);

    // Run for 1000 orbits
    let period = scenario.period();
    let steps_per_orbit = (period / dt).round() as u64;
    let total_steps = steps_per_orbit * 1000;

    let mut time = 0.0;
    for _ in 0..total_steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;
    }

    let final_diag = diagnostics::compute(&particles, softening, time, total_steps);
    let de_rel = ((final_diag.total_energy - initial.total_energy) / initial.total_energy.abs()).abs();

    assert!(
        de_rel < 1e-4,
        "Energy drift over 1000 orbits: dE/E = {de_rel:.6e} (limit: 1e-4)"
    );
}

/// Verify angular momentum conservation to machine precision.
#[test]
fn angular_momentum_conservation() {
    let scenario = TwoBody {
        eccentricity: 0.7, // high eccentricity is a harder test
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

    // Run for 100 orbits
    let period = scenario.period();
    let steps = ((100.0 * period) / dt).round() as u64;

    let mut time = 0.0;
    for _ in 0..steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;
    }

    let final_diag = diagnostics::compute(&particles, softening, time, steps);
    let l_final = final_diag.angular_momentum_magnitude();
    let dl_rel = ((l_final - l0) / l0).abs();

    // Angular momentum should be conserved to near machine precision
    // (limited by the softening introducing a small non-central component)
    assert!(
        dl_rel < 1e-6,
        "Angular momentum drift: dL/L = {dl_rel:.6e} (limit: 1e-6)"
    );
}

/// Verify orbital period matches Kepler's third law.
#[test]
fn kepler_third_law_period() {
    let scenario = TwoBody {
        eccentricity: 0.3,
        ..Default::default()
    };
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BruteForce::new(softening);
    let integrator = LeapfrogKDK;

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let expected_period = scenario.period();

    // Track when particle 1 crosses y=0 going positive (periapsis passage)
    let mut last_y = particles.y[1];
    let mut crossing_times = Vec::new();
    let mut time = 0.0;

    // Run for 10 orbits
    let total_steps = (10.0 * expected_period / dt).round() as u64;

    for _ in 0..total_steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;

        let curr_y = particles.y[1];
        // Detect upward zero crossing of y
        if last_y <= 0.0 && curr_y > 0.0 {
            // Linear interpolation for more precise crossing time
            let frac = last_y.abs() / (curr_y - last_y).abs();
            crossing_times.push(time - dt * (1.0 - frac));
        }
        last_y = curr_y;
    }

    // Need at least 2 crossings to measure a period
    assert!(
        crossing_times.len() >= 2,
        "Not enough zero crossings detected: {}",
        crossing_times.len()
    );

    // Measure period from consecutive crossings
    let measured_period = crossing_times[crossing_times.len() - 1] - crossing_times[crossing_times.len() - 2];
    let period_error = ((measured_period - expected_period) / expected_period).abs();

    assert!(
        period_error < 1e-4,
        "Period error: {period_error:.6e} (measured={measured_period:.6e}, expected={expected_period:.6e})"
    );
}
