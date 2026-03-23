//! Smoke test: protoplanetary disk scenario.

use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::protoplanetary::Protoplanetary;
use sim_core::sph::solver::{self, SphSolver};

#[test]
fn protoplanetary_smoke() {
    let scenario = Protoplanetary {
        n_gas: 500,
        ..Default::default()
    };
    let mut particles = scenario.generate();

    assert!(particles.has_gas(), "Protoplanetary should have gas particles");
    assert_eq!(particles.count, 501, "500 gas + 1 star");

    let gravity = BruteForce::new(scenario.suggested_softening());
    let mut sph = SphSolver::new();

    // Initialize
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);
    let intermediates = sph.compute(&mut particles);
    let mut dt = sph.compute_timestep(&particles, &intermediates).min(scenario.suggested_dt());

    // Evolve 50 steps
    for step in 0..50 {
        let dt_next = solver::step_with_sph(&mut particles, &gravity, &mut sph, dt);
        dt = dt_next.min(dt * 1.5);

        if step % 10 == 0 {
            for i in 0..particles.count {
                assert!(
                    particles.x[i].is_finite()
                        && particles.y[i].is_finite()
                        && particles.z[i].is_finite(),
                    "NaN/Inf position at step {step}, particle {i}"
                );
            }
        }
    }

    // Particles should remain bound: no particle more than 10× the outer radius
    let r_max = 10.0 * scenario.r_out;
    for i in 0..particles.count {
        let r = (particles.x[i].powi(2) + particles.y[i].powi(2) + particles.z[i].powi(2)).sqrt();
        assert!(
            r < r_max,
            "Particle {i} escaped: r={r:.2} > {r_max:.2}"
        );
    }
}
