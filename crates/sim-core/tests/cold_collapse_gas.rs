//! Smoke test: cold collapse with gas particles (SPH).

use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::cold_collapse::ColdCollapse;
use sim_core::sph::solver::{self, SphSolver};

#[test]
fn cold_collapse_gas_smoke() {
    let scenario = ColdCollapse {
        n: 500,
        sph: true,
        ..Default::default()
    };
    let mut particles = scenario.generate();

    assert!(particles.has_gas(), "SPH cold collapse should have gas particles");

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

        // Check no NaN/Inf every 10 steps
        if step % 10 == 0 {
            for i in 0..particles.count {
                assert!(
                    particles.x[i].is_finite()
                        && particles.density[i].is_finite()
                        && particles.internal_energy[i].is_finite(),
                    "NaN/Inf at step {step}, particle {i}: x={}, rho={}, u={}",
                    particles.x[i], particles.density[i], particles.internal_energy[i]
                );
            }
        }
    }
}
