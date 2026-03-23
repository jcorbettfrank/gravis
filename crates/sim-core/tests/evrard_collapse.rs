//! Integration test: Evrard collapse — energy conservation with gravity + SPH.

use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::evrard_collapse::EvrardCollapse;
use sim_core::sph::solver::{self, SphSolver};

#[test]
fn evrard_energy_conservation() {
    let scenario = EvrardCollapse {
        n_particles: 1_000, // Small N for test speed
        ..EvrardCollapse::default()
    };
    let mut particles = scenario.generate();

    let softening = scenario.suggested_softening();
    let gravity = BruteForce::new(softening);
    let mut sph = SphSolver::new();

    // Initial force computation
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);
    let intermediates = sph.compute(&mut particles);
    let mut dt = sph.compute_timestep(&particles, &intermediates).min(1e-3);

    // Record initial energy
    let diag_0 = diagnostics::compute(&particles, softening, 0.0, 0);
    let e_0 = diag_0.total_energy;

    // Evolve for some time (partial collapse — not all the way to bounce)
    let t_end = 0.5; // About half the free-fall time
    let mut t = 0.0;
    let mut step = 0;

    while t < t_end {
        if t + dt > t_end {
            dt = t_end - t;
        }
        let dt_next = solver::step_with_sph(&mut particles, &gravity, &mut sph, dt);
        t += dt;
        dt = dt_next.min(dt * 2.0);
        step += 1;

        if step > 100_000 {
            panic!("Too many steps ({step})");
        }
    }

    // Check energy conservation
    let diag_f = diagnostics::compute(&particles, softening, t, step as u64);
    let e_f = diag_f.total_energy;
    let de = (e_f - e_0).abs() / e_0.abs();

    assert!(
        de < 0.05,
        "Evrard energy conservation: ΔE/|E₀| = {:.1}% (expected <5%)",
        de * 100.0
    );

    // Verify the system is actually collapsing: kinetic energy should increase
    assert!(
        diag_f.kinetic_energy > diag_0.kinetic_energy * 2.0,
        "Expected significant kinetic energy gain during collapse: K_0={:.3e}, K_f={:.3e}",
        diag_0.kinetic_energy,
        diag_f.kinetic_energy,
    );

    // Thermal energy should also increase (PdV work heats the gas)
    assert!(
        diag_f.thermal_energy > diag_0.thermal_energy,
        "Expected thermal energy increase during collapse: E_th_0={:.3e}, E_th_f={:.3e}",
        diag_0.thermal_energy,
        diag_f.thermal_energy,
    );
}
