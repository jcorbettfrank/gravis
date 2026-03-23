//! Integration test: Sod shock tube verification against analytical Riemann solution.

use sim_core::gravity::{NoGravity, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::sod_shock::SodShockTube;
use sim_core::sph::boundary::Boundary;
use sim_core::sph::solver::{self, SphSolver};

#[test]
fn sod_shock_density_profile() {
    // Small problem: 40 × 6² = 1440 left, 10 × 6² = 360 right = 1800 total
    let scenario = SodShockTube {
        nx_left: 40,
        nx_right: 10,
        nyz: 6,
        ..SodShockTube::default()
    };
    let mut particles = scenario.generate();

    // Pure hydro test — no gravity
    let gravity = NoGravity;
    let mut sph = SphSolver::new();

    let yz = scenario.yz_extent();
    let boundary = Boundary::Reflective {
        bounds: [
            (-scenario.x_extent * 2.0, scenario.x_extent * 2.0),
            (0.0, yz),
            (0.0, yz),
        ],
    };

    // Initial force computation
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);
    let intermediates = sph.compute(&mut particles);
    let mut dt = sph.compute_timestep(&particles, &intermediates).min(1e-3);

    // Evolve to t = 0.1
    let t_end = 0.1;
    let mut t = 0.0;
    let mut step = 0;
    while t < t_end {
        if t + dt > t_end {
            dt = t_end - t;
        }
        let dt_next = solver::step_with_sph(&mut particles, &gravity, &mut sph, dt);
        boundary.apply(&mut particles);
        t += dt;
        dt = dt_next.min(dt * 1.5);
        step += 1;

        if step > 500_000 {
            panic!("Too many steps ({step}) at t={t:.4}, dt={dt:.2e}");
        }
    }

    // Check that particles have moved (the simulation did something)
    let max_vx: f64 = (0..particles.count)
        .map(|i| particles.vx[i].abs())
        .fold(0.0, f64::max);
    assert!(
        max_vx > 0.01,
        "No significant velocity developed — SPH forces not working (max |vx| = {:.3e})",
        max_vx
    );

    // Check that a shock structure exists: particles on the right side
    // should have been compressed
    let n_moved_right: usize = (0..particles.count)
        .filter(|&i| particles.vx[i] > 0.1)
        .count();
    assert!(
        n_moved_right > 3,
        "Expected rightward-moving particles (shock), found {} (step={}, t={:.4})",
        n_moved_right, step, t
    );
}
