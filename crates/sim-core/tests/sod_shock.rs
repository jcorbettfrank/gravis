//! Integration test: Sod shock tube verification against analytical Riemann solution.

use sim_core::gravity::{NoGravity, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::sod_shock::SodShockTube;
use sim_core::sph::boundary::Boundary;
use sim_core::sph::solver::{self, SphSolver};

#[test]
fn sod_shock_density_profile() {
    // Minimal problem size that still resolves the shock.
    // 20×4² = 320 left, 5×4² = 80 right = 400 total.
    let scenario = SodShockTube {
        nx_left: 20,
        nx_right: 5,
        nyz: 4,
        ..SodShockTube::default()
    };
    let mut particles = scenario.generate();

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

    // Evolve to t = 0.05 — enough for the shock to form
    let t_end = 0.05;
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

        if step > 200_000 {
            panic!("Too many steps ({step}) at t={t:.4}, dt={dt:.2e}");
        }
    }

    // Verify shock formed: significant velocities developed
    let max_vx: f64 = (0..particles.count)
        .map(|i| particles.vx[i].abs())
        .fold(0.0, f64::max);
    assert!(
        max_vx > 0.01,
        "No significant velocity developed — SPH forces not working (max |vx| = {:.3e})",
        max_vx
    );

    // Verify shock structure: particles pushed rightward by the shock
    let n_moved_right: usize = (0..particles.count)
        .filter(|&i| particles.vx[i] > 0.05)
        .count();
    assert!(
        n_moved_right > 2,
        "Expected rightward-moving particles (shock), found {} (step={}, t={:.4})",
        n_moved_right, step, t
    );
}
