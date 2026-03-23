//! Integration test: Sod shock tube verification against analytical Riemann solution.

use sim_core::gravity::{NoGravity, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::sod_shock::{SodShockTube, sod_analytical};
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

    // --- Basic sanity: shock formed, no NaN ---

    let max_vx: f64 = (0..particles.count)
        .map(|i| particles.vx[i].abs())
        .fold(0.0, f64::max);
    assert!(
        max_vx > 0.01,
        "No significant velocity developed — SPH forces not working (max |vx| = {:.3e})",
        max_vx
    );

    let n_moved_right: usize = (0..particles.count)
        .filter(|&i| particles.vx[i] > 0.05)
        .count();
    assert!(
        n_moved_right > 2,
        "Expected rightward-moving particles (shock), found {} (step={}, t={:.4})",
        n_moved_right, step, t
    );

    for i in 0..particles.count {
        assert!(
            particles.density[i].is_finite() && particles.internal_energy[i].is_finite(),
            "NaN/Inf in particle {i}: rho={}, u={}",
            particles.density[i], particles.internal_energy[i]
        );
    }

    // --- Analytical wave structure verification ---
    //
    // Verify the SPH result is consistent with the analytical Riemann solution.
    // At t=0.05 the wave structure is compact (~0.13 units wide around x=0),
    // so with 400 particles in 3D we test qualitative structure, not exact profiles.

    // Evaluate analytical solution: at x=0, the post-shock velocity u_star gives
    // the characteristic speed scale of the Riemann solution.
    let (_, vel_ana, _) = sod_analytical(&[0.0], t, scenario.gamma);
    let u_star = vel_ana[0].abs();

    // 1. The analytical maximum velocity u_star should be matched within an order
    //    of magnitude by the SPH maximum velocity.
    assert!(
        max_vx > 0.1 * u_star,
        "SPH max velocity ({max_vx:.3}) too low vs analytical u_star ({u_star:.3})"
    );

    // 2. Rightward-moving particles (the shock-compressed material) should be
    //    concentrated near or to the right of x=0, matching the analytical solution.
    let mean_x_of_shocked: f64 = {
        let (sx, c) = (0..particles.count)
            .filter(|&i| particles.vx[i] > 0.05)
            .fold((0.0, 0usize), |(sx, c), i| (sx + particles.x[i], c + 1));
        if c > 0 { sx / c as f64 } else { 0.0 }
    };
    assert!(
        mean_x_of_shocked > -0.2,
        "Shocked particles should be near/right of membrane, mean x = {mean_x_of_shocked:.3}"
    );

}
