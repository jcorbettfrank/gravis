//! Integration test: Sedov-Taylor blast wave — verify R ∝ t^{2/5}.

use sim_core::gravity::{NoGravity, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::sedov_blast::{sedov_radius, SedovBlast};
use sim_core::sph::solver::{self, SphSolver};

#[test]
fn sedov_blast_radius_scaling() {
    let scenario = SedovBlast {
        n_particles: 5_000,
        ..SedovBlast::default()
    };
    let mut particles = scenario.generate();

    // Sedov is a pure hydro test — gravity is negligible
    let gravity = NoGravity;
    let mut sph = SphSolver::new();

    // Initial force computation
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);
    let intermediates = sph.compute(&mut particles);
    let mut dt = sph.compute_timestep(&particles, &intermediates).min(1e-4);

    // Collect blast radius at several times
    let mut measurements: Vec<(f64, f64)> = Vec::new();
    let sample_times = [0.01, 0.02, 0.04];

    let mut t = 0.0;
    let mut sample_idx = 0;
    let mut step = 0;

    while sample_idx < sample_times.len() {
        let t_target = sample_times[sample_idx];
        if t + dt > t_target {
            dt = t_target - t;
        }

        let dt_next = solver::step_with_sph(&mut particles, &gravity, &mut sph, dt);
        t += dt;
        dt = dt_next.min(dt * 2.0);
        step += 1;

        if step > 100_000 {
            panic!("Too many steps ({step})");
        }

        if (t - sample_times[sample_idx]).abs() < dt * 0.5 {
            // Measure blast radius: 90th percentile of |v| > threshold
            // particles as the "shock front"
            let mut radii: Vec<f64> = (0..particles.count)
                .filter(|&i| {
                    let v2 = particles.vx[i].powi(2)
                        + particles.vy[i].powi(2)
                        + particles.vz[i].powi(2);
                    v2 > 0.01 // Only particles that have been hit by the shock
                })
                .map(|i| {
                    (particles.x[i].powi(2)
                        + particles.y[i].powi(2)
                        + particles.z[i].powi(2))
                    .sqrt()
                })
                .collect();

            if radii.len() > 10 {
                radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let r_90 = radii[radii.len() * 90 / 100];
                measurements.push((t, r_90));
            }
            sample_idx += 1;
        }
    }

    // Verify R ∝ t^{2/5} by comparing to analytical
    let gamma = 5.0 / 3.0;
    for &(t_meas, r_meas) in &measurements {
        let r_ana = sedov_radius(scenario.e_blast, scenario.rho_0, t_meas, gamma);
        let err = (r_meas - r_ana).abs() / r_ana;
        // Allow 60% error — SPH with 5K particles at early times is rough
        assert!(
            err < 0.6,
            "Sedov at t={:.3}: measured R={:.3}, analytical R={:.3}, error={:.0}%",
            t_meas,
            r_meas,
            r_ana,
            err * 100.0
        );
    }

    // Verify that the blast is actually expanding (radius increases with time)
    if measurements.len() >= 2 {
        for w in measurements.windows(2) {
            assert!(
                w[1].1 > w[0].1,
                "Blast radius should increase: R({:.3})={:.3} < R({:.3})={:.3}",
                w[1].0, w[1].1, w[0].0, w[0].1
            );
        }
    }
}
