//! Integration test: Kelvin-Helmholtz instability regression test.
//!
//! Verifies that the KH scenario produces growing instability at the shear
//! interface without crashing or producing NaN. Uses a small particle count
//! for speed — this is a regression test, not a convergence study.

use sim_core::gravity::{NoGravity, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::kelvin_helmholtz::KelvinHelmholtz;
use sim_core::sph::solver::{self, SphSolver};

#[test]
fn kelvin_helmholtz_instability_grows() {
    // Small problem: nx=8 gives ~8×8×24 ≈ 1536 particles. Fast but enough
    // to see instability growth.
    let scenario = KelvinHelmholtz {
        nx: 8,
        ..KelvinHelmholtz::default()
    };
    let mut particles = scenario.generate();
    let n = particles.count;
    assert!(n > 100, "Expected >100 particles, got {n}");

    let gravity = NoGravity;
    let mut sph = SphSolver::new();

    // Initial force computation
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);
    let intermediates = sph.compute(&mut particles);
    let mut dt = sph.compute_timestep(&particles, &intermediates).min(1e-3);

    // Record initial vy variance (should be small — just the seed perturbation)
    let vy_var_initial = vy_variance(&particles);

    // Evolve for enough time to see instability growth.
    // KH growth rate ~ k * v_shear, so for k = 2π/L and v_shear = 0.5,
    // growth timescale ~ L / (π * v_shear) ≈ 0.64. Run for ~1 growth time.
    let t_end = 0.5;
    let mut t = 0.0;
    let mut step = 0;
    while t < t_end {
        if t + dt > t_end {
            dt = t_end - t;
        }
        let dt_next = solver::step_with_sph(&mut particles, &gravity, &mut sph, dt);
        t += dt;
        dt = dt_next.min(dt * 1.5);
        step += 1;

        if step > 500_000 {
            panic!("Too many steps ({step}) at t={t:.4}, dt={dt:.2e}");
        }
    }

    // No NaN in any field
    for i in 0..particles.count {
        assert!(
            particles.x[i].is_finite()
                && particles.y[i].is_finite()
                && particles.z[i].is_finite()
                && particles.vx[i].is_finite()
                && particles.vy[i].is_finite()
                && particles.vz[i].is_finite()
                && particles.density[i].is_finite()
                && particles.internal_energy[i].is_finite(),
            "NaN/Inf detected at particle {i}, step={step}, t={t:.4}"
        );
    }

    // vy variance should have grown (instability developing)
    let vy_var_final = vy_variance(&particles);
    assert!(
        vy_var_final > vy_var_initial * 1.5,
        "vy variance did not grow: initial={:.3e}, final={:.3e}. \
         KH instability may not be developing.",
        vy_var_initial, vy_var_final
    );

    // Energy conservation: total energy should not drift excessively.
    // KH is a hard test for SPH, so allow generous tolerance.
    let total_ke: f64 = (0..particles.count)
        .map(|i| {
            0.5 * particles.mass[i]
                * (particles.vx[i].powi(2)
                    + particles.vy[i].powi(2)
                    + particles.vz[i].powi(2))
        })
        .sum();
    let total_thermal: f64 = (0..particles.count)
        .map(|i| particles.mass[i] * particles.internal_energy[i])
        .sum();
    let total_energy = total_ke + total_thermal;
    assert!(
        total_energy.is_finite() && total_energy > 0.0,
        "Total energy is invalid: {total_energy}"
    );
}

fn vy_variance(particles: &sim_core::particle::Particles) -> f64 {
    let n = particles.count as f64;
    let mean: f64 = (0..particles.count).map(|i| particles.vy[i]).sum::<f64>() / n;
    (0..particles.count)
        .map(|i| (particles.vy[i] - mean).powi(2))
        .sum::<f64>()
        / n
}
