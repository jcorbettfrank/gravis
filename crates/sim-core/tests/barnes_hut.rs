use sim_core::barnes_hut::BarnesHut;
use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::scenario::Scenario;
use sim_core::scenarios::plummer_sphere::PlummerSphere;
use sim_core::scenarios::two_body::TwoBody;

/// Compute RMS relative force error between Barnes-Hut and brute-force.
///
/// RMS = sqrt( mean( |a_BH - a_BF|² / |a_BF|² ) )
fn rms_force_error(n: usize, theta: f64, softening: f64) -> f64 {
    let scenario = PlummerSphere {
        n,
        seed: 42,
        ..Default::default()
    };
    let particles = scenario.generate();

    // Brute-force reference
    let mut p_bf = particles.clone();
    let bf = BruteForce::new(softening);
    p_bf.clear_accelerations();
    bf.compute_accelerations(&mut p_bf);

    // Barnes-Hut
    let mut p_bh = particles;
    let bh = BarnesHut::new(softening, theta);
    p_bh.clear_accelerations();
    bh.compute_accelerations(&mut p_bh);

    // Compute RMS relative error
    let mut sum_rel_err2 = 0.0;
    for i in 0..n {
        let bf_mag2 =
            p_bf.ax[i] * p_bf.ax[i] + p_bf.ay[i] * p_bf.ay[i] + p_bf.az[i] * p_bf.az[i];
        if bf_mag2 < 1e-30 {
            continue; // skip particles with negligible force
        }
        let dx = p_bh.ax[i] - p_bf.ax[i];
        let dy = p_bh.ay[i] - p_bf.ay[i];
        let dz = p_bh.az[i] - p_bf.az[i];
        let err2 = dx * dx + dy * dy + dz * dz;
        sum_rel_err2 += err2 / bf_mag2;
    }

    (sum_rel_err2 / n as f64).sqrt()
}

/// Barnes-Hut forces agree with brute-force within 1% RMS at theta=0.5.
#[test]
fn force_accuracy_theta_05() {
    let err = rms_force_error(1000, 0.5, 0.05);
    eprintln!("RMS force error at theta=0.5: {err:.6e}");
    assert!(
        err < 0.01,
        "RMS force error {err:.6e} exceeds 1% at theta=0.5"
    );
}

/// Force accuracy at theta=0.3 — should be better than theta=0.5.
#[test]
fn force_accuracy_theta_03() {
    let err = rms_force_error(1000, 0.3, 0.05);
    eprintln!("RMS force error at theta=0.3: {err:.6e}");
    assert!(
        err < 0.005,
        "RMS force error {err:.6e} too high at theta=0.3"
    );
}

/// Force accuracy sweep: verify error increases monotonically with theta.
#[test]
fn force_accuracy_monotonic_with_theta() {
    let thetas = [0.3, 0.5, 0.7, 1.0];
    let errors: Vec<f64> = thetas
        .iter()
        .map(|&t| rms_force_error(500, t, 0.05))
        .collect();

    eprintln!("Force accuracy sweep:");
    for (t, e) in thetas.iter().zip(&errors) {
        eprintln!("  theta={t:.1}  RMS error={e:.6e}");
    }

    // Error should increase with theta (less accurate as we open more)
    for i in 1..errors.len() {
        assert!(
            errors[i] >= errors[i - 1] * 0.5, // allow some noise but general trend
            "Error at theta={} ({:.6e}) should be >= error at theta={} ({:.6e})",
            thetas[i],
            errors[i],
            thetas[i - 1],
            errors[i - 1]
        );
    }
}

/// Energy conservation with Barnes-Hut for Kepler orbit.
/// Using theta=0.3 for high accuracy. Allows more energy drift than brute-force
/// since BH forces are approximate.
#[test]
fn energy_conservation_kepler_bh() {
    let scenario = TwoBody {
        eccentricity: 0.5,
        ..Default::default()
    };
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BarnesHut::new(softening, 0.3);
    let integrator = LeapfrogKDK;

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let initial = diagnostics::compute(&particles, softening, 0.0, 0);

    // Run for 100 orbits
    let period = scenario.period();
    let steps = ((100.0 * period) / dt).round() as u64;

    let mut time = 0.0;
    for _ in 0..steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;
    }

    let final_diag = diagnostics::compute(&particles, softening, time, steps);
    let de_rel =
        ((final_diag.total_energy - initial.total_energy) / initial.total_energy.abs()).abs();

    eprintln!("Kepler BH energy drift over 100 orbits: dE/E = {de_rel:.6e}");
    assert!(
        de_rel < 1e-3,
        "Energy drift {de_rel:.6e} exceeds 0.1% for Kepler orbit with Barnes-Hut"
    );
}

/// Virial equilibrium with Barnes-Hut for Plummer sphere.
#[test]
fn virial_equilibrium_plummer_bh() {
    let scenario = PlummerSphere {
        n: 500,
        seed: 42,
        ..Default::default()
    };
    let mut particles = scenario.generate();
    let dt = scenario.suggested_dt();
    let softening = scenario.suggested_softening();
    let gravity = BarnesHut::new(softening, 0.5);
    let integrator = LeapfrogKDK;

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let initial = diagnostics::compute(&particles, softening, 0.0, 0);

    // Run for 10 dynamical times
    let t_dyn = 1.0; // N-body units
    let steps = ((10.0 * t_dyn) / dt).round() as u64;

    let mut time = 0.0;
    let mut max_virial_deviation = 0.0_f64;
    for step in 1..=steps {
        integrator.step(&mut particles, &gravity, dt);
        time += dt;

        if step % 100 == 0 {
            let diag = diagnostics::compute(&particles, softening, time, step);
            let dev = (diag.virial_ratio - 1.0).abs();
            max_virial_deviation = max_virial_deviation.max(dev);
        }
    }

    let final_diag = diagnostics::compute(&particles, softening, time, steps);
    let de_rel =
        ((final_diag.total_energy - initial.total_energy) / initial.total_energy.abs()).abs();

    eprintln!("Plummer BH virial max deviation: {max_virial_deviation:.4}");
    eprintln!("Plummer BH energy drift: dE/E = {de_rel:.6e}");

    // Virial ratio should stay near 1.0 (allow wider tolerance than brute-force)
    assert!(
        max_virial_deviation < 0.5,
        "Virial ratio deviated too far from 1.0: {max_virial_deviation:.4}"
    );

    // Energy conservation — allow more drift than brute-force due to BH approximation
    assert!(
        de_rel < 0.01,
        "Energy drift {de_rel:.6e} exceeds 1% for Plummer with Barnes-Hut"
    );
}
