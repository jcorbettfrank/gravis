//! Generate M3 benchmark artifacts.
//!
//! Run with: cargo test -p sim-core --release -- m3_artifacts --nocapture --ignored
//!
//! Generates:
//!   artifacts/benchmarks/m3_scaling.csv
//!   artifacts/benchmarks/m3_accuracy.csv

use std::fs;
use std::io::Write;
use std::time::Instant;

use sim_core::barnes_hut::BarnesHut;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::plummer_sphere::PlummerSphere;

#[test]
#[ignore] // Only run explicitly for artifact generation
fn generate_m3_scaling_csv() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let dir = workspace.join("artifacts/benchmarks");
    let dir = dir.to_str().unwrap();
    fs::create_dir_all(dir).unwrap();

    let mut out = fs::File::create(format!("{dir}/m3_scaling.csv")).unwrap();
    writeln!(out, "algorithm,n,theta,wall_time_ms,steps_per_sec").unwrap();

    let softening = 0.05;
    let iters = 5;

    // Brute-force scaling (up to 10K — higher is too slow)
    for &n in &[1_000, 2_000, 5_000, 10_000] {
        let scenario = PlummerSphere {
            n,
            seed: 42,
            ..Default::default()
        };
        let mut particles = scenario.generate();
        let solver = BruteForce::new(softening);

        // Warm up
        particles.clear_accelerations();
        solver.compute_accelerations(&mut particles);

        let start = Instant::now();
        for _ in 0..iters {
            particles.clear_accelerations();
            solver.compute_accelerations(&mut particles);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let steps_per_sec = 1000.0 / elapsed_ms;

        eprintln!("brute-force  N={n:>7}  {elapsed_ms:.2}ms");
        writeln!(out, "brute-force,{n},,{elapsed_ms:.3},{steps_per_sec:.1}").unwrap();
    }

    // Barnes-Hut scaling
    for &n in &[1_000, 5_000, 10_000, 50_000, 100_000, 500_000] {
        let scenario = PlummerSphere {
            n,
            seed: 42,
            ..Default::default()
        };
        let mut particles = scenario.generate();
        let solver = BarnesHut::new(softening, 0.5);

        // Warm up
        particles.clear_accelerations();
        solver.compute_accelerations(&mut particles);

        let run_iters = if n >= 100_000 { 3 } else { iters };
        let start = Instant::now();
        for _ in 0..run_iters {
            particles.clear_accelerations();
            solver.compute_accelerations(&mut particles);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / run_iters as f64;
        let steps_per_sec = 1000.0 / elapsed_ms;

        eprintln!("barnes-hut   N={n:>7}  {elapsed_ms:.2}ms");
        writeln!(out, "barnes-hut,{n},0.5,{elapsed_ms:.3},{steps_per_sec:.1}").unwrap();
    }

    eprintln!("Wrote {dir}/m3_scaling.csv");
}

#[test]
#[ignore]
fn generate_m3_accuracy_csv() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let dir = workspace.join("artifacts/benchmarks");
    let dir = dir.to_str().unwrap();
    fs::create_dir_all(dir).unwrap();

    let mut out = fs::File::create(format!("{dir}/m3_accuracy.csv")).unwrap();
    writeln!(out, "theta,n,rms_force_error,max_force_error").unwrap();

    let n = 1000;
    let softening = 0.05;

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

    for &theta in &[0.3, 0.5, 0.7, 1.0] {
        let mut p_bh = particles.clone();
        let bh = BarnesHut::new(softening, theta);
        p_bh.clear_accelerations();
        bh.compute_accelerations(&mut p_bh);

        let mut sum_rel_err2 = 0.0;
        let mut max_rel_err = 0.0_f64;

        for i in 0..n {
            let bf_mag2 =
                p_bf.ax[i] * p_bf.ax[i] + p_bf.ay[i] * p_bf.ay[i] + p_bf.az[i] * p_bf.az[i];
            if bf_mag2 < 1e-30 {
                continue;
            }
            let dx = p_bh.ax[i] - p_bf.ax[i];
            let dy = p_bh.ay[i] - p_bf.ay[i];
            let dz = p_bh.az[i] - p_bf.az[i];
            let err2 = dx * dx + dy * dy + dz * dz;
            let rel_err = (err2 / bf_mag2).sqrt();
            sum_rel_err2 += err2 / bf_mag2;
            max_rel_err = max_rel_err.max(rel_err);
        }

        let rms_err = (sum_rel_err2 / n as f64).sqrt();

        eprintln!("theta={theta:.1}  RMS={rms_err:.6e}  max={max_rel_err:.6e}");
        writeln!(out, "{theta},{n},{rms_err:.6e},{max_rel_err:.6e}").unwrap();
    }

    eprintln!("Wrote {dir}/m3_accuracy.csv");
}
