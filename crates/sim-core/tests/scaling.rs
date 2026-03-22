use std::time::Instant;

use sim_core::barnes_hut::BarnesHut;
use sim_core::gravity::GravitySolver;
use sim_core::scenario::Scenario;
use sim_core::scenarios::plummer_sphere::PlummerSphere;

/// Verify that Barnes-Hut scaling is O(N log N) by measuring wall-clock
/// time at several N values and checking the log-log slope.
///
/// The slope of log(time) vs log(N) should be in [1.0, 1.5] for
/// O(N log N), compared to 2.0 for O(N²).
#[test]
fn barnes_hut_scaling_is_nlogn() {
    let ns = [1_000, 2_000, 5_000, 10_000, 20_000];
    let theta = 0.5;
    let softening = 0.05;

    let mut times = Vec::new();

    for &n in &ns {
        let scenario = PlummerSphere {
            n,
            seed: 42,
            ..Default::default()
        };
        let mut particles = scenario.generate();
        let solver = BarnesHut::new(softening, theta);

        // Warm up
        particles.clear_accelerations();
        solver.compute_accelerations(&mut particles);

        // Measure
        let iters = 3;
        let start = Instant::now();
        for _ in 0..iters {
            particles.clear_accelerations();
            solver.compute_accelerations(&mut particles);
        }
        let elapsed = start.elapsed().as_secs_f64() / iters as f64;
        times.push(elapsed);

        eprintln!("N={n:>6}  time={elapsed:.4e}s");
    }

    // Compute log-log slope using least squares on the last few points
    // (skip smallest N where overhead dominates)
    let n_vals: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let t_vals: Vec<f64> = times.iter().map(|t| t.ln()).collect();

    // Use points from N=2000 onward (indices 1..5)
    let start_idx = 1;
    let count = n_vals.len() - start_idx;
    let mean_n: f64 = n_vals[start_idx..].iter().sum::<f64>() / count as f64;
    let mean_t: f64 = t_vals[start_idx..].iter().sum::<f64>() / count as f64;

    let mut num = 0.0;
    let mut den = 0.0;
    for i in start_idx..n_vals.len() {
        let dn = n_vals[i] - mean_n;
        let dt = t_vals[i] - mean_t;
        num += dn * dt;
        den += dn * dn;
    }
    let slope = num / den;

    eprintln!("Log-log slope: {slope:.3} (expected 1.0-1.5 for N log N, 2.0 for N²)");

    // O(N log N) should have slope between 1.0 and 1.5
    // Allow some margin for measurement noise
    assert!(
        slope >= 0.8 && slope <= 1.8,
        "Scaling slope {slope:.3} is outside expected range [0.8, 1.8] for O(N log N)"
    );
}
