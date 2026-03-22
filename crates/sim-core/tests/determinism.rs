//! Verify that simulations are deterministic: the same initial conditions and
//! solver produce bit-identical results across runs. This is the foundation for
//! reproducible science — and it means the WASM sequential path (same algorithm,
//! same accumulation order) also matches native sequential runs.

use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::scenario::Scenario;
use sim_core::scenarios::plummer_sphere::PlummerSphere;

/// Run a simulation for the given number of steps and return final positions.
fn run_sim(
    mut particles: sim_core::particle::Particles,
    gravity: &dyn GravitySolver,
    steps: usize,
    dt: f64,
) -> Vec<[f64; 3]> {
    let integrator = LeapfrogKDK;
    gravity.compute_accelerations(&mut particles);
    for _ in 0..steps {
        integrator.step(&mut particles, gravity, dt);
    }
    (0..particles.count)
        .map(|i| [particles.x[i], particles.y[i], particles.z[i]])
        .collect()
}

#[test]
fn brute_force_deterministic() {
    let scenario = PlummerSphere {
        n: 200,
        seed: 42,
        ..Default::default()
    };
    let softening = 0.05;
    let gravity = BruteForce::new(softening);
    let dt = 0.001;
    let steps = 50;

    let pos1 = run_sim(scenario.generate(), &gravity, steps, dt);
    let pos2 = run_sim(scenario.generate(), &gravity, steps, dt);

    for (i, (a, b)) in pos1.iter().zip(pos2.iter()).enumerate() {
        assert_eq!(a, b, "Particle {i} diverged between runs");
    }
}

#[test]
fn barnes_hut_deterministic() {
    use sim_core::barnes_hut::BarnesHut;

    let scenario = PlummerSphere {
        n: 200,
        seed: 42,
        ..Default::default()
    };
    let softening = 0.05;
    let gravity = BarnesHut::new(softening, 0.5);
    let dt = 0.001;
    let steps = 50;

    let pos1 = run_sim(scenario.generate(), &gravity, steps, dt);
    let pos2 = run_sim(scenario.generate(), &gravity, steps, dt);

    for (i, (a, b)) in pos1.iter().zip(pos2.iter()).enumerate() {
        assert_eq!(a, b, "Particle {i} diverged between runs");
    }
}
