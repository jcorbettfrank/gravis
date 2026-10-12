use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sim_core::barnes_hut::BarnesHut;
use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::scenario::Scenario;
use sim_core::scenarios::plummer_sphere::PlummerSphere;

fn bench_brute_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("brute_force");

    for &n in &[100, 500, 1000, 2000, 5000, 10_000] {
        let scenario = PlummerSphere {
            n,
            ..Default::default()
        };
        let mut particles = scenario.generate();
        let solver = BruteForce::new(scenario.suggested_softening());

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                particles.clear_accelerations();
                solver.compute_accelerations(&mut particles);
            });
        });
    }

    group.finish();
}

fn bench_barnes_hut(c: &mut Criterion) {
    let mut group = c.benchmark_group("barnes_hut");

    for &n in &[1_000, 5_000, 10_000, 50_000, 100_000] {
        let scenario = PlummerSphere {
            n,
            ..Default::default()
        };
        let mut particles = scenario.generate();
        let solver = BarnesHut::new(scenario.suggested_softening(), 0.5);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                particles.clear_accelerations();
                solver.compute_accelerations(&mut particles);
            });
        });
    }

    group.finish();
}

/// Potential energy for the energy readout: exact pair sum vs tree estimate.
fn bench_potential_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("potential_energy");
    group.sample_size(10);

    for &n in &[5_000, 20_000, 50_000] {
        let scenario = PlummerSphere {
            n,
            ..Default::default()
        };
        let particles = scenario.generate();
        let eps = scenario.suggested_softening();
        group.bench_with_input(BenchmarkId::new("exact", n), &n, |b, _| {
            b.iter(|| diagnostics::compute(&particles, eps, 0.0, 0));
        });
    }

    for &n in &[20_000, 50_000, 100_000] {
        let scenario = PlummerSphere {
            n,
            ..Default::default()
        };
        let particles = scenario.generate();
        let eps = scenario.suggested_softening();
        group.bench_with_input(BenchmarkId::new("tree_0.3", n), &n, |b, _| {
            b.iter(|| diagnostics::compute_tree(&particles, eps, 0.3, 0.0, 0));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_brute_force, bench_barnes_hut, bench_potential_energy);
criterion_main!(benches);
