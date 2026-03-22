use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sim_core::barnes_hut::BarnesHut;
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

criterion_group!(benches, bench_brute_force, bench_barnes_hut);
criterion_main!(benches);
