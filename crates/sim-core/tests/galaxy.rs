use sim_core::barnes_hut::BarnesHut;
use sim_core::diagnostics;
use sim_core::gravity::GravitySolver;
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::particle::ParticleType;
use sim_core::scenario::Scenario;
use sim_core::scenarios::cold_collapse::ColdCollapse;
use sim_core::scenarios::galaxy_collision::GalaxyCollision;

#[test]
fn cold_collapse_generates_correct_count() {
    let scenario = ColdCollapse {
        n: 500,
        ..Default::default()
    };
    let particles = scenario.generate();
    assert_eq!(particles.count, 500);
}

#[test]
fn cold_collapse_zero_velocity() {
    let scenario = ColdCollapse {
        n: 100,
        ..Default::default()
    };
    let particles = scenario.generate();
    for i in 0..particles.count {
        assert_eq!(particles.vx[i], 0.0);
        assert_eq!(particles.vy[i], 0.0);
        assert_eq!(particles.vz[i], 0.0);
    }
}

#[test]
fn cold_collapse_particles_within_sphere() {
    let radius = 2.0;
    let scenario = ColdCollapse {
        n: 1000,
        radius,
        ..Default::default()
    };
    let particles = scenario.generate();
    // After COM shift, particles may be slightly outside the original sphere
    // (by up to the COM offset, which is ~radius/sqrt(N)). Just verify they're
    // all within a reasonable margin.
    let margin = radius * 0.1; // allow 10% margin from COM shift
    for i in 0..particles.count {
        let r2 = particles.x[i].powi(2) + particles.y[i].powi(2) + particles.z[i].powi(2);
        assert!(
            r2 <= (radius + margin) * (radius + margin),
            "Particle {i} at r={} exceeds radius={radius} + margin",
            r2.sqrt()
        );
    }
}

#[test]
fn cold_collapse_energy_conservation() {
    let scenario = ColdCollapse {
        n: 500,
        ..Default::default()
    };
    let softening = scenario.suggested_softening();
    let dt = scenario.suggested_dt();
    let gravity = BarnesHut::new(softening, 0.5);
    let integrator = LeapfrogKDK;

    let mut particles = scenario.generate();
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let initial = diagnostics::compute(&particles, softening, 0.0, 0);
    let e0 = initial.total_energy;

    // Run for 100 steps
    for _ in 0..100 {
        integrator.step(&mut particles, &gravity, dt);
    }

    let final_diag = diagnostics::compute(&particles, softening, 100.0 * dt, 100);
    let e_final = final_diag.total_energy;
    let drift = ((e_final - e0) / e0).abs();
    assert!(
        drift < 0.05,
        "Energy drifted by {:.2}% (expected < 5%)",
        drift * 100.0
    );
}

#[test]
fn galaxy_generates_correct_total_count() {
    let scenario = GalaxyCollision {
        n_per_galaxy: 500,
        ..Default::default()
    };
    let particles = scenario.generate();
    assert_eq!(particles.count, 1000); // 2 galaxies × 500
}

#[test]
fn galaxy_has_all_particle_types() {
    let scenario = GalaxyCollision {
        n_per_galaxy: 500,
        ..Default::default()
    };
    let particles = scenario.generate();

    let n_disk = particles
        .particle_type
        .iter()
        .filter(|&&t| t == ParticleType::DiskStar as u8)
        .count();
    let n_bulge = particles
        .particle_type
        .iter()
        .filter(|&&t| t == ParticleType::BulgeStar as u8)
        .count();
    let n_halo = particles
        .particle_type
        .iter()
        .filter(|&&t| t == ParticleType::DarkMatter as u8)
        .count();

    assert!(n_disk > 0, "Should have disk particles");
    assert!(n_bulge > 0, "Should have bulge particles");
    assert!(n_halo > 0, "Should have halo particles");
    assert_eq!(n_disk + n_bulge + n_halo, particles.count);

    // Halo should be the dominant component by count (mass ratio = 10)
    assert!(
        n_halo > n_disk,
        "Halo ({n_halo}) should outnumber disk ({n_disk})"
    );
}

#[test]
fn galaxy_deterministic() {
    let scenario = GalaxyCollision {
        n_per_galaxy: 200,
        ..Default::default()
    };
    let p1 = scenario.generate();
    let p2 = scenario.generate();
    for i in 0..p1.count {
        assert_eq!(p1.x[i], p2.x[i]);
        assert_eq!(p1.vx[i], p2.vx[i]);
        assert_eq!(p1.particle_type[i], p2.particle_type[i]);
    }
}

/// Stability test: run a single isolated galaxy and check that
/// the virial ratio stays bounded (galaxy doesn't explode or collapse).
#[test]
fn single_galaxy_stability() {
    let scenario = GalaxyCollision {
        n_per_galaxy: 1000,
        // No collision — use a single galaxy by setting separation to 0
        separation: 0.0,
        impact_parameter: 0.0,
        approach_velocity: 0.0,
        ..Default::default()
    };

    // Generate both galaxies on top of each other (acts as one 2000-particle galaxy)
    // Alternative: could generate just one, but this tests the actual code path
    let softening = scenario.suggested_softening();
    let dt = scenario.suggested_dt();
    let gravity = BarnesHut::new(softening, 0.5);
    let integrator = LeapfrogKDK;

    let mut particles = GalaxyCollision {
        n_per_galaxy: 1000,
        separation: 0.0,
        impact_parameter: 0.0,
        approach_velocity: 0.0,
        ..Default::default()
    }
    .generate();

    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);

    let initial = diagnostics::compute(&particles, softening, 0.0, 0);
    let e0 = initial.total_energy;

    // Run for ~5 dynamical times (500 steps at dt ~ t_orb/500)
    for _ in 0..500 {
        integrator.step(&mut particles, &gravity, dt);
    }

    let final_diag = diagnostics::compute(&particles, softening, 500.0 * dt, 500);

    // Virial ratio should stay in reasonable range
    let vr = final_diag.virial_ratio;
    assert!(
        (0.5..2.0).contains(&vr),
        "Virial ratio {vr:.3} out of stable range [0.5, 2.0]"
    );

    // Energy should not drift excessively
    let drift = ((final_diag.total_energy - e0) / e0.abs()).abs();
    assert!(
        drift < 0.05,
        "Energy drifted by {:.2}% (expected < 5%)",
        drift * 100.0
    );
}
