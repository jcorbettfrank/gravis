//! Disk heating of an isolated galaxy as a function of the star particle
//! fraction (ADR 011).
//!
//! Evolves one default `GalaxyModel` with Barnes-Hut and prints, every
//! `--every` time units, the disk's thickness and velocity dispersions in
//! the annulus 1 < R < 3 (around the disk scale radius), plus the virial
//! ratio and energy drift. Heavy halo particles scatter disk stars (two-body
//! heating), so the disk thickens faster as halo particles get heavier.
//!
//! ```text
//! cargo run -p sim-core --release --example disk_heating -- \
//!     --particles 20000 --star-fraction 0.5 --time 50
//! ```
//!
//! `--star-fraction equal` uses equal-mass particles (the split by mass).

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sim_core::barnes_hut::BarnesHut;
use sim_core::diagnostics;
use sim_core::gravity::GravitySolver;
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::particle::{ParticleType, Particles};
use sim_core::scenarios::galaxy_collision::GalaxyModel;

struct Args {
    particles: usize,
    star_fraction: Option<f64>,
    time: f64,
    every: f64,
    theta: f64,
    seed: u64,
}

fn parse_args() -> Args {
    let mut args = Args {
        particles: 20_000,
        star_fraction: None,
        time: 50.0,
        every: 5.0,
        theta: 0.5,
        seed: 42,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    for pair in argv.chunks(2) {
        let [flag, value] = pair else {
            panic!("expected --flag value pairs, got {argv:?}");
        };
        let num = || value.parse::<f64>().unwrap_or_else(|e| panic!("{flag} {value}: {e}"));
        match flag.as_str() {
            "--particles" => args.particles = num() as usize,
            "--star-fraction" => {
                args.star_fraction = if value == "equal" { None } else { Some(num()) }
            }
            "--time" => args.time = num(),
            "--every" => args.every = num(),
            "--theta" => args.theta = num(),
            "--seed" => args.seed = num() as u64,
            _ => panic!("unknown flag {flag}"),
        }
    }
    args
}

/// Disk-star statistics in the annulus 1 < R < 3, measured around the disk's
/// own center of mass: (RMS z, σ_z, σ_R, number of stars in the annulus).
fn disk_stats(p: &Particles) -> (f64, f64, f64, usize) {
    let disk: Vec<usize> = (0..p.count)
        .filter(|&i| p.particle_type[i] == ParticleType::DiskStar as u8)
        .collect();
    let m: f64 = disk.iter().map(|&i| p.mass[i]).sum();
    let mut c = [0.0; 6];
    for &i in &disk {
        let w = p.mass[i] / m;
        for (k, v) in [p.x[i], p.y[i], p.z[i], p.vx[i], p.vy[i], p.vz[i]].iter().enumerate() {
            c[k] += w * v;
        }
    }

    let (mut z2, mut vz2, mut vr2, mut count) = (0.0, 0.0, 0.0, 0usize);
    for &i in &disk {
        let (x, y, z) = (p.x[i] - c[0], p.y[i] - c[1], p.z[i] - c[2]);
        let (vx, vy, vz) = (p.vx[i] - c[3], p.vy[i] - c[4], p.vz[i] - c[5]);
        let r = x.hypot(y);
        if !(1.0..3.0).contains(&r) {
            continue;
        }
        z2 += z * z;
        vz2 += vz * vz;
        let vr = (x * vx + y * vy) / r;
        vr2 += vr * vr;
        count += 1;
    }
    let n = count.max(1) as f64;
    ((z2 / n).sqrt(), (vz2 / n).sqrt(), (vr2 / n).sqrt(), count)
}

fn main() {
    let args = parse_args();
    let model = GalaxyModel { star_particle_fraction: args.star_fraction, ..Default::default() };
    let (n_disk, n_bulge, n_halo) = model.particle_split(args.particles);

    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let mut p = Particles::new(args.particles);
    model.generate_into(&mut p, &mut rng, args.particles);
    p.shift_to_com_frame();

    let m_star = model.baryon_mass * model.disk_fraction / n_disk as f64;
    let m_dm = model.baryon_mass * model.halo_mass_ratio / n_halo as f64;
    let dt = model.suggested_dt();
    let eps = model.suggested_softening();
    let gravity = BarnesHut::new(eps, args.theta);

    let label = args.star_fraction.map_or("equal".to_string(), |f| f.to_string());
    eprintln!(
        "N={} star_fraction={label} disk/bulge/halo={n_disk}/{n_bulge}/{n_halo} \
         m_dm/m_star={:.2} dt={dt:.4e} eps={eps}",
        args.particles,
        m_dm / m_star
    );

    p.clear_accelerations();
    gravity.compute_accelerations(&mut p);
    let e0 = diagnostics::compute(&p, eps, 0.0, 0).total_energy;

    println!("star_fraction,particles,m_ratio,t,z_rms,sigma_z,sigma_r,n_annulus,virial,dE_over_E");
    let steps_per_sample = (args.every / dt).round().max(1.0) as u64;
    let samples = (args.time / args.every).round() as u64;
    let integrator = LeapfrogKDK;
    let mut step = 0u64;
    for sample in 0..=samples {
        if sample > 0 {
            for _ in 0..steps_per_sample {
                integrator.step(&mut p, &gravity, dt);
                step += 1;
            }
        }
        let t = step as f64 * dt;
        let d = diagnostics::compute(&p, eps, t, step);
        let (z_rms, sigma_z, sigma_r, n_ann) = disk_stats(&p);
        println!(
            "{label},{},{:.3},{t:.2},{z_rms:.5},{sigma_z:.5},{sigma_r:.5},{n_ann},{:.4},{:+.3e}",
            args.particles,
            m_dm / m_star,
            d.virial_ratio,
            (d.total_energy - e0) / e0.abs()
        );
    }
}
