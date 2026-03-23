use clap::Parser;
use sim_core::barnes_hut::BarnesHut;
use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver, NoGravity};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::scenario::Scenario;
use sim_core::scenarios::cold_collapse::ColdCollapse;
use sim_core::scenarios::evrard_collapse::EvrardCollapse;
use sim_core::scenarios::galaxy_collision::GalaxyCollision;
use sim_core::scenarios::kelvin_helmholtz::KelvinHelmholtz;
use sim_core::scenarios::plummer_sphere::PlummerSphere;
use sim_core::scenarios::sedov_blast::SedovBlast;
use sim_core::scenarios::sod_shock::SodShockTube;
use sim_core::scenarios::two_body::TwoBody;
use sim_core::snapshot::Snapshot;
use sim_core::sph::solver::SphSolver;
use std::fs::{self, File};
use std::io::BufWriter;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nbody-headless")]
#[command(about = "Headless N-body simulation runner for benchmarks and artifact generation")]
struct Cli {
    /// Scenario to run
    #[arg(short, long, default_value = "plummer")]
    scenario: String,

    /// Number of particles (overrides scenario default)
    #[arg(short = 'n', long)]
    particles: Option<usize>,

    /// Number of timesteps to run
    #[arg(long, default_value_t = 1000)]
    steps: u64,

    /// Timestep size (overrides scenario suggestion)
    #[arg(long)]
    dt: Option<f64>,

    /// Softening length (overrides scenario suggestion)
    #[arg(long)]
    softening: Option<f64>,

    /// Write snapshots every N steps (0 = disabled)
    #[arg(long, default_value_t = 0)]
    snapshot_interval: u64,

    /// Directory for snapshot output
    #[arg(long, default_value = "artifacts/snapshots")]
    snapshot_dir: String,

    /// Print diagnostics every N steps (0 = disabled)
    #[arg(long, default_value_t = 100)]
    diag_interval: u64,

    /// Write diagnostics CSV to this file
    #[arg(long)]
    diag_csv: Option<String>,

    /// RNG seed for scenario generation
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Eccentricity for two-body scenario (must be in [0, 1))
    #[arg(long, default_value_t = 0.5)]
    eccentricity: f64,

    /// Force calculation algorithm
    #[arg(long, default_value = "brute-force", value_parser = ["brute-force", "barnes-hut"])]
    algorithm: String,

    /// Barnes-Hut opening angle (lower = more accurate, higher = faster)
    #[arg(long, default_value_t = 0.5)]
    theta: f64,
}

fn main() {
    let cli = Cli::parse();

    // Validate CLI inputs
    if let Some(n) = cli.particles
        && n == 0
    {
        eprintln!("Error: --particles must be > 0");
        std::process::exit(1);
    }
    if cli.eccentricity < 0.0 || cli.eccentricity >= 1.0 {
        eprintln!("Error: --eccentricity must be in [0, 1), got {}", cli.eccentricity);
        std::process::exit(1);
    }
    if let Some(dt) = cli.dt
        && dt <= 0.0
    {
        eprintln!("Error: --dt must be positive, got {dt}");
        std::process::exit(1);
    }
    if let Some(s) = cli.softening
        && s < 0.0
    {
        eprintln!("Error: --softening must be non-negative, got {s}");
        std::process::exit(1);
    }

    // Build scenario
    let (mut particles, dt, softening, scenario_name) = match cli.scenario.as_str() {
        "plummer" => {
            let n = cli.particles.unwrap_or(1000);
            let scenario = PlummerSphere {
                n,
                seed: cli.seed,
                ..Default::default()
            };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = cli.softening.unwrap_or_else(|| scenario.suggested_softening());
            (p, dt, soft, scenario.name().to_string())
        }
        "two-body" | "kepler" => {
            let scenario = TwoBody {
                eccentricity: cli.eccentricity,
                ..Default::default()
            };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = cli.softening.unwrap_or_else(|| scenario.suggested_softening());
            (p, dt, soft, scenario.name().to_string())
        }
        "cold-collapse" => {
            let n = cli.particles.unwrap_or(5000);
            let scenario = ColdCollapse {
                n,
                seed: cli.seed,
                ..Default::default()
            };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = cli.softening.unwrap_or_else(|| scenario.suggested_softening());
            (p, dt, soft, scenario.name().to_string())
        }
        "galaxy-collision" => {
            let n = cli.particles.unwrap_or(10000);
            let scenario = GalaxyCollision {
                n_per_galaxy: n / 2,
                seed: cli.seed,
                ..Default::default()
            };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = cli.softening.unwrap_or_else(|| scenario.suggested_softening());
            (p, dt, soft, scenario.name().to_string())
        }
        // SPH scenarios
        "sod-shock" => {
            let scenario = SodShockTube::default();
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = scenario.suggested_softening();
            (p, dt, soft, scenario.name().to_string())
        }
        "sedov-blast" => {
            let n = cli.particles.unwrap_or(5000);
            let scenario = SedovBlast { n_particles: n, seed: cli.seed, ..Default::default() };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = scenario.suggested_softening();
            (p, dt, soft, scenario.name().to_string())
        }
        "evrard-collapse" => {
            let n = cli.particles.unwrap_or(5000);
            let scenario = EvrardCollapse { n_particles: n, seed: cli.seed, ..Default::default() };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = scenario.suggested_softening();
            (p, dt, soft, scenario.name().to_string())
        }
        "kelvin-helmholtz" | "kh" => {
            let scenario = KelvinHelmholtz::default();
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = scenario.suggested_softening();
            (p, dt, soft, scenario.name().to_string())
        }
        other => {
            eprintln!("Unknown scenario: {other}");
            eprintln!("Available: plummer, two-body, cold-collapse, galaxy-collision, sod-shock, sedov-blast, evrard-collapse, kelvin-helmholtz");
            std::process::exit(1);
        }
    };

    // Detect if scenario uses SPH (has gas particles)
    let use_sph = particles.has_gas();

    let gravity: Box<dyn GravitySolver> = if use_sph && !["evrard-collapse"].contains(&cli.scenario.as_str()) {
        Box::new(NoGravity)
    } else {
        match cli.algorithm.as_str() {
            "barnes-hut" => Box::new(BarnesHut::new(softening, cli.theta)),
            _ => Box::new(BruteForce::new(softening)),
        }
    };
    let integrator = LeapfrogKDK;
    let mut sph_solver = if use_sph { Some(SphSolver::new()) } else { None };

    eprintln!("Scenario:   {scenario_name}");
    eprintln!("Particles:  {}", particles.count);
    eprintln!("Algorithm:  {}", cli.algorithm);
    if cli.algorithm == "barnes-hut" {
        eprintln!("Theta:      {}", cli.theta);
    }
    eprintln!("Steps:      {}", cli.steps);
    eprintln!("dt:         {dt:.6e}");
    eprintln!("Softening:  {softening:.6e}");
    eprintln!();

    // Set up diagnostics CSV output
    let mut diag_writer: Option<BufWriter<File>> = cli.diag_csv.as_ref().map(|path| {
        let file = File::create(path).expect("Failed to create diagnostics CSV");
        let mut w = BufWriter::new(file);
        use std::io::Write;
        writeln!(w, "{}", diagnostics::Diagnostics::csv_header()).unwrap();
        w
    });

    // Set up snapshot directory
    if cli.snapshot_interval > 0 {
        fs::create_dir_all(&cli.snapshot_dir).expect("Failed to create snapshot directory");
    }

    // Initialize accelerations before first step
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);
    if let Some(sph) = &mut sph_solver {
        let _ = sph.compute(&mut particles);
    }

    // Record initial diagnostics
    // Use fast (O(N)) diagnostics when the O(N²) potential energy calculation
    // is not needed (no diagnostics output requested).
    let use_full_diag = cli.diag_interval > 0 || cli.diag_csv.is_some();
    let initial_diag = if use_full_diag {
        diagnostics::compute(&particles, softening, 0.0, 0)
    } else {
        diagnostics::compute_fast(&particles, 0.0, 0)
    };
    eprintln!(
        "Initial:    E={:.10e}  2K/|U|={:.4}",
        initial_diag.total_energy, initial_diag.virial_ratio
    );

    if let Some(ref mut w) = diag_writer {
        use std::io::Write;
        writeln!(w, "{}", initial_diag.to_csv_row()).unwrap();
    }

    // Main simulation loop
    let wall_start = Instant::now();
    let mut sim_time = 0.0;
    let mut current_dt = dt;

    for step in 1..=cli.steps {
        if let Some(sph) = &mut sph_solver {
            let dt_next = sim_core::sph::solver::step_with_sph(
                &mut particles, gravity.as_ref(), sph, current_dt,
            );
            current_dt = dt_next.min(current_dt * 1.5);
        } else {
            integrator.step(&mut particles, gravity.as_ref(), current_dt);
        }
        sim_time += current_dt;

        // Diagnostics
        if cli.diag_interval > 0 && step % cli.diag_interval == 0 {
            let diag = diagnostics::compute(&particles, softening, sim_time, step);
            let de = (diag.total_energy - initial_diag.total_energy) / initial_diag.total_energy.abs();

            eprintln!(
                "Step {:>8}  t={:.4e}  E={:.10e}  dE/E={:+.4e}  2K/|U|={:.4}",
                step, sim_time, diag.total_energy, de, diag.virial_ratio
            );

            if let Some(ref mut w) = diag_writer {
                use std::io::Write;
                writeln!(w, "{}", diag.to_csv_row()).unwrap();
            }
        }

        // Snapshots
        if cli.snapshot_interval > 0 && step % cli.snapshot_interval == 0 {
            let snap = Snapshot::capture(&particles, sim_time, step, softening, current_dt);
            let path = format!("{}/snap_{:08}.bin", cli.snapshot_dir, step);
            let mut file = BufWriter::new(File::create(&path).expect("Failed to create snapshot"));
            snap.write_to(&mut file).expect("Failed to write snapshot");
        }
    }

    let wall_elapsed = wall_start.elapsed();

    // Final diagnostics
    let final_diag = if use_full_diag {
        diagnostics::compute(&particles, softening, sim_time, cli.steps)
    } else {
        diagnostics::compute_fast(&particles, sim_time, cli.steps)
    };
    let de = (final_diag.total_energy - initial_diag.total_energy) / initial_diag.total_energy.abs();

    eprintln!();
    eprintln!("=== Final ===");
    eprintln!("Sim time:   {sim_time:.6e}");
    eprintln!("Wall time:  {:.3}s", wall_elapsed.as_secs_f64());
    eprintln!("Steps/sec:  {:.0}", cli.steps as f64 / wall_elapsed.as_secs_f64());
    eprintln!("Energy:     {:.10e}  (dE/E = {:+.6e})", final_diag.total_energy, de);
    eprintln!("Momentum:   {:.6e}", final_diag.momentum_magnitude());
    eprintln!("Ang. Mom:   {:.6e}", final_diag.angular_momentum_magnitude());
    eprintln!("COM drift:  {:.6e}", final_diag.com_drift());
    eprintln!("Virial:     {:.4}", final_diag.virial_ratio);

    // Only write final row if not already written by the periodic output
    let already_written = cli.diag_interval > 0 && cli.steps % cli.diag_interval == 0;
    if !already_written
        && let Some(ref mut w) = diag_writer
    {
        use std::io::Write;
        writeln!(w, "{}", final_diag.to_csv_row()).unwrap();
    }
}
