mod render;
mod summary;

use clap::Parser;
use sim_core::barnes_hut::BarnesHut;
use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver, NoGravity};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::scenario_builder::{self, ScenarioConfig};
use sim_core::snapshot::Snapshot;
use sim_core::spec::ScenarioSpec;
use sim_core::sph::solver::SphSolver;
use render::{RenderOptions, View};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nbody-headless")]
#[command(about = "Headless N-body simulation runner for benchmarks and artifact generation")]
struct Cli {
    /// Scenario to run
    #[arg(short, long, default_value = "plummer")]
    scenario: String,

    /// Run a JSON scenario spec instead of a named scenario (see specs/)
    #[arg(long, conflicts_with = "scenario")]
    spec: Option<String>,

    /// Number of particles (overrides scenario default; rescales a spec)
    #[arg(short = 'n', long)]
    particles: Option<usize>,

    /// Number of timesteps to run [default: 1000, or until the last
    /// --render-at time]
    #[arg(long)]
    steps: Option<u64>,

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

    /// Render PNG frames at these sim times, e.g. 0,25,50 (each at the
    /// nearest step). Writes a summary.json beside them.
    #[arg(long, value_delimiter = ',')]
    render_at: Vec<f64>,

    /// Directory for rendered frames
    #[arg(long, default_value = "artifacts/renders")]
    render_dir: String,

    /// Frame width and height in pixels
    #[arg(long, default_value_t = 800)]
    render_size: usize,

    /// Projection planes, comma-separated: xy (face-on to the z axis), xz, yz
    #[arg(long, value_delimiter = ',', default_value = "xy")]
    render_view: Vec<String>,

    /// Half-width of the field of view [default: the spec's view_radius, or
    /// a radius holding 90% of the visible particles]
    #[arg(long)]
    render_radius: Option<f64>,

    /// Show dark matter (faint) in rendered frames
    #[arg(long)]
    render_dm: bool,

    /// Expose each frame on its own instead of holding the first frame's
    /// exposure (shows faint late structure, but brightness is no longer
    /// comparable between frames)
    #[arg(long)]
    render_autoexpose: bool,

    /// Write a JSON run summary here [default with --render-at:
    /// <render-dir>/summary.json]
    #[arg(long)]
    summary_json: Option<String>,
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
    let mut render_times = cli.render_at.clone();
    if let Some(t) = render_times.iter().find(|t| !t.is_finite() || **t < 0.0) {
        eprintln!("Error: --render-at times must be non-negative, got {t}");
        std::process::exit(1);
    }
    render_times.sort_by(f64::total_cmp);
    render_times.dedup();
    let views: Vec<View> = cli
        .render_view
        .iter()
        .map(|v| View::parse(v))
        .collect::<Result<_, _>>()
        .unwrap_or_else(|e| {
            eprintln!("Error: --render-view: {e}");
            std::process::exit(1);
        });
    if !(16..=8192).contains(&cli.render_size) {
        eprintln!("Error: --render-size must be in [16, 8192], got {}", cli.render_size);
        std::process::exit(1);
    }
    if let Some(r) = cli.render_radius
        && !(r.is_finite() && r > 0.0)
    {
        eprintln!("Error: --render-radius must be positive, got {r}");
        std::process::exit(1);
    }
    let rendering = !render_times.is_empty();
    let summary_path: Option<PathBuf> = cli
        .summary_json
        .as_ref()
        .map(PathBuf::from)
        .or_else(|| rendering.then(|| Path::new(&cli.render_dir).join("summary.json")));

    // Build scenario. Keep the spec as built (after any rescale): its
    // components map to particle index ranges for tracking and rendering.
    let mut built_spec: Option<ScenarioSpec> = None;
    let built = match &cli.spec {
        Some(path) => fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {path}: {e}"))
            .and_then(|json| ScenarioSpec::from_json(&json))
            .and_then(|spec| {
                let spec = match cli.particles {
                    Some(n) => spec.with_particle_budget(n),
                    None => spec,
                };
                let built = scenario_builder::build_spec(&spec, None, cli.dt, cli.softening);
                built_spec = Some(spec);
                built
            }),
        None => {
            let config = ScenarioConfig {
                particles: cli.particles,
                seed: cli.seed,
                eccentricity: cli.eccentricity,
            };
            scenario_builder::build(&cli.scenario, &config, cli.dt, cli.softening)
        }
    }
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let mut particles = built.particles;
    let dt = built.dt;
    let softening = built.softening;
    let scenario_name = built.name;
    let use_sph = particles.has_gas();

    let pure_hydro = cli.spec.is_none() && scenario_builder::is_pure_hydro(&cli.scenario);
    let gravity: Box<dyn GravitySolver> = if pure_hydro {
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
    match (cli.steps, render_times.last()) {
        (Some(steps), _) => eprintln!("Steps:      {steps}"),
        (None, Some(t)) => eprintln!("Run until:  t = {t}"),
        (None, None) => eprintln!("Steps:      1000"),
    }
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
    let use_full_diag = cli.diag_interval > 0 || cli.diag_csv.is_some() || summary_path.is_some();
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

    // Component tracking, frames, and summary
    let mut tracked = match &built_spec {
        Some(spec) => summary::from_spec(&particles, spec),
        None => summary::whole(&particles, &scenario_name),
    };
    let mut encounters = summary::EncounterLog::new(&tracked);
    let track = summary_path.is_some();
    let markers = summary::point_masses(&particles, built_spec.as_ref());
    let render_opts = RenderOptions {
        size: cli.render_size,
        radius: cli
            .render_radius
            .or_else(|| built_spec.as_ref().map(ScenarioSpec::view_radius))
            .unwrap_or_else(|| visible_radius(&particles, 0.9) * 1.2),
        show_dark_matter: cli.render_dm,
    };
    if rendering {
        fs::create_dir_all(&cli.render_dir).expect("Failed to create render directory");
        eprintln!("Rendering:  t = {render_times:?} -> {}/", cli.render_dir);
    }
    let mut exposure: Vec<Option<f32>> = vec![None; views.len()];
    let mut frames: Vec<summary::Frame> = Vec::new();
    let mut next_render = 0;
    let mut max_drift: f64 = 0.0;
    let mut note_drift = |d: &diagnostics::Diagnostics| {
        let drift = ((d.total_energy - initial_diag.total_energy) / initial_diag.total_energy.abs()).abs();
        if drift.is_finite() {
            max_drift = max_drift.max(drift);
        }
    };
    let mut capture = |particles: &sim_core::particle::Particles,
                       tracked: &[summary::Tracked],
                       time: f64,
                       step: u64| {
        let diag = diagnostics::compute(particles, softening, time, step);
        let mut images = Vec::new();
        let mut exposures = Vec::new();
        for (v, view) in views.iter().enumerate() {
            let light = render::splat(particles, &markers, *view, &render_opts);
            let exp = if cli.render_autoexpose {
                render::auto_exposure(&light)
            } else {
                *exposure[v].get_or_insert_with(|| render::auto_exposure(&light))
            };
            exposures.push(exp);
            let mut img = render::tone_map(&light, render_opts.size, exp);
            render::draw_markers(&mut img, particles, &markers, *view, render_opts.radius);
            let path = Path::new(&cli.render_dir).join(format!("t{time:07.2}_{}.png", view.name()));
            img.write_png(&path).unwrap_or_else(|e| {
                eprintln!("Failed to write frame: {e}");
                std::process::exit(1);
            });
            images.push(path.display().to_string());
        }
        eprintln!("Frame:      t={time:.4}  step={step}  {}", images.join(" "));
        let frame = summary::Frame {
            time,
            step,
            images,
            exposure: exposures,
            energy: summary::Energetics::new(&diag, initial_diag.total_energy),
            components: tracked.iter().enumerate().map(|(i, t)| t.state(particles, i)).collect(),
        };
        (frame, diag)
    };
    if rendering && render_times[0] <= 0.5 * dt {
        let (frame, _) = capture(&particles, &tracked, 0.0, 0);
        frames.push(frame);
        next_render = 1;
    }

    // Main simulation loop
    let wall_start = Instant::now();
    let mut sim_time = 0.0;
    let mut current_dt = dt;
    let max_steps = cli.steps.unwrap_or(if rendering { u64::MAX } else { 1000 });
    let mut steps_run = 0;

    for step in 1..=max_steps {
        if cli.steps.is_none() && rendering && next_render == render_times.len() {
            break;
        }
        let step_dt = current_dt;
        if let Some(sph) = &mut sph_solver {
            let dt_next = sim_core::sph::solver::step_with_sph(
                &mut particles, gravity.as_ref(), sph, current_dt,
            );
            current_dt = dt_next.min(current_dt * 1.5);
        } else {
            integrator.step(&mut particles, gravity.as_ref(), current_dt);
        }
        sim_time += step_dt;
        steps_run = step;

        if track {
            for t in &mut tracked {
                t.update(&particles);
            }
            encounters.record(&tracked, sim_time);
        }

        // Frames, at the step nearest each requested time
        while next_render < render_times.len() && sim_time >= render_times[next_render] - 0.5 * current_dt {
            let (frame, diag) = capture(&particles, &tracked, sim_time, step);
            note_drift(&diag);
            frames.push(frame);
            next_render += 1;
        }

        // Diagnostics
        if cli.diag_interval > 0 && step % cli.diag_interval == 0 {
            let diag = diagnostics::compute(&particles, softening, sim_time, step);
            note_drift(&diag);
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
        diagnostics::compute(&particles, softening, sim_time, steps_run)
    } else {
        diagnostics::compute_fast(&particles, sim_time, steps_run)
    };
    let de = (final_diag.total_energy - initial_diag.total_energy) / initial_diag.total_energy.abs();

    eprintln!();
    eprintln!("=== Final ===");
    eprintln!("Sim time:   {sim_time:.6e}");
    eprintln!("Wall time:  {:.3}s", wall_elapsed.as_secs_f64());
    eprintln!("Steps/sec:  {:.0}", steps_run as f64 / wall_elapsed.as_secs_f64());
    eprintln!("Energy:     {:.10e}  (dE/E = {:+.6e})", final_diag.total_energy, de);
    eprintln!("Momentum:   {:.6e}", final_diag.momentum_magnitude());
    eprintln!("Ang. Mom:   {:.6e}", final_diag.angular_momentum_magnitude());
    eprintln!("COM drift:  {:.6e}", final_diag.com_drift());
    eprintln!("Virial:     {:.4}", final_diag.virial_ratio);

    // Only write final row if not already written by the periodic output
    let already_written = cli.diag_interval > 0 && steps_run % cli.diag_interval == 0;
    if !already_written
        && let Some(ref mut w) = diag_writer
    {
        use std::io::Write;
        writeln!(w, "{}", final_diag.to_csv_row()).unwrap();
    }

    if next_render < render_times.len() {
        eprintln!(
            "Warning:    run ended at t={sim_time:.4}; frames at t >= {} were not rendered",
            render_times[next_render]
        );
    }

    if let Some(path) = summary_path {
        note_drift(&final_diag);
        let l0 = initial_diag.angular_momentum_magnitude();
        let dl = summary::dist(final_diag.angular_momentum, initial_diag.angular_momentum);
        let report = summary::Summary {
            scenario: scenario_name,
            spec: cli.spec.clone(),
            particles: particles.count,
            particle_types: summary::TypeCounts::of(&particles),
            algorithm: cli.algorithm.clone(),
            theta: (cli.algorithm == "barnes-hut").then_some(cli.theta),
            dt,
            softening,
            steps: steps_run,
            sim_time,
            wall_seconds: wall_elapsed.as_secs_f64(),
            initial: summary::Energetics::new(&initial_diag, initial_diag.total_energy),
            final_state: summary::Energetics::new(&final_diag, initial_diag.total_energy),
            max_abs_energy_drift: max_drift,
            angular_momentum_drift: if l0 > 0.0 { dl / l0 } else { dl },
            components: tracked.iter().enumerate().map(|(i, t)| t.state(&particles, i)).collect(),
            encounters: encounters.encounters,
            render: rendering.then(|| summary::RenderInfo {
                size: render_opts.size,
                radius: render_opts.radius,
                views: views.iter().map(|v| v.name().to_string()).collect(),
                dark_matter_shown: render_opts.show_dark_matter,
                exposure: if cli.render_autoexpose { "per_frame" } else { "fixed" }.to_string(),
            }),
            frames,
        };
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).expect("Failed to create summary directory");
        }
        let json = serde_json::to_string_pretty(&report).expect("summary serializes");
        fs::write(&path, json + "\n").unwrap_or_else(|e| {
            eprintln!("Failed to write {}: {e}", path.display());
            std::process::exit(1);
        });
        eprintln!("Summary:    {}", path.display());
    }
}

/// Radius around the origin (the center of mass) holding `fraction` of the
/// visible (non-dark-matter) particles.
fn visible_radius(p: &sim_core::particle::Particles, fraction: f64) -> f64 {
    let dm = sim_core::particle::ParticleType::DarkMatter as u8;
    let mut r: Vec<f64> = (0..p.count)
        .filter(|&i| p.particle_type[i] != dm)
        .map(|i| (p.x[i] * p.x[i] + p.y[i] * p.y[i] + p.z[i] * p.z[i]).sqrt())
        .collect();
    if r.is_empty() {
        return 1.0;
    }
    let k = ((r.len() - 1) as f64 * fraction) as usize;
    let (_, v, _) = r.select_nth_unstable_by(k, f64::total_cmp);
    if *v > 0.0 { *v } else { 1.0 }
}
