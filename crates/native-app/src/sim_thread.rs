use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

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
use sim_core::sph::solver::SphSolver;

use crate::Cli;

/// Lightweight snapshot for rendering — positions already converted to f32.
pub struct RenderSnapshot {
    pub positions: Vec<[f32; 3]>,
    pub masses: Vec<f32>,
    pub particle_types: Arc<Vec<u8>>,
    /// Internal energy per particle (f32, for gas temperature coloring).
    pub internal_energies: Vec<f32>,
    pub center_of_mass: [f32; 3],
    pub sim_time: f64,
    pub step: u64,
    pub total_energy: f64,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub thermal_energy: f64,
    pub virial_ratio: f64,
    pub particle_count: usize,
}

pub enum SimCommand {
    Pause,
    Resume,
    SetSpeed(f64),
    Stop,
}

pub struct SimHandle {
    snapshot_rx: mpsc::Receiver<RenderSnapshot>,
    command_tx: mpsc::Sender<SimCommand>,
    thread: Option<thread::JoinHandle<()>>,
    paused: bool,
}

impl SimHandle {
    pub fn drain_latest(&self) -> Option<RenderSnapshot> {
        let mut latest = None;
        while let Ok(snap) = self.snapshot_rx.try_recv() {
            latest = Some(snap);
        }
        latest
    }

    pub fn set_speed(&self, speed: f64) {
        let _ = self.command_tx.send(SimCommand::SetSpeed(speed));
    }

    pub fn pause(&mut self) {
        if !self.paused {
            let _ = self.command_tx.send(SimCommand::Pause);
            self.paused = true;
        }
    }

    pub fn resume(&mut self) {
        if self.paused {
            let _ = self.command_tx.send(SimCommand::Resume);
            self.paused = false;
        }
    }

    pub fn stop(&mut self) {
        let _ = self.command_tx.send(SimCommand::Stop);
        if let Some(handle) = self.thread.take() {
            handle.join().ok();
        }
    }
}

impl Drop for SimHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn spawn(cli: &Cli) -> SimHandle {
    let (snap_tx, snap_rx) = mpsc::channel();
    let (cmd_tx, cmd_rx) = mpsc::channel();

    let cli = cli.clone();

    let handle = thread::spawn(move || {
        run_sim(cli, snap_tx, cmd_rx);
    });

    SimHandle {
        snapshot_rx: snap_rx,
        command_tx: cmd_tx,
        thread: Some(handle),
        paused: false,
    }
}

fn run_sim(cli: Cli, tx: mpsc::Sender<RenderSnapshot>, cmd_rx: mpsc::Receiver<SimCommand>) {
    // Build scenario
    let (mut particles, dt, softening, scenario_name, use_sph) = match cli.scenario.as_str() {
        "plummer" => {
            let n = cli.particles.unwrap_or(1000);
            let scenario = PlummerSphere { n, seed: cli.seed, ..Default::default() };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = cli.softening.unwrap_or_else(|| scenario.suggested_softening());
            (p, dt, soft, scenario.name().to_string(), false)
        }
        "two-body" | "kepler" => {
            let scenario = TwoBody { eccentricity: cli.eccentricity, ..Default::default() };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = cli.softening.unwrap_or_else(|| scenario.suggested_softening());
            (p, dt, soft, scenario.name().to_string(), false)
        }
        "cold-collapse" => {
            let n = cli.particles.unwrap_or(5000);
            let scenario = ColdCollapse { n, seed: cli.seed, ..Default::default() };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = cli.softening.unwrap_or_else(|| scenario.suggested_softening());
            (p, dt, soft, scenario.name().to_string(), false)
        }
        "galaxy-collision" => {
            let n = cli.particles.unwrap_or(10000);
            let scenario = GalaxyCollision { n_per_galaxy: n / 2, seed: cli.seed, ..Default::default() };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = cli.softening.unwrap_or_else(|| scenario.suggested_softening());
            (p, dt, soft, scenario.name().to_string(), false)
        }
        // SPH scenarios
        "sod-shock" => {
            let scenario = SodShockTube::default();
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = scenario.suggested_softening();
            (p, dt, soft, scenario.name().to_string(), true)
        }
        "sedov-blast" => {
            let n = cli.particles.unwrap_or(5000);
            let scenario = SedovBlast { n_particles: n, seed: cli.seed, ..Default::default() };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = scenario.suggested_softening();
            (p, dt, soft, scenario.name().to_string(), true)
        }
        "evrard-collapse" => {
            let n = cli.particles.unwrap_or(5000);
            let scenario = EvrardCollapse { n_particles: n, seed: cli.seed, ..Default::default() };
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = scenario.suggested_softening();
            (p, dt, soft, scenario.name().to_string(), true)
        }
        "kelvin-helmholtz" | "kh" => {
            let scenario = KelvinHelmholtz::default();
            let p = scenario.generate();
            let dt = cli.dt.unwrap_or_else(|| scenario.suggested_dt());
            let soft = scenario.suggested_softening();
            (p, dt, soft, scenario.name().to_string(), true)
        }
        other => {
            eprintln!("Unknown scenario: {other}. Available: plummer, two-body, cold-collapse, galaxy-collision, sod-shock, sedov-blast, evrard-collapse, kelvin-helmholtz");
            return;
        }
    };

    // Gravity solver: SPH-only scenarios (sod, sedov, kh) use NoGravity; evrard needs gravity
    let gravity: Box<dyn GravitySolver + Send> = if use_sph && cli.scenario != "evrard-collapse" {
        Box::new(NoGravity)
    } else {
        match cli.algorithm.as_str() {
            "barnes-hut" => Box::new(BarnesHut::new(softening, cli.theta)),
            _ => Box::new(BruteForce::new(softening)),
        }
    };
    let integrator = LeapfrogKDK;
    let mut sph_solver = if use_sph { Some(SphSolver::new()) } else { None };

    eprintln!(
        "Sim thread: {scenario_name}, N={}, algorithm={}, sph={}, dt={dt:.6e}, ε={softening:.6e}",
        particles.count, cli.algorithm, use_sph
    );

    // Initialize accelerations
    particles.clear_accelerations();
    gravity.compute_accelerations(&mut particles);
    if let Some(sph) = &mut sph_solver {
        let _ = sph.compute(&mut particles);
    }

    let mut sim_time = 0.0_f64;
    let mut step = 0_u64;
    let mut speed_multiplier = cli.speed;
    let mut paused = false;
    let mut current_dt = dt;

    // Particle types are invariant — share via Arc to avoid cloning per snapshot
    let particle_types = Arc::new(particles.particle_type.clone());

    // Send initial snapshot with diagnostics
    let initial_diag = diagnostics::compute(&particles, softening, sim_time, step);
    let _ = tx.send(build_render_snapshot(&particles, &initial_diag, &particle_types));

    let wall_start = Instant::now();
    let mut last_snap_send = Instant::now();
    let mut last_diag_time = Instant::now();
    let mut cached_diag = initial_diag;

    const SNAP_INTERVAL: Duration = Duration::from_millis(16);
    const DIAG_INTERVAL: Duration = Duration::from_secs(1);

    loop {
        // Handle commands
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                SimCommand::Pause => paused = true,
                SimCommand::Resume => paused = false,
                SimCommand::SetSpeed(s) => speed_multiplier = s,
                SimCommand::Stop => return,
            }
        }

        if paused {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Advance simulation
        let target_sim_time = wall_start.elapsed().as_secs_f64() * speed_multiplier;
        let mut steps_this_batch = 0u32;
        let batch_start = Instant::now();
        while sim_time < target_sim_time {
            if let Some(sph) = &mut sph_solver {
                let dt_next = sim_core::sph::solver::step_with_sph(
                    &mut particles, gravity.as_ref(), sph, current_dt,
                );
                current_dt = dt_next.min(current_dt * 1.5);
            } else {
                integrator.step(&mut particles, gravity.as_ref(), current_dt);
            }
            sim_time += current_dt;
            step += 1;
            steps_this_batch += 1;

            if steps_this_batch >= 1000 || batch_start.elapsed() >= SNAP_INTERVAL {
                break;
            }
        }

        if sim_time >= target_sim_time && steps_this_batch == 0 {
            thread::sleep(Duration::from_millis(1));
            continue;
        }

        if last_diag_time.elapsed() >= DIAG_INTERVAL {
            cached_diag = diagnostics::compute_fast(&particles, sim_time, step);
            last_diag_time = Instant::now();
        }

        if last_snap_send.elapsed() >= SNAP_INTERVAL {
            let snap = build_render_snapshot_with_diag(&particles, &cached_diag, sim_time, step, &particle_types);
            if tx.send(snap).is_err() {
                return;
            }
            last_snap_send = Instant::now();
        }
    }
}

fn build_render_snapshot(
    particles: &sim_core::particle::Particles,
    diag: &diagnostics::Diagnostics,
    particle_types: &Arc<Vec<u8>>,
) -> RenderSnapshot {
    build_render_snapshot_with_diag(particles, diag, diag.time, diag.step, particle_types)
}

fn build_render_snapshot_with_diag(
    particles: &sim_core::particle::Particles,
    diag: &diagnostics::Diagnostics,
    sim_time: f64,
    step: u64,
    particle_types: &Arc<Vec<u8>>,
) -> RenderSnapshot {
    let n = particles.count;
    let mut positions = Vec::with_capacity(n);
    let mut masses = Vec::with_capacity(n);
    let mut internal_energies = Vec::with_capacity(n);

    for i in 0..n {
        positions.push([
            particles.x[i] as f32,
            particles.y[i] as f32,
            particles.z[i] as f32,
        ]);
        masses.push(particles.mass[i] as f32);
        internal_energies.push(particles.internal_energy[i] as f32);
    }

    RenderSnapshot {
        positions,
        masses,
        particle_types: Arc::clone(particle_types),
        internal_energies,
        center_of_mass: [
            diag.center_of_mass[0] as f32,
            diag.center_of_mass[1] as f32,
            diag.center_of_mass[2] as f32,
        ],
        sim_time,
        step,
        total_energy: diag.total_energy,
        kinetic_energy: diag.kinetic_energy,
        potential_energy: diag.potential_energy,
        thermal_energy: diag.thermal_energy,
        virial_ratio: diag.virial_ratio,
        particle_count: n,
    }
}
