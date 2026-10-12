use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use sim_core::barnes_hut::BarnesHut;
use sim_core::diagnostics;
use sim_core::gravity::{BruteForce, GravitySolver, NoGravity};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::particle::Particles;
use sim_core::scenario_builder::{self, ScenarioConfig};
use sim_core::sph::solver::SphSolver;

use crate::Cli;

/// Lightweight snapshot for rendering — positions already converted to f32.
pub struct RenderSnapshot {
    pub positions: Vec<[f32; 3]>,
    pub particle_types: Arc<Vec<u8>>,
    /// Internal energy per particle (f32, for gas temperature coloring).
    pub internal_energies: Vec<f32>,
    pub center_of_mass: [f32; 3],
    pub sim_time: f64,
    pub step: u64,
    /// Latest energy measurement; `None` until the first one finishes.
    pub energy: Option<EnergyReadout>,
    pub particle_count: usize,
}

/// How the potential energy in an `EnergyReadout` was computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnergyMethod {
    /// Exact O(N²) pair sum.
    Exact,
    /// Barnes-Hut tree estimate with this opening angle.
    Tree(f64),
    /// Gravity is off (pure hydro), so U is not part of the energy.
    NoGravity,
}

/// One energy measurement, taken at `sim_time`. Snapshots between
/// measurements repeat the latest one.
#[derive(Clone, Copy, Debug)]
pub struct EnergyReadout {
    pub sim_time: f64,
    pub total: f64,
    pub kinetic: f64,
    pub potential: f64,
    #[allow(dead_code)] // will be shown in HUD
    pub thermal: f64,
    pub virial_ratio: f64,
    pub method: EnergyMethod,
}

/// Largest N for which the HUD energy uses the exact pair sum. Above it the
/// potential comes from a tree walk (ADR 012).
const EXACT_ENERGY_MAX_N: usize = 50_000;
/// Opening angle for the tree potential: ~1e-5 to 1e-4 relative error in E
/// (ADR 012).
const ENERGY_TREE_THETA: f64 = 0.3;
/// Energy is measured at least this far apart...
const MIN_ENERGY_INTERVAL: Duration = Duration::from_secs(1);
/// ...and the worker is busy at most ~1/ENERGY_COST_FACTOR of the time.
const ENERGY_COST_FACTOR: u32 = 10;

fn energy_method(n: usize, gravity_on: bool) -> EnergyMethod {
    if !gravity_on {
        EnergyMethod::NoGravity
    } else if n <= EXACT_ENERGY_MAX_N {
        EnergyMethod::Exact
    } else {
        EnergyMethod::Tree(ENERGY_TREE_THETA)
    }
}

fn measure_energy(
    particles: &Particles,
    softening: f64,
    method: EnergyMethod,
    sim_time: f64,
    step: u64,
) -> EnergyReadout {
    let d = match method {
        EnergyMethod::Exact => diagnostics::compute(particles, softening, sim_time, step),
        EnergyMethod::Tree(theta) => {
            diagnostics::compute_tree(particles, softening, theta, sim_time, step)
        }
        EnergyMethod::NoGravity => diagnostics::compute_fast(particles, sim_time, step),
    };
    EnergyReadout {
        sim_time,
        total: d.total_energy,
        kinetic: d.kinetic_energy,
        potential: d.potential_energy,
        thermal: d.thermal_energy,
        virial_ratio: d.virial_ratio,
        method,
    }
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
    let built = match &cli.spec {
        Some(spec) => scenario_builder::build_spec(spec, cli.particles, cli.dt, cli.softening),
        None => {
            let config = ScenarioConfig {
                particles: cli.particles,
                seed: cli.seed,
                eccentricity: cli.eccentricity,
            };
            scenario_builder::build(&cli.scenario, &config, cli.dt, cli.softening)
        }
    };
    let built = match built {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    let mut particles = built.particles;
    let dt = built.dt;
    let softening = built.softening;
    let scenario_name = built.name;
    let use_sph = particles.has_gas();

    let pure_hydro = cli.spec.is_none() && scenario_builder::is_pure_hydro(&cli.scenario);
    let gravity: Box<dyn GravitySolver + Send> = if pure_hydro {
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
    let mut held_for_screenshot = false;

    // Particle types are invariant — share via Arc to avoid cloning per snapshot
    let particle_types = Arc::new(particles.particle_type.clone());

    // Show the initial state right away; energy arrives from the worker
    let _ = tx.send(build_render_snapshot(&particles, sim_time, step, None, &particle_types));

    let mut energy_worker =
        EnergyWorker::spawn(softening, energy_method(particles.count, !pure_hydro));
    energy_worker.start(&particles, sim_time, step);
    let mut energy: Option<EnergyReadout> = None;
    let mut next_energy_at = Instant::now();
    let mut screenshot_energy_started = false;

    let wall_start = Instant::now();
    let mut last_snap_send = Instant::now();

    const SNAP_INTERVAL: Duration = Duration::from_millis(16);

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

        // Pick up a finished energy measurement; the next waits in proportion to its cost
        let mut new_energy = false;
        if let Some((readout, cost)) = energy_worker.poll() {
            energy = Some(readout);
            next_energy_at = Instant::now() + energy_interval(cost);
            new_energy = true;
        }

        if paused {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Advance simulation: paced by wall clock, or flat out to a screenshot time
        let target_sim_time = match cli.screenshot_time {
            Some(t) => t,
            None => wall_start.elapsed().as_secs_f64() * speed_multiplier,
        };
        let mut steps_this_batch = 0u32;
        let batch_start = Instant::now();
        while sim_time < target_sim_time {
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
            step += 1;
            steps_this_batch += 1;

            if steps_this_batch >= 1000 || batch_start.elapsed() >= SNAP_INTERVAL {
                break;
            }
        }

        if cli.screenshot_time.is_some() && sim_time >= target_sim_time {
            // Hold at the screenshot time: send its snapshot, measure its energy
            // once the worker is free, and resend when that lands
            if !held_for_screenshot || new_energy {
                let snap = build_render_snapshot(&particles, sim_time, step, energy, &particle_types);
                let _ = tx.send(snap);
                held_for_screenshot = true;
            }
            if !screenshot_energy_started && !energy_worker.busy {
                energy_worker.start(&particles, sim_time, step);
                screenshot_energy_started = true;
            }
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        if sim_time >= target_sim_time && steps_this_batch == 0 {
            thread::sleep(Duration::from_millis(1));
            continue;
        }

        if !energy_worker.busy && Instant::now() >= next_energy_at {
            energy_worker.start(&particles, sim_time, step);
        }

        if last_snap_send.elapsed() >= SNAP_INTERVAL {
            let snap = build_render_snapshot(&particles, sim_time, step, energy, &particle_types);
            if tx.send(snap).is_err() {
                return;
            }
            last_snap_send = Instant::now();
        }
    }
}

/// Measures energy on its own thread, from a copy of the particles, so a
/// slow measurement (seconds at large N) never stalls the sim or the frames.
struct EnergyWorker {
    job_tx: mpsc::Sender<(Particles, f64, u64)>,
    result_rx: mpsc::Receiver<(EnergyReadout, Duration)>,
    /// A measurement is in flight.
    busy: bool,
}

impl EnergyWorker {
    fn spawn(softening: f64, method: EnergyMethod) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<(Particles, f64, u64)>();
        let (result_tx, result_rx) = mpsc::channel();
        // Exits when the sim thread drops `job_tx`
        thread::spawn(move || {
            while let Ok((particles, sim_time, step)) = job_rx.recv() {
                let start = Instant::now();
                let readout = measure_energy(&particles, softening, method, sim_time, step);
                if result_tx.send((readout, start.elapsed())).is_err() {
                    return;
                }
            }
        });
        Self { job_tx, result_rx, busy: false }
    }

    fn start(&mut self, particles: &Particles, sim_time: f64, step: u64) {
        self.busy = self.job_tx.send((particles.clone(), sim_time, step)).is_ok();
    }

    /// The finished measurement and what it cost, if one landed.
    fn poll(&mut self) -> Option<(EnergyReadout, Duration)> {
        let result = self.result_rx.try_recv().ok();
        if result.is_some() {
            self.busy = false;
        }
        result
    }
}

/// Wait between energy measurements, given what the last one cost.
fn energy_interval(cost: Duration) -> Duration {
    MIN_ENERGY_INTERVAL.max(cost * ENERGY_COST_FACTOR)
}

fn build_render_snapshot(
    particles: &Particles,
    sim_time: f64,
    step: u64,
    energy: Option<EnergyReadout>,
    particle_types: &Arc<Vec<u8>>,
) -> RenderSnapshot {
    let n = particles.count;
    let mut positions = Vec::with_capacity(n);
    let mut internal_energies = Vec::with_capacity(n);
    let mut com = [0.0f64; 3];
    let mut m_total = 0.0f64;

    for i in 0..n {
        let m = particles.mass[i];
        com[0] += m * particles.x[i];
        com[1] += m * particles.y[i];
        com[2] += m * particles.z[i];
        m_total += m;
        positions.push([
            particles.x[i] as f32,
            particles.y[i] as f32,
            particles.z[i] as f32,
        ]);
        internal_energies.push(particles.internal_energy[i] as f32);
    }
    if m_total > 0.0 {
        for c in &mut com {
            *c /= m_total;
        }
    }

    RenderSnapshot {
        positions,
        particle_types: Arc::clone(particle_types),
        internal_energies,
        center_of_mass: [com[0] as f32, com[1] as f32, com[2] as f32],
        sim_time,
        step,
        energy,
        particle_count: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The HUD energy includes U for the whole run, not just the first
    /// snapshot (known issue 2 in docs/text-to-universe-handoff.md).
    #[test]
    fn energy_stays_conserved_after_first_second() {
        let cli = Cli::parse_from(["native-app", "--scenario", "plummer", "-n", "1000", "--speed", "2"]);
        let sim = spawn(&cli);
        let mut readings: Vec<EnergyReadout> = Vec::new();
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(3500) {
            while let Ok(snap) = sim.snapshot_rx.try_recv() {
                if let Some(e) = snap.energy
                    && readings.last().is_none_or(|prev| prev.sim_time != e.sim_time)
                {
                    readings.push(e);
                }
            }
            thread::sleep(Duration::from_millis(20));
        }
        drop(sim);

        assert!(readings.len() >= 3, "only {} energy readings", readings.len());
        let e0 = readings[0].total;
        for e in &readings {
            assert_eq!(e.method, EnergyMethod::Exact);
            assert!(e.potential < 0.0, "U missing at t={}", e.sim_time);
            let de = ((e.total - e0) / e0).abs();
            assert!(de < 1e-3, "dE/E = {de:.2e} at t={}", e.sim_time);
        }
    }

    #[test]
    fn energy_method_by_size() {
        assert_eq!(energy_method(1000, true), EnergyMethod::Exact);
        assert_eq!(energy_method(EXACT_ENERGY_MAX_N + 1, true), EnergyMethod::Tree(ENERGY_TREE_THETA));
        assert_eq!(energy_method(1000, false), EnergyMethod::NoGravity);
    }
}
