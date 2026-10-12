//! A quick check run for a scenario spec.
//!
//! The `/universe` loop ([ADR 013](../../../docs/adr/013-universe-skill.md))
//! judges a spec by running it: did energy hold, and did the encounters the
//! description implies actually happen? `Preview` is that run, small and
//! steppable in chunks so the browser can drive it from a Web Worker without
//! freezing ([ADR 014](../../../docs/adr/014-public-universe-page.md)). It
//! uses the same component tracking as `headless`'s summary.
//!
//! Gravity only: a spec with gas is rejected, since the SPH energy problem
//! (see `specs/README.md`) makes a gas preview untrustworthy.

use crate::barnes_hut::BarnesHut;
use crate::diagnostics;
use crate::gravity::GravitySolver;
use crate::integrator::{Integrator, LeapfrogKDK};
use crate::particle::Particles;
use crate::scenario_builder;
use crate::spec::ScenarioSpec;
use crate::tracking::{self, ComponentState, Encounter, EncounterLog, Tracked};
use serde::Serialize;

/// Max |dE/E| that passes; the same thresholds as the skill's `check.py`.
pub const ENERGY_PASS: f64 = 1e-3;
/// Max |dE/E| that only warns; anything larger fails.
pub const ENERGY_WARN: f64 = 1e-2;

#[derive(Clone, Debug)]
pub struct PreviewConfig {
    /// Particle budget: larger specs are rescaled down to it.
    pub particles: usize,
    /// Barnes-Hut opening angle.
    pub theta: f64,
    /// Sim time to run to.
    pub until: f64,
    /// Step cap, so a spec with a tiny `dt` can't run for minutes. The run
    /// stops early (and says so) when `until` would need more.
    pub max_steps: u64,
    /// Energy samples over the run (each an O(N²) potential).
    pub energy_samples: u64,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        PreviewConfig {
            particles: 2000,
            theta: 0.5,
            until: 30.0,
            max_steps: 5000,
            energy_samples: 20,
        }
    }
}

pub struct Preview {
    name: String,
    view_radius: f64,
    requested_particles: usize,
    particles: Particles,
    gravity: BarnesHut,
    theta: f64,
    dt: f64,
    softening: f64,
    time: f64,
    step: u64,
    until: f64,
    total_steps: u64,
    energy_every: u64,
    e0: f64,
    max_drift: f64,
    energy: Vec<[f64; 2]>,
    tracked: Vec<Tracked>,
    initial: Vec<ComponentState>,
    log: EncounterLog,
}

/// What a preview found, serialized for the page (and for Claude to read).
#[derive(Serialize)]
pub struct PreviewReport {
    pub name: String,
    pub particles: usize,
    /// The spec's own particle count, before rescaling to the budget.
    pub requested_particles: usize,
    pub theta: f64,
    pub dt: f64,
    pub softening: f64,
    pub steps: u64,
    pub sim_time: f64,
    pub until: f64,
    pub done: bool,
    /// True if the step cap stopped the run short of `until`.
    pub truncated: bool,
    pub max_abs_energy_drift: f64,
    /// "pass", "warn" or "fail" (non-finite energy fails).
    pub energy_verdict: &'static str,
    /// `[time, (E − E₀)/|E₀|]` samples.
    pub energy: Vec<[f64; 2]>,
    /// Each component at t = 0 and now.
    pub initial: Vec<ComponentState>,
    pub components: Vec<ComponentState>,
    pub encounters: Vec<Encounter>,
}

impl Preview {
    pub fn new(spec: &ScenarioSpec, cfg: &PreviewConfig) -> Result<Self, String> {
        spec.validate()?;
        if spec.has_gas() {
            return Err(
                "gas_cloud components can't be previewed: self-gravitating gas \
                        is unreliable past peak compression (a known issue)"
                    .into(),
            );
        }
        if !(cfg.until > 0.0 && cfg.until.is_finite()) {
            return Err(format!("preview time must be positive, got {}", cfg.until));
        }
        let requested_particles = spec.particle_count();
        let spec = if requested_particles > cfg.particles {
            spec.with_particle_budget(cfg.particles)
        } else {
            spec.clone()
        };
        let built = scenario_builder::build_spec(&spec, None, None, None)?;
        let mut particles = built.particles;
        let gravity = BarnesHut::new(built.softening, cfg.theta);
        particles.clear_accelerations();
        gravity.compute_accelerations(&mut particles);

        let wanted = (cfg.until / built.dt).ceil() as u64;
        let total_steps = wanted.clamp(1, cfg.max_steps.max(1));
        let energy_every = (total_steps / cfg.energy_samples.max(1)).max(1);
        let e0 = diagnostics::compute(&particles, built.softening, 0.0, 0).total_energy;

        let tracked = tracking::from_spec(&particles, &spec);
        let initial = tracked
            .iter()
            .enumerate()
            .map(|(i, t)| t.state(&particles, i))
            .collect();
        let log = EncounterLog::new(&tracked);

        Ok(Preview {
            name: spec.name.clone(),
            view_radius: spec.view_radius(),
            requested_particles,
            particles,
            gravity,
            theta: cfg.theta,
            dt: built.dt,
            softening: built.softening,
            time: 0.0,
            step: 0,
            until: cfg.until,
            total_steps,
            energy_every,
            e0,
            max_drift: 0.0,
            energy: vec![[0.0, 0.0]],
            tracked,
            initial,
            log,
        })
    }

    /// Run up to `steps` more steps. Returns true once the run is complete.
    pub fn advance(&mut self, steps: u64) -> bool {
        for _ in 0..steps {
            if self.done() {
                break;
            }
            LeapfrogKDK.step(&mut self.particles, &self.gravity, self.dt);
            self.step += 1;
            self.time = self.step as f64 * self.dt;
            for t in &mut self.tracked {
                t.update(&self.particles);
            }
            self.log.record(&self.tracked, self.time);
            if self.step.is_multiple_of(self.energy_every) || self.done() {
                self.sample_energy();
            }
        }
        self.done()
    }

    fn sample_energy(&mut self) {
        let d = diagnostics::compute(&self.particles, self.softening, self.time, self.step);
        let drift = (d.total_energy - self.e0) / self.e0.abs();
        // NaN propagates through max(), so record it explicitly.
        self.max_drift = if drift.is_finite() {
            self.max_drift.max(drift.abs())
        } else {
            f64::INFINITY
        };
        self.energy.push([self.time, drift]);
    }

    pub fn done(&self) -> bool {
        self.step >= self.total_steps
    }

    /// Share of the run completed, in [0, 1].
    pub fn progress(&self) -> f64 {
        self.step as f64 / self.total_steps as f64
    }

    pub fn particle_count(&self) -> usize {
        self.particles.count
    }

    /// The current state, for drawing frames.
    pub fn particles(&self) -> &Particles {
        &self.particles
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// The spec's framing radius around the center of mass.
    pub fn view_radius(&self) -> f64 {
        self.view_radius
    }

    pub fn report(&self) -> PreviewReport {
        let verdict = if !self.max_drift.is_finite() || self.max_drift > ENERGY_WARN {
            "fail"
        } else if self.max_drift > ENERGY_PASS {
            "warn"
        } else {
            "pass"
        };
        PreviewReport {
            name: self.name.clone(),
            particles: self.particles.count,
            requested_particles: self.requested_particles,
            theta: self.theta,
            dt: self.dt,
            softening: self.softening,
            steps: self.step,
            sim_time: self.time,
            until: self.until,
            done: self.done(),
            truncated: (self.total_steps as f64) * self.dt < self.until - 0.5 * self.dt,
            max_abs_energy_drift: self.max_drift,
            energy_verdict: verdict,
            energy: self.energy.clone(),
            initial: self.initial.clone(),
            components: self
                .tracked
                .iter()
                .enumerate()
                .map(|(i, t)| t.state(&self.particles, i))
                .collect(),
            encounters: self.log.snapshot(),
        }
    }
}
