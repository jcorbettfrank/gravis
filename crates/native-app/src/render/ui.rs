use std::collections::VecDeque;

use crate::sim_thread::RenderSnapshot;

const FPS_HISTORY_CAP: usize = 120;
const ENERGY_HISTORY_CAP: usize = 500;

pub struct UiState {
    fps_history: VecDeque<f32>,
    energy_history: VecDeque<[f64; 2]>, // [sim_time, total_energy]
    initial_energy: Option<f64>,
    paused: bool,
    speed: f64,
    particle_count: usize,
    sim_time: f64,
    step: u64,
    total_energy: f64,
    kinetic_energy: f64,
    potential_energy: f64,
    virial_ratio: f64,
}

impl UiState {
    pub fn new(initial_speed: f64) -> Self {
        Self {
            fps_history: VecDeque::with_capacity(FPS_HISTORY_CAP),
            energy_history: VecDeque::with_capacity(ENERGY_HISTORY_CAP),
            initial_energy: None,
            paused: false,
            speed: initial_speed,
            particle_count: 0,
            sim_time: 0.0,
            step: 0,
            total_energy: 0.0,
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            virial_ratio: 0.0,
        }
    }

    pub fn record_frame_time(&mut self, dt: f32) {
        if self.fps_history.len() >= FPS_HISTORY_CAP {
            self.fps_history.pop_front();
        }
        self.fps_history.push_back(dt);
    }

    pub fn update_from_snapshot(&mut self, snap: &RenderSnapshot) {
        self.particle_count = snap.particle_count;
        self.sim_time = snap.sim_time;
        self.step = snap.step;
        self.total_energy = snap.total_energy;
        self.kinetic_energy = snap.kinetic_energy;
        self.potential_energy = snap.potential_energy;
        self.virial_ratio = snap.virial_ratio;

        if self.initial_energy.is_none() && snap.total_energy != 0.0 {
            self.initial_energy = Some(snap.total_energy);
        }

        if self.energy_history.len() >= ENERGY_HISTORY_CAP {
            self.energy_history.pop_front();
        }
        self.energy_history
            .push_back([snap.sim_time, snap.total_energy]);
    }

    pub fn speed_multiplier(&self) -> f64 {
        self.speed
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    fn avg_fps(&self) -> f32 {
        if self.fps_history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.fps_history.iter().sum();
        let avg_dt = sum / self.fps_history.len() as f32;
        if avg_dt > 0.0 {
            1.0 / avg_dt
        } else {
            0.0
        }
    }

    pub fn draw(&mut self, ctx: &egui::Context) {
        egui::Window::new("Simulation")
            .default_pos([10.0, 10.0])
            .default_width(260.0)
            .resizable(true)
            .show(ctx, |ui| {
                // Stats
                ui.label(format!(
                    "FPS: {:.0}  |  Particles: {}",
                    self.avg_fps(),
                    self.particle_count
                ));
                ui.label(format!(
                    "Sim time: {:.4}  |  Step: {}",
                    self.sim_time, self.step
                ));

                ui.separator();

                // Speed control
                ui.horizontal(|ui| {
                    ui.label("Speed:");
                    for &s in &[1.0, 10.0, 100.0] {
                        let label = format!("{s:.0}x");
                        if ui
                            .selectable_label((self.speed - s).abs() < 0.1, &label)
                            .clicked()
                        {
                            self.speed = s;
                        }
                    }
                });

                // Pause / Resume
                if ui
                    .button(if self.paused { "Resume" } else { "Pause" })
                    .clicked()
                {
                    self.paused = !self.paused;
                }

                ui.separator();

                // Energy diagnostics
                let de_rel = if let Some(e0) = self.initial_energy {
                    if e0.abs() > 0.0 {
                        (self.total_energy - e0) / e0.abs()
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                ui.label(format!("E = {:.6e}", self.total_energy));
                ui.label(format!("dE/E = {:+.4e}", de_rel));
                ui.label(format!(
                    "K = {:.4e}  U = {:.4e}",
                    self.kinetic_energy, self.potential_energy
                ));
                ui.label(format!("Virial 2K/|U| = {:.4}", self.virial_ratio));

                // Energy conservation plot
                if !self.energy_history.is_empty() {
                    ui.separator();
                    ui.label("Energy drift (dE/E):");

                    let e0 = self.initial_energy.unwrap_or(self.total_energy);
                    let points: egui_plot::PlotPoints = self
                        .energy_history
                        .iter()
                        .map(|&[t, e]| {
                            let de = if e0.abs() > 0.0 {
                                (e - e0) / e0.abs()
                            } else {
                                0.0
                            };
                            [t, de]
                        })
                        .collect();

                    let line = egui_plot::Line::new("dE/E", points)
                        .color(egui::Color32::from_rgb(100, 200, 255));

                    egui_plot::Plot::new("energy_plot")
                        .height(120.0)
                        .show_axes([true, true])
                        .auto_bounds(true)
                        .show(ui, |plot_ui| {
                            plot_ui.line(line);
                        });
                }
            });
    }
}
