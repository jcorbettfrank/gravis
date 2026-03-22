use glam::Vec3;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

use render_core::camera::OrbitalCamera;
use render_core::gpu_types::CameraUniform;

/// Native camera that wraps render-core's OrbitalCamera and adds
/// winit input handling (mouse drag for orbit, scroll for zoom).
pub struct NativeCamera {
    pub inner: OrbitalCamera,
    dragging: bool,
    last_cursor: [f64; 2],
}

impl NativeCamera {
    pub fn new(aspect_ratio: f32) -> Self {
        Self {
            inner: OrbitalCamera::new(aspect_ratio),
            dragging: false,
            last_cursor: [0.0; 2],
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect: f32) {
        self.inner.set_aspect_ratio(aspect);
    }

    pub fn set_target(&mut self, target: Vec3) {
        self.inner.set_target(target);
    }

    pub fn update(&mut self, dt: f32) {
        self.inner.update(dt);
    }

    pub fn build_uniform(&self) -> CameraUniform {
        self.inner.build_uniform()
    }

    pub fn handle_event(&mut self, event: &WindowEvent) {
        match *event {
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.dragging = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    let dx = (position.x - self.last_cursor[0]) as f32;
                    let dy = (position.y - self.last_cursor[1]) as f32;
                    self.inner.desired_yaw -= dx * 0.005;
                    self.inner.desired_pitch += dy * 0.005;
                    self.inner.desired_pitch = self.inner.desired_pitch.clamp(-1.55, 1.55);
                }
                self.last_cursor = [position.x, position.y];
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.inner.desired_distance *= 1.0 - scroll * 0.1;
                self.inner.desired_distance = self.inner.desired_distance.clamp(0.1, 100.0);
            }
            _ => {}
        }
    }
}
