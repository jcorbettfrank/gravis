use glam::{Mat4, Vec3};

use crate::gpu_types::CameraUniform;

/// Orbital camera that orbits around a target point.
///
/// Platform code drives this by setting desired values (yaw, pitch, distance, target)
/// and calling `update()` each frame. The camera smoothly interpolates to the desired
/// state using exponential smoothing.
pub struct OrbitalCamera {
    // Current smoothed values
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,

    // Target values (input drives these, smoothing catches up)
    pub desired_target: Vec3,
    pub desired_distance: f32,
    pub desired_yaw: f32,
    pub desired_pitch: f32,

    // Projection
    aspect_ratio: f32,
    fov_y: f32,
    near: f32,
    far: f32,

    // Smoothing factor (higher = snappier)
    smoothing: f32,
}

impl OrbitalCamera {
    pub fn new(aspect_ratio: f32) -> Self {
        let distance = 5.0;
        let yaw = 0.3;
        let pitch = 0.3;
        Self {
            target: Vec3::ZERO,
            distance,
            yaw,
            pitch,
            desired_target: Vec3::ZERO,
            desired_distance: distance,
            desired_yaw: yaw,
            desired_pitch: pitch,
            aspect_ratio,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.001,
            far: 1000.0,
            smoothing: 10.0,
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect: f32) {
        self.aspect_ratio = aspect;
    }

    pub fn set_target(&mut self, target: Vec3) {
        self.desired_target = target;
    }

    pub fn update(&mut self, dt: f32) {
        let t = 1.0 - (-self.smoothing * dt).exp();
        self.yaw = lerp(self.yaw, self.desired_yaw, t);
        self.pitch = lerp(self.pitch, self.desired_pitch, t);
        self.distance = lerp(self.distance, self.desired_distance, t);
        self.target = self.target.lerp(self.desired_target, t);
    }

    pub fn eye_position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    pub fn build_uniform(&self) -> CameraUniform {
        let eye = self.eye_position();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.near, self.far);
        let view_proj = proj * view;

        // Extract camera right and up from view matrix for billboarding
        let camera_right = Vec3::new(view.col(0).x, view.col(1).x, view.col(2).x);
        let camera_up = Vec3::new(view.col(0).y, view.col(1).y, view.col(2).y);

        CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_right: camera_right.to_array(),
            _pad0: 0.0,
            camera_up: camera_up.to_array(),
            _pad1: 0.0,
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
