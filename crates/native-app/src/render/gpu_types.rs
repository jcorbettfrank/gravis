/// Per-instance data for a single particle, packed for GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 3],
    pub mass: f32,
}

/// Camera uniform sent to vertex shaders for view-projection and billboarding.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_right: [f32; 3],
    pub _pad0: f32,
    pub camera_up: [f32; 3],
    pub _pad1: f32,
}

/// Per-vertex data for the billboard quad.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadVertex {
    pub offset: [f32; 2],
}

/// Per-vertex data for coordinate axes.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AxesVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}
