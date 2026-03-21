use wgpu::util::DeviceExt;

use super::gpu_types::{CameraUniform, GpuParticle, QuadVertex};
use crate::sim_thread::RenderSnapshot;

const PARTICLE_WGSL: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_right: vec3<f32>,
    _pad0: f32,
    camera_up: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) offset: vec2<f32>,
    @location(1) position: vec3<f32>,
    @location(2) mass: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) brightness: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let base_size = 0.03;
    let size = base_size * pow(max(in.mass * 1000.0, 0.01), 0.333);
    let world_pos = in.position
        + camera.camera_right * in.offset.x * size
        + camera.camera_up * in.offset.y * size;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = in.offset + vec2<f32>(0.5, 0.5);
    out.brightness = clamp(pow(max(in.mass * 1000.0, 0.01), 0.5), 0.3, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    if dist > 1.0 {
        discard;
    }
    let alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    let color = vec3<f32>(0.7, 0.85, 1.0) * in.brightness;
    return vec4<f32>(color * alpha, alpha);
}
"#;

// Billboard quad corners
const QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex { offset: [-0.5, -0.5] },
    QuadVertex { offset: [0.5, -0.5] },
    QuadVertex { offset: [0.5, 0.5] },
    QuadVertex { offset: [-0.5, 0.5] },
];

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

pub struct ParticlePipeline {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    max_particles: usize,
    current_count: u32,
}

impl ParticlePipeline {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat, depth_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // Slot 0: per-vertex quad offset
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    // Slot 1: per-instance particle data
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuParticle>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 2,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // billboards are double-sided
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false, // additive blending: no depth write
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad vertices"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad indices"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let max_particles = 20_000;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle instances"),
            size: (max_particles * std::mem::size_of::<GpuParticle>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            camera_buffer,
            camera_bind_group,
            max_particles,
            current_count: 0,
        }
    }

    pub fn camera_buffer(&self) -> &wgpu::Buffer {
        &self.camera_buffer
    }

    pub fn camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_bind_group
    }

    pub fn update_instances(&mut self, queue: &wgpu::Queue, snapshot: &RenderSnapshot, device: &wgpu::Device) {
        let count = snapshot.positions.len();

        // Grow instance buffer if needed
        if count > self.max_particles {
            self.max_particles = count * 2;
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particle instances"),
                size: (self.max_particles * std::mem::size_of::<GpuParticle>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        let gpu_particles: Vec<GpuParticle> = snapshot
            .positions
            .iter()
            .zip(snapshot.masses.iter())
            .map(|(pos, &mass)| GpuParticle {
                position: *pos,
                mass,
            })
            .collect();

        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&gpu_particles));
        self.current_count = count as u32;
    }

    pub fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.current_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..self.current_count);
    }
}
