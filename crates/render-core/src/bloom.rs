//! Multi-pass bloom post-processing pipeline.
//!
//! Bloom creates a glow effect around bright pixels:
//! 1. **Threshold**: extract pixels above a luminance threshold
//! 2. **Blur H/V**: separable 9-tap Gaussian blur at half resolution
//! 3. **Composite**: additive blend bloom back onto the scene
//!
//! All intermediate textures use Rgba16Float at half resolution for performance.

/// Fullscreen triangle vertex shader, shared by bloom and tonemap passes.
pub const FULLSCREEN_VERT: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle trick: 3 vertices cover the entire screen
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = uv[idx];
    return out;
}
"#;

const THRESHOLD_FRAG: &str = r#"
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

struct BloomParams {
    threshold: f32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(2) var<uniform> params: BloomParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@fragment
fn fs_threshold(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_input, s_input, in.uv);
    let luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let contribution = max(luminance - params.threshold, 0.0);
    let scale = contribution / max(luminance, 0.001);
    return vec4<f32>(color.rgb * scale, 1.0);
}
"#;

const BLUR_FRAG: &str = r#"
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

struct BlurDirection {
    direction: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(2) var<uniform> blur_dir: BlurDirection;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@fragment
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    // 9-tap Gaussian kernel (sigma ≈ 2.0)
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    let tex_size = vec2<f32>(textureDimensions(t_input));
    let texel = blur_dir.direction / tex_size;

    var result = textureSample(t_input, s_input, in.uv) * weights[0];
    for (var i: i32 = 1; i < 5; i++) {
        let offset = texel * f32(i);
        result += textureSample(t_input, s_input, in.uv + offset) * weights[i];
        result += textureSample(t_input, s_input, in.uv - offset) * weights[i];
    }
    return result;
}
"#;

const COMPOSITE_FRAG: &str = r#"
@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var t_bloom: texture_2d<f32>;
@group(0) @binding(2) var s_input: sampler;

struct BloomParams {
    threshold: f32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(3) var<uniform> params: BloomParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(t_scene, s_input, in.uv);
    let bloom = textureSample(t_bloom, s_input, in.uv);
    return vec4<f32>(scene.rgb + bloom.rgb * params.intensity, scene.a);
}
"#;

/// Uniform buffer for bloom parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomParams {
    threshold: f32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Uniform buffer for blur direction.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurDirection {
    direction: [f32; 2],
    _pad: [f32; 2],
}

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct BloomPipeline {
    threshold_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
    blur_h_buffer: wgpu::Buffer,
    blur_v_buffer: wgpu::Buffer,

    // Textures at half resolution
    brightness_texture: wgpu::Texture,
    blur_intermediate: wgpu::Texture,
    bloom_texture: wgpu::Texture,

    // Bind group layouts (needed for recreation on resize)
    threshold_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,

    // Bind groups (recreated on resize and when HDR source changes)
    threshold_bg: wgpu::BindGroup,
    blur_h_bg: wgpu::BindGroup,
    blur_v_bg: wgpu::BindGroup,
    composite_bg: wgpu::BindGroup,

    half_width: u32,
    half_height: u32,
}

impl BloomPipeline {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, hdr_view: &wgpu::TextureView) -> Self {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params = BloomParams {
            threshold: 0.8,
            intensity: 0.3,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom params"),
            size: std::mem::size_of::<BloomParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let blur_h = BlurDirection {
            direction: [1.0, 0.0],
            _pad: [0.0; 2],
        };
        let blur_v = BlurDirection {
            direction: [0.0, 1.0],
            _pad: [0.0; 2],
        };
        let blur_h_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur h dir"),
            size: std::mem::size_of::<BlurDirection>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur v dir"),
            size: std::mem::size_of::<BlurDirection>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&blur_h_buffer, 0, bytemuck::bytes_of(&blur_h));
        queue.write_buffer(&blur_v_buffer, 0, bytemuck::bytes_of(&blur_v));

        // Textures
        let brightness_texture =
            create_half_res_texture(device, half_width, half_height, "bloom brightness");
        let blur_intermediate =
            create_half_res_texture(device, half_width, half_height, "bloom blur intermediate");
        let bloom_texture =
            create_half_res_texture(device, half_width, half_height, "bloom final");

        // --- Threshold pipeline ---
        let threshold_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom threshold bgl"),
                entries: &[
                    bgl_texture(0),
                    bgl_sampler(1),
                    bgl_uniform(2),
                ],
            });

        let threshold_shader_src = format!("{FULLSCREEN_VERT}\n{THRESHOLD_FRAG}");
        let threshold_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bloom threshold shader"),
                source: wgpu::ShaderSource::Wgsl(threshold_shader_src.into()),
            });

        let threshold_pipeline = create_fullscreen_pipeline(
            device,
            &threshold_shader,
            "vs_main",
            "fs_threshold",
            &threshold_bgl,
            HDR_FORMAT,
            "bloom threshold",
        );

        // --- Blur pipeline ---
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom blur bgl"),
            entries: &[
                bgl_texture(0),
                bgl_sampler(1),
                bgl_uniform(2),
            ],
        });

        let blur_shader_src = format!("{FULLSCREEN_VERT}\n{BLUR_FRAG}");
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom blur shader"),
            source: wgpu::ShaderSource::Wgsl(blur_shader_src.into()),
        });

        let blur_pipeline = create_fullscreen_pipeline(
            device,
            &blur_shader,
            "vs_main",
            "fs_blur",
            &blur_bgl,
            HDR_FORMAT,
            "bloom blur",
        );

        // --- Composite pipeline ---
        let composite_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom composite bgl"),
                entries: &[
                    bgl_texture(0),
                    bgl_texture(1),
                    bgl_sampler(2),
                    bgl_uniform(3),
                ],
            });

        let composite_shader_src = format!("{FULLSCREEN_VERT}\n{COMPOSITE_FRAG}");
        let composite_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bloom composite shader"),
                source: wgpu::ShaderSource::Wgsl(composite_shader_src.into()),
            });

        let composite_pipeline = create_fullscreen_pipeline(
            device,
            &composite_shader,
            "vs_main",
            "fs_composite",
            &composite_bgl,
            HDR_FORMAT,
            "bloom composite",
        );

        // Create initial bind groups
        let brightness_view = brightness_texture.create_view(&Default::default());
        let blur_int_view = blur_intermediate.create_view(&Default::default());

        let blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom blur h bg"),
            layout: &blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&brightness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blur_h_buffer.as_entire_binding(),
                },
            ],
        });

        let blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom blur v bg"),
            layout: &blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_int_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blur_v_buffer.as_entire_binding(),
                },
            ],
        });

        let bloom_view = bloom_texture.create_view(&Default::default());

        let threshold_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom threshold bg"),
            layout: &threshold_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom composite bg"),
            layout: &composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            threshold_pipeline,
            blur_pipeline,
            composite_pipeline,
            sampler,
            params_buffer,
            blur_h_buffer,
            blur_v_buffer,
            brightness_texture,
            blur_intermediate,
            bloom_texture,
            threshold_bgl,
            blur_bgl,
            composite_bgl,
            threshold_bg,
            blur_h_bg,
            blur_v_bg,
            composite_bg,
            half_width,
            half_height,
        }
    }

    /// Recreate textures and bind groups when the window resizes.
    /// Must be followed by `update_source` with the new HDR view.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.half_width = (width / 2).max(1);
        self.half_height = (height / 2).max(1);

        self.brightness_texture =
            create_half_res_texture(device, self.half_width, self.half_height, "bloom brightness");
        self.blur_intermediate = create_half_res_texture(
            device,
            self.half_width,
            self.half_height,
            "bloom blur intermediate",
        );
        self.bloom_texture =
            create_half_res_texture(device, self.half_width, self.half_height, "bloom final");

        self.recreate_blur_bind_groups(device);
    }

    /// Recreate threshold and composite bind groups when the HDR source changes.
    pub fn update_source(&mut self, device: &wgpu::Device, hdr_view: &wgpu::TextureView) {
        let bloom_view = self.bloom_texture.create_view(&Default::default());

        self.threshold_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom threshold bg"),
            layout: &self.threshold_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        self.composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom composite bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
    }

    fn recreate_blur_bind_groups(&mut self, device: &wgpu::Device) {
        let brightness_view = self.brightness_texture.create_view(&Default::default());
        let blur_int_view = self.blur_intermediate.create_view(&Default::default());

        self.blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom blur h bg"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&brightness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.blur_h_buffer.as_entire_binding(),
                },
            ],
        });

        self.blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom blur v bg"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_int_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.blur_v_buffer.as_entire_binding(),
                },
            ],
        });
    }

    /// Run the bloom post-processing passes using cached bind groups.
    /// Call `update_source` whenever the HDR view changes (e.g. on resize).
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
        let brightness_view = self.brightness_texture.create_view(&Default::default());
        let blur_int_view = self.blur_intermediate.create_view(&Default::default());
        let bloom_view = self.bloom_texture.create_view(&Default::default());

        // Pass 1: Threshold → brightness_texture
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom threshold pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &brightness_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.threshold_pipeline);
            pass.set_bind_group(0, &self.threshold_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: Horizontal blur → blur_intermediate
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom blur h pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &blur_int_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_h_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 3: Vertical blur → bloom_texture
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom blur v pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &bloom_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_v_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 4: Composite scene + bloom → output
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_half_res_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Create a depth texture for the scene render pass.
pub fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Create an HDR texture at full resolution for the scene render target.
pub fn create_hdr_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("HDR render target"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn bgl_texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_sampler(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
    bgl: &wgpu::BindGroupLayout,
    target_format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label} layout")),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs_entry),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}
