use winit::window::Window;

use render_core::axes::AxesPipeline;
use render_core::bloom::{self, BloomPipeline, HDR_FORMAT};
use render_core::color;
use render_core::gpu_types::CameraUniform;
use render_core::particles::ParticlePipeline;
use render_core::tonemap::ToneMapPipeline;

use super::ui::UiState;
use crate::sim_thread::RenderSnapshot;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Renderer {
    particle_pipeline: ParticlePipeline,
    axes_pipeline: AxesPipeline,
    bloom_pipeline: BloomPipeline,
    tonemap_pipeline: ToneMapPipeline,
    #[allow(dead_code)] // texture must outlive its view
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    #[allow(dead_code)] // texture must outlive its view
    hdr_composited: wgpu::Texture,
    hdr_composited_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    egui_ctx: egui::Context,
    // Cached colors for current snapshot
    cached_colors: Vec<[f32; 4]>,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        window: &Window,
    ) -> Self {
        // Particle and axes pipelines target HDR format (not surface format)
        let particle_pipeline = ParticlePipeline::new(device, HDR_FORMAT, DEPTH_FORMAT);

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bind group layout (axes)"),
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

        let axes_pipeline =
            AxesPipeline::new(device, HDR_FORMAT, DEPTH_FORMAT, &camera_bind_group_layout);

        let hdr_texture = bloom::create_hdr_texture(device, width, height);
        let hdr_view = hdr_texture.create_view(&Default::default());
        let hdr_composited = bloom::create_hdr_texture(device, width, height);
        let hdr_composited_view = hdr_composited.create_view(&Default::default());
        let depth_view = create_depth_texture(device, width, height);

        let bloom_pipeline = BloomPipeline::new(device, queue, width, height);
        let tonemap_pipeline = ToneMapPipeline::new(device, surface_format);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            particle_pipeline,
            axes_pipeline,
            bloom_pipeline,
            tonemap_pipeline,
            hdr_texture,
            hdr_view,
            hdr_composited,
            hdr_composited_view,
            depth_view,
            egui_renderer,
            egui_state,
            egui_ctx,
            cached_colors: Vec::new(),
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth_view = create_depth_texture(device, width, height);
        self.hdr_texture = bloom::create_hdr_texture(device, width, height);
        self.hdr_view = self.hdr_texture.create_view(&Default::default());
        self.hdr_composited = bloom::create_hdr_texture(device, width, height);
        self.hdr_composited_view = self.hdr_composited.create_view(&Default::default());
        self.bloom_pipeline.resize(device, width, height);
    }

    /// Returns true if egui consumed the event.
    pub fn handle_window_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let response = self.egui_state.on_window_event(window, event);
        response.consumed
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        camera_uniform: &CameraUniform,
        new_snapshot: Option<&RenderSnapshot>,
        ui_state: &mut UiState,
        window: &Window,
        size: [u32; 2],
    ) {
        // Update camera uniform
        queue.write_buffer(
            self.particle_pipeline.camera_buffer(),
            0,
            bytemuck::bytes_of(camera_uniform),
        );

        // Update particle instances if we have new data
        if let Some(snap) = new_snapshot {
            // Compute colors from particle types
            self.cached_colors.clear();
            self.cached_colors.reserve(snap.particle_types.len());
            for &pt in &snap.particle_types {
                self.cached_colors.push(color::particle_type_to_color(pt));
            }
            self.particle_pipeline.update_instances(
                queue,
                device,
                &snap.positions,
                &snap.masses,
                &self.cached_colors,
            );
        }

        // Build egui
        let raw_input = self.egui_state.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            ui_state.draw(ctx);
        });
        self.egui_state
            .handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size,
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(device, queue, *id, delta);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

        // Pass 1: Render axes + particles to HDR texture
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HDR scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Draw axes first (writes depth)
            self.axes_pipeline
                .draw(&mut pass, self.particle_pipeline.camera_bind_group());

            // Draw particles (reads depth, no write, additive blend)
            self.particle_pipeline.draw(&mut pass);
        }

        // Pass 2: Bloom (threshold → blur → composite) → hdr_composited
        self.bloom_pipeline
            .render(device, &mut encoder, &self.hdr_view, &self.hdr_composited_view);

        // Pass 3: Tone map HDR → LDR surface
        self.tonemap_pipeline
            .render(device, &mut encoder, &self.hdr_composited_view, target);

        queue.submit(Some(encoder.finish()));

        // egui pass (on top of LDR surface)
        let mut egui_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui encoder"),
        });

        self.egui_renderer.update_buffers(
            device,
            queue,
            &mut egui_encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        let mut pass = egui_encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            })
            .forget_lifetime();

        self.egui_renderer
            .render(&mut pass, &paint_jobs, &screen_descriptor);
        drop(pass);

        queue.submit(Some(egui_encoder.finish()));

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
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
