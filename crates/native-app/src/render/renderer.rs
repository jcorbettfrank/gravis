use std::sync::Arc;

use winit::window::Window;

use render_core::axes::AxesPipeline;
use render_core::bloom::{self, BloomPipeline, DEPTH_FORMAT, HDR_FORMAT};
use render_core::color::{ParticleStyle, ParticleStyler};
use render_core::gpu_types::CameraUniform;
use render_core::particles::ParticlePipeline;
use render_core::tonemap::ToneMapPipeline;

use super::ui::UiState;
use crate::sim_thread::RenderSnapshot;

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
    // Latest snapshot's positions and styles, kept so a style change (the
    // dark-matter toggle) can be re-uploaded while the sim is paused.
    cached_positions: Vec<[f32; 3]>,
    cached_types: Arc<Vec<u8>>,
    cached_energies: Vec<f32>,
    cached_styles: Vec<ParticleStyle>,
    styled_with_dark_matter: bool,
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
        let depth_view = bloom::create_depth_texture(device, width, height);

        let bloom_pipeline = BloomPipeline::new(device, queue, width, height, &hdr_view);
        let tonemap_pipeline = ToneMapPipeline::new(device, surface_format, &hdr_composited_view);

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
            cached_positions: Vec::new(),
            cached_types: Arc::new(Vec::new()),
            cached_energies: Vec::new(),
            cached_styles: Vec::new(),
            styled_with_dark_matter: false,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth_view = bloom::create_depth_texture(device, width, height);
        self.hdr_texture = bloom::create_hdr_texture(device, width, height);
        self.hdr_view = self.hdr_texture.create_view(&Default::default());
        self.hdr_composited = bloom::create_hdr_texture(device, width, height);
        self.hdr_composited_view = self.hdr_composited.create_view(&Default::default());
        self.bloom_pipeline.resize(device, width, height);
        self.bloom_pipeline.update_source(device, &self.hdr_view);
        self.tonemap_pipeline.update_source(device, &self.hdr_composited_view);
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

    #[allow(clippy::too_many_arguments)]
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

        // Update particle instances on new data or a style change
        let show_dark_matter = ui_state.show_dark_matter();
        if let Some(snap) = new_snapshot {
            self.cached_positions.clone_from(&snap.positions);
            self.cached_types = Arc::clone(&snap.particle_types);
            self.cached_energies.clone_from(&snap.internal_energies);
        }
        if new_snapshot.is_some() || show_dark_matter != self.styled_with_dark_matter {
            let styler = ParticleStyler::new(&self.cached_types, show_dark_matter);
            self.cached_styles = styler.style_all(&self.cached_types, &self.cached_energies);
            self.styled_with_dark_matter = show_dark_matter;
            self.particle_pipeline.update_instances(
                queue,
                device,
                &self.cached_positions,
                &self.cached_styles,
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
            .render(&mut encoder, &self.hdr_composited_view);

        // Pass 3: Tone map HDR → LDR surface
        self.tonemap_pipeline.render(&mut encoder, target);

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

