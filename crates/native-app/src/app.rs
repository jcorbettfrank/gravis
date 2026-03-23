use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};

use crate::render::camera::NativeCamera;
use crate::render::renderer::Renderer;
use crate::render::ui::UiState;
use crate::sim_thread::{self, SimHandle};
use crate::Cli;

pub struct App {
    cli: Cli,
    state: AppState,
}

#[allow(clippy::large_enum_variant)]
enum AppState {
    Uninitialized,
    Running(RunningState),
}

struct RunningState {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    camera: NativeCamera,
    ui: UiState,
    sim: SimHandle,
    last_frame: Instant,
    screenshot_frames_remaining: Option<u32>,
    screenshot_path: Option<String>,
}

impl App {
    pub fn new(cli: Cli) -> Self {
        Self {
            cli,
            state: AppState::Uninitialized,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if matches!(self.state, AppState::Running(_)) {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("gravis — N-body simulation")
            .with_inner_size(LogicalSize::new(1280u32, 800u32))
            .with_min_inner_size(LogicalSize::new(640u32, 480u32));

        let window = Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gravis device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .expect("Failed to create device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let renderer = Renderer::new(&device, &queue, format, size.width, size.height, &window);
        let camera = NativeCamera::new(size.width as f32 / size.height as f32);
        let ui = UiState::new(self.cli.speed, self.cli.algorithm.clone());
        let sim = sim_thread::spawn(&self.cli);

        let screenshot_frames_remaining = self.cli.screenshot.as_ref().map(|_| 100u32);
        let screenshot_path = self.cli.screenshot.clone();

        self.state = AppState::Running(RunningState {
            window,
            device,
            queue,
            surface,
            surface_config,
            renderer,
            camera,
            ui,
            sim,
            last_frame: Instant::now(),
            screenshot_frames_remaining,
            screenshot_path,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let AppState::Running(state) = &mut self.state else {
            return;
        };

        // Let egui handle input first
        let egui_consumed = state
            .renderer
            .handle_window_event(&state.window, &event);

        match event {
            WindowEvent::CloseRequested => {
                state.sim.stop();
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                state.sim.stop();
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.surface_config.width = new_size.width;
                    state.surface_config.height = new_size.height;
                    state.surface.configure(&state.device, &state.surface_config);
                    state.renderer.resize(
                        &state.device,
                        new_size.width,
                        new_size.height,
                    );
                    state
                        .camera
                        .set_aspect_ratio(new_size.width as f32 / new_size.height as f32);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let frame_dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;

                // Drain sim snapshots — keep only latest
                let new_snapshot = state.sim.drain_latest();
                if let Some(ref snap) = new_snapshot {
                    state
                        .camera
                        .set_target(glam::Vec3::from(snap.center_of_mass));
                    state.ui.update_from_snapshot(snap);
                }

                state.ui.record_frame_time(frame_dt);
                state.camera.update(frame_dt);

                // Send speed/pause commands to sim thread
                state.sim.set_speed(state.ui.speed_multiplier());
                if state.ui.paused() {
                    state.sim.pause();
                } else {
                    state.sim.resume();
                }

                let camera_uniform = state.camera.build_uniform();

                let surface_tex = match state.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                        state.surface.configure(&state.device, &state.surface_config);
                        return;
                    }
                    Err(e) => {
                        eprintln!("Surface error: {e}");
                        return;
                    }
                };

                let view = surface_tex
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                state.renderer.render(
                    &state.device,
                    &state.queue,
                    &view,
                    &camera_uniform,
                    new_snapshot.as_ref(),
                    &mut state.ui,
                    &state.window,
                    [state.surface_config.width, state.surface_config.height],
                );

                surface_tex.present();

                // Screenshot mode
                if let Some(ref mut remaining) = state.screenshot_frames_remaining {
                    if *remaining == 0 {
                        if let Some(path) = state.screenshot_path.take() {
                            capture_screenshot(
                                &state.device,
                                &state.queue,
                                &state.surface,
                                &state.surface_config,
                                &mut state.renderer,
                                &state.camera,
                                &mut state.ui,
                                &state.window,
                                &path,
                            );
                        }
                        state.sim.stop();
                        event_loop.exit();
                    } else {
                        *remaining -= 1;
                    }
                }
            }
            ref evt => {
                if !egui_consumed {
                    state.camera.handle_event(evt);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let AppState::Running(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_screenshot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    _surface: &wgpu::Surface,
    config: &wgpu::SurfaceConfiguration,
    renderer: &mut Renderer,
    camera: &NativeCamera,
    ui: &mut UiState,
    window: &Window,
    path: &str,
) {
    let width = config.width;
    let height = config.height;

    // Render one more frame to an offscreen texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let camera_uniform = camera.build_uniform();
    renderer.render(device, queue, &view, &camera_uniform, None, ui, window, [width, height]);

    // Copy texture to buffer
    let bytes_per_row = (width * 4).div_ceil(256) * 256;
    let buf_size = (bytes_per_row * height) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot staging"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("screenshot copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().unwrap().expect("Failed to map screenshot buffer");

    let data = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * bytes_per_row) as usize;
        let end = start + (width * 4) as usize;
        pixels.extend_from_slice(&data[start..end]);
    }
    drop(data);
    staging.unmap();

    // Convert BGRA to RGBA if needed
    if config.format == wgpu::TextureFormat::Bgra8UnormSrgb
        || config.format == wgpu::TextureFormat::Bgra8Unorm
    {
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
    }

    // Save as PNG
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let img = image::RgbaImage::from_raw(width, height, pixels).expect("Invalid image data");
    img.save(path).expect("Failed to save screenshot");
    eprintln!("Screenshot saved to {path}");
}
