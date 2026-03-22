use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use render_core::axes::AxesPipeline;
use render_core::camera::OrbitalCamera;
use render_core::particles::ParticlePipeline;

use sim_core::barnes_hut::BarnesHut;
use sim_core::gravity::{BruteForce, GravitySolver};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::particle::Particles;
use sim_core::scenarios::plummer_sphere::PlummerSphere;
use sim_core::scenarios::two_body::TwoBody;
use sim_core::scenario::Scenario;

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------

struct SimState {
    particles: Particles,
    gravity: Box<dyn GravitySolver>,
    integrator: LeapfrogKDK,
    dt: f64,
    sim_time: f64,
    step: u64,
}

impl SimState {
    fn step_once(&mut self) {
        let dt = self.dt;
        self.integrator
            .step(&mut self.particles, self.gravity.as_ref(), dt);
        self.sim_time += dt;
        self.step += 1;
    }
}

struct AppState {
    // wgpu
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // render-core pipelines
    particle_pipeline: ParticlePipeline,
    axes_pipeline: AxesPipeline,
    depth_view: wgpu::TextureView,
    camera: OrbitalCamera,

    // simulation
    sim: SimState,

    // controls
    speed_multiplier: f64,
    paused: bool,
    scenario_name: String,
    algorithm_name: String,
    particle_count: usize,

    // timing
    accumulator: f64,
    last_frame_ms: f64,
    fps: f64,
    frames_since_stats: u32,

    // input
    dragging: bool,
    last_cursor: [f64; 2],
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

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

fn create_sim(scenario: &str, n: usize, algorithm: &str) -> SimState {
    let softening = 0.05;
    let particles: Particles = match scenario {
        "two-body" => {
            let s = TwoBody {
                eccentricity: 0.5,
                ..Default::default()
            };
            s.generate()
        }
        _ => {
            let s = PlummerSphere {
                n,
                ..Default::default()
            };
            s.generate()
        }
    };

    let gravity: Box<dyn GravitySolver> = match algorithm {
        "brute-force" => Box::new(BruteForce::new(softening)),
        _ => Box::new(BarnesHut::new(softening, 0.5)),
    };

    // Larger timestep than native — reduces steps-per-frame by 5x.
    // Still accurate enough for demo purposes (energy drift < 1% over ~1000 steps).
    let dt = 0.005;
    let integrator = LeapfrogKDK;
    // Initialize accelerations for leapfrog
    let mut p = particles;
    gravity.compute_accelerations(&mut p);

    SimState {
        particles: p,
        gravity,
        integrator,
        dt,
        sim_time: 0.0,
        step: 0,
    }
}

// ---------------------------------------------------------------------------
// WASM entry point
// ---------------------------------------------------------------------------

#[wasm_bindgen(start)]
pub async fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("gravis web-app starting");

    let window = web_sys::window().expect("no global window");
    let document = window.document().expect("no document");
    let canvas: HtmlCanvasElement = document
        .get_element_by_id("simulation")
        .expect("no #simulation canvas")
        .dyn_into()
        .expect("not a canvas element");

    let width = canvas.client_width() as u32;
    let height = canvas.client_height() as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    // wgpu init
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
    let surface = instance
        .create_surface(surface_target)
        .expect("Failed to create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("No suitable GPU adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("gravis device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits()),
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats[0];

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    let particle_pipeline = ParticlePipeline::new(&device, format, DEPTH_FORMAT);

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

    let axes_pipeline = AxesPipeline::new(&device, format, DEPTH_FORMAT, &camera_bind_group_layout);
    let depth_view = create_depth_texture(&device, width, height);
    let camera = OrbitalCamera::new(width as f32 / height.max(1) as f32);

    // Read initial config from DOM
    let scenario_name = read_select_value(&document, "scenario").unwrap_or("plummer".into());
    let algorithm_name = read_select_value(&document, "algorithm").unwrap_or("barnes-hut".into());
    let particle_count = read_range_value(&document, "particle-count").unwrap_or(5000) as usize;
    let speed = read_range_value(&document, "speed").unwrap_or(1) as f64;

    let sim = create_sim(&scenario_name, particle_count, &algorithm_name);

    let state = Rc::new(RefCell::new(AppState {
        device,
        queue,
        surface,
        surface_config,
        particle_pipeline,
        axes_pipeline,
        depth_view,
        camera,
        sim,
        speed_multiplier: speed,
        paused: false,
        scenario_name,
        algorithm_name,
        particle_count,
        accumulator: 0.0,
        last_frame_ms: 0.0,
        fps: 0.0,
        frames_since_stats: 0,
        dragging: false,
        last_cursor: [0.0; 2],
    }));

    // --- Input event listeners ---
    setup_input_handlers(&canvas, &state);

    // --- Control event listeners ---
    setup_control_handlers(&document, &state);

    // --- Animation loop ---
    let perf = window.performance().expect("no performance API");
    let last_time = Rc::new(RefCell::new(perf.now()));

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    let state_loop = state.clone();
    let last_time_loop = last_time.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let now = perf.now();
        let dt_ms = now - *last_time_loop.borrow();
        *last_time_loop.borrow_mut() = now;
        let dt_s = (dt_ms / 1000.0) as f32;

        {
            let mut s = state_loop.borrow_mut();
            s.last_frame_ms = dt_ms;
            s.fps = if dt_ms > 0.0 { 1000.0 / dt_ms } else { 0.0 };

            // Fixed timestep with capped accumulator ("Fix Your Timestep!" pattern).
            //
            // Accumulate real elapsed time, then drain it in fixed sim_dt chunks.
            // The cap prevents death spiral: if we fall behind, we drop time debt
            // rather than trying to catch up (which would make things worse).
            // The sim just runs slower than real-time at high N — visually fine.
            if !s.paused {
                let dt = s.sim.dt;

                // Accumulate real time (scaled by speed multiplier)
                s.accumulator += dt_s as f64 * s.speed_multiplier;

                // Cap at 10 steps worth — prevents spiral on slow frames or tab-away
                let max_accumulator = dt * 10.0;
                if s.accumulator > max_accumulator {
                    s.accumulator = max_accumulator;
                }

                // Drain in fixed-size chunks.
                while s.accumulator >= dt {
                    s.sim.step_once();
                    s.accumulator -= dt;
                }
            }

            // Update camera
            let diag = sim_core::diagnostics::compute_fast(
                &s.sim.particles,
                s.sim.sim_time,
                s.sim.step,
            );
            let com = diag.center_of_mass;
            s.camera
                .set_target(glam::Vec3::new(com[0] as f32, com[1] as f32, com[2] as f32));
            s.camera.update(dt_s);

            // Render
            render_frame(&mut s);

            // Update stats in DOM (throttled — DOM writes are expensive in WASM)
            s.frames_since_stats += 1;
            if s.frames_since_stats >= 15 {
                s.frames_since_stats = 0;
                update_stats_dom(&s);
            }
        }

        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());
}

fn render_frame(s: &mut AppState) {
    let camera_uniform = s.camera.build_uniform();

    // Write camera uniform
    s.queue.write_buffer(
        s.particle_pipeline.camera_buffer(),
        0,
        bytemuck::bytes_of(&camera_uniform),
    );

    // Update particle instances
    let n = s.sim.particles.count;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            [
                s.sim.particles.x[i] as f32,
                s.sim.particles.y[i] as f32,
                s.sim.particles.z[i] as f32,
            ]
        })
        .collect();
    let masses: Vec<f32> = s.sim.particles.mass.iter().map(|&m| m as f32).collect();
    s.particle_pipeline
        .update_instances(&s.queue, &s.device, &positions, &masses);

    let surface_tex = match s.surface.get_current_texture() {
        Ok(t) => t,
        Err(_) => {
            s.surface.configure(&s.device, &s.surface_config);
            return;
        }
    };
    let view = surface_tex
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
                view: &s.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        s.axes_pipeline
            .draw(&mut pass, s.particle_pipeline.camera_bind_group());
        s.particle_pipeline.draw(&mut pass);
    }

    s.queue.submit(Some(encoder.finish()));
    surface_tex.present();
}

// ---------------------------------------------------------------------------
// Input handlers
// ---------------------------------------------------------------------------

fn setup_input_handlers(canvas: &HtmlCanvasElement, state: &Rc<RefCell<AppState>>) {
    // Mouse down
    {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
            if e.button() == 0 {
                let mut st = s.borrow_mut();
                st.dragging = true;
                st.last_cursor = [e.client_x() as f64, e.client_y() as f64];
            }
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Mouse up — listen on document so releasing outside the canvas still clears drag state
    {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
            s.borrow_mut().dragging = false;
        }) as Box<dyn FnMut(_)>);
        let document = web_sys::window().unwrap().document().unwrap();
        document
            .add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Mouse move
    {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
            let mut st = s.borrow_mut();
            let x = e.client_x() as f64;
            let y = e.client_y() as f64;
            if st.dragging {
                let dx = (x - st.last_cursor[0]) as f32;
                let dy = (y - st.last_cursor[1]) as f32;
                st.camera.desired_yaw -= dx * 0.005;
                st.camera.desired_pitch += dy * 0.005;
                st.camera.desired_pitch = st.camera.desired_pitch.clamp(-1.55, 1.55);
            }
            st.last_cursor = [x, y];
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Wheel (zoom)
    {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |e: web_sys::WheelEvent| {
            e.prevent_default();
            let mut st = s.borrow_mut();
            let delta = e.delta_y();
            let scroll = if delta.abs() > 50.0 {
                // Pixel deltas (trackpad)
                -(delta as f32) * 0.002
            } else {
                // Line deltas (mouse wheel)
                -(delta as f32) * 0.05
            };
            st.camera.desired_distance *= 1.0 - scroll;
            st.camera.desired_distance = st.camera.desired_distance.clamp(0.1, 100.0);
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Prevent context menu on canvas
    {
        let cb = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
            e.prevent_default();
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }
}

// ---------------------------------------------------------------------------
// Control handlers (scenario, algorithm, particle count, speed, pause)
// ---------------------------------------------------------------------------

fn setup_control_handlers(document: &web_sys::Document, state: &Rc<RefCell<AppState>>) {
    // Scenario change
    if let Some(el) = document.get_element_by_id("scenario") {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let mut st = s.borrow_mut();
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                if let Some(val) = read_select_value(&doc, "scenario") {
                    st.scenario_name = val;
                    st.sim = create_sim(&st.scenario_name, st.particle_count, &st.algorithm_name);
                }
            }
        }) as Box<dyn FnMut(_)>);
        el.add_event_listener_with_callback("change", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Algorithm change
    if let Some(el) = document.get_element_by_id("algorithm") {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let mut st = s.borrow_mut();
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                if let Some(val) = read_select_value(&doc, "algorithm") {
                    st.algorithm_name = val;
                    st.sim = create_sim(&st.scenario_name, st.particle_count, &st.algorithm_name);
                }
            }
        }) as Box<dyn FnMut(_)>);
        el.add_event_listener_with_callback("change", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Particle count slider
    if let Some(el) = document.get_element_by_id("particle-count") {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let mut st = s.borrow_mut();
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                if let Some(val) = read_range_value(&doc, "particle-count") {
                    st.particle_count = val as usize;
                    st.sim = create_sim(&st.scenario_name, st.particle_count, &st.algorithm_name);
                }
            }
        }) as Box<dyn FnMut(_)>);
        el.add_event_listener_with_callback("input", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Speed slider
    if let Some(el) = document.get_element_by_id("speed") {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let mut st = s.borrow_mut();
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                if let Some(val) = read_range_value(&doc, "speed") {
                    st.speed_multiplier = val as f64;
                }
            }
        }) as Box<dyn FnMut(_)>);
        el.add_event_listener_with_callback("input", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Pause button
    if let Some(el) = document.get_element_by_id("pause-btn") {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let mut st = s.borrow_mut();
            st.paused = !st.paused;
        }) as Box<dyn FnMut(_)>);
        el.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }
}

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

fn read_select_value(doc: &web_sys::Document, id: &str) -> Option<String> {
    doc.get_element_by_id(id)
        .and_then(|el| el.dyn_into::<web_sys::HtmlSelectElement>().ok())
        .map(|sel| sel.value())
}

fn read_range_value(doc: &web_sys::Document, id: &str) -> Option<i32> {
    doc.get_element_by_id(id)
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok())
        .and_then(|input| input.value().parse().ok())
}

fn update_stats_dom(s: &AppState) {
    let window = match web_sys::window() {
        Some(w) => w,
        None => return,
    };
    let document = match window.document() {
        Some(d) => d,
        None => return,
    };

    set_text(&document, "stat-fps", &format!("{:.0}", s.fps));
    set_text(
        &document,
        "stat-particles",
        &format!("{}", s.sim.particles.count),
    );
    set_text(
        &document,
        "stat-time",
        &format!("{:.3}", s.sim.sim_time),
    );

    // For small N, compute full diagnostics (O(N²) potential) to show total energy
    // and virial ratio — these are the quantities the book discusses. For large N,
    // fall back to O(N) kinetic-only to avoid killing frame rate.
    let softening = 0.05;
    if s.sim.particles.count <= 500 {
        let diag = sim_core::diagnostics::compute(
            &s.sim.particles,
            softening,
            s.sim.sim_time,
            s.sim.step,
        );
        set_text(
            &document,
            "stat-energy",
            &format!("{:.6}", diag.total_energy),
        );
        set_text(
            &document,
            "stat-virial",
            &format!("{:.3}", diag.virial_ratio),
        );
    } else {
        let diag =
            sim_core::diagnostics::compute_fast(&s.sim.particles, s.sim.sim_time, s.sim.step);
        set_text(
            &document,
            "stat-energy",
            &format!("K={:.4}", diag.kinetic_energy),
        );
        set_text(&document, "stat-virial", "—");
    }

    // Update pause button text
    if let Some(el) = document.get_element_by_id("pause-btn") {
        if let Some(btn) = el.dyn_ref::<web_sys::HtmlElement>() {
            btn.set_inner_text(if s.paused { "Resume" } else { "Pause" });
        }
    }
}

fn set_text(doc: &web_sys::Document, id: &str, text: &str) {
    if let Some(el) = doc.get_element_by_id(id) {
        el.set_text_content(Some(text));
    }
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}
