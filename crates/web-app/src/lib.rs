// This crate only compiles for wasm32 targets (uses wasm_bindgen, web_sys).
// The cfg gate prevents native-target clippy/check from failing on web-only APIs.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use render_core::axes::AxesPipeline;
use render_core::bloom::{self, BloomPipeline, DEPTH_FORMAT, HDR_FORMAT};
use render_core::camera::OrbitalCamera;
use render_core::color;
use render_core::particles::ParticlePipeline;
use render_core::tonemap::ToneMapPipeline;

use sim_core::barnes_hut::BarnesHut;
use sim_core::gravity::{BruteForce, GravitySolver, NoGravity};
use sim_core::integrator::{Integrator, LeapfrogKDK};
use sim_core::particle::Particles;
use sim_core::scenario::Scenario;
use sim_core::scenarios::cold_collapse::ColdCollapse;
use sim_core::scenarios::evrard_collapse::EvrardCollapse;
use sim_core::scenarios::galaxy_collision::GalaxyCollision;
use sim_core::scenarios::kelvin_helmholtz::KelvinHelmholtz;
use sim_core::scenarios::plummer_sphere::PlummerSphere;
use sim_core::scenarios::protoplanetary::Protoplanetary;
use sim_core::scenarios::sedov_blast::SedovBlast;
use sim_core::scenarios::sod_shock::SodShockTube;
use sim_core::scenarios::two_body::TwoBody;
use sim_core::sph::boundary::Boundary;
use sim_core::sph::solver::{self, SphSolver};

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------

struct SimState {
    particles: Particles,
    gravity: Box<dyn GravitySolver>,
    integrator: LeapfrogKDK,
    sph_solver: Option<SphSolver>,
    boundary: Boundary,
    dt: f64,
    sim_time: f64,
    step: u64,
}

impl SimState {
    fn step_once(&mut self) {
        let step_dt = self.dt;
        if let Some(sph) = &mut self.sph_solver {
            let dt_next = solver::step_with_sph(
                &mut self.particles,
                self.gravity.as_ref(),
                sph,
                step_dt,
            );
            self.boundary.apply(&mut self.particles);
            self.dt = dt_next.min(self.dt * 1.5);
        } else {
            self.integrator
                .step(&mut self.particles, self.gravity.as_ref(), step_dt);
        }
        self.sim_time += step_dt;
        self.step += 1;
    }
}

struct AppState {
    // wgpu
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    canvas: HtmlCanvasElement,

    // render-core pipelines
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

    // input (mouse)
    dragging: bool,
    last_cursor: [f64; 2],

    // input (touch)
    touch: TouchState,

    // resize
    pending_resize: bool,

    // scratch buffers (reused across frames to avoid per-frame allocation)
    scratch_positions: Vec<[f32; 3]>,
    scratch_masses: Vec<f32>,
    cached_colors: Vec<[f32; 4]>,
}

#[derive(Default)]
struct TouchState {
    primary: Option<TrackedTouch>,
    secondary: Option<TrackedTouch>,
    pinch_distance: Option<f64>,
}

struct TrackedTouch {
    id: i32,
    pos: [f64; 2],
}


struct SimSetup {
    particles: Particles,
    dt: f64,
    gravity: Box<dyn GravitySolver>,
    sph_solver: Option<SphSolver>,
    boundary: Boundary,
}

fn create_sim(scenario: &str, n: usize, algorithm: &str) -> SimState {
    let setup = match scenario {
        "two-body" => {
            let s = TwoBody { eccentricity: 0.5, ..Default::default() };
            let p = s.generate();
            let soft = s.suggested_softening();
            SimSetup {
                particles: p, dt: 0.005, gravity: make_gravity(algorithm, soft),
                sph_solver: None, boundary: Boundary::None,
            }
        }
        "cold-collapse" => {
            let s = ColdCollapse { n, ..Default::default() };
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: make_gravity(algorithm, s.suggested_softening()),
                sph_solver: None, boundary: Boundary::None,
            }
        }
        "galaxy-collision" => {
            let s = GalaxyCollision { n_per_galaxy: n / 2, ..Default::default() };
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: make_gravity(algorithm, s.suggested_softening()),
                sph_solver: None, boundary: Boundary::None,
            }
        }
        "sod-shock" => {
            let s = SodShockTube { nx_left: 40, nyz: 4, ..Default::default() };
            let yz = s.yz_extent();
            let x_ext = s.x_extent;
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: Box::new(NoGravity),
                sph_solver: Some(SphSolver::new()),
                boundary: Boundary::Reflective {
                    bounds: [(-x_ext * 2.0, x_ext * 2.0), (0.0, yz), (0.0, yz)],
                },
            }
        }
        "sedov-blast" => {
            let s = SedovBlast { n_particles: n.min(2000), ..Default::default() };
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: Box::new(NoGravity),
                sph_solver: Some(SphSolver::new()),
                boundary: Boundary::None,
            }
        }
        "evrard-collapse" => {
            let s = EvrardCollapse { n_particles: n, ..Default::default() };
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: Box::new(BruteForce::new(s.suggested_softening())),
                sph_solver: Some(SphSolver::new()),
                boundary: Boundary::None,
            }
        }
        "kelvin-helmholtz" => {
            let s = KelvinHelmholtz::default();
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: Box::new(NoGravity),
                sph_solver: Some(SphSolver::new()),
                boundary: Boundary::None,
            }
        }
        "protoplanetary" => {
            let s = Protoplanetary { n_gas: n, ..Default::default() };
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: Box::new(BruteForce::new(s.suggested_softening())),
                sph_solver: Some(SphSolver::new()),
                boundary: Boundary::None,
            }
        }
        "cold-collapse-gas" => {
            let s = ColdCollapse { n, sph: true, ..Default::default() };
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: Box::new(BruteForce::new(s.suggested_softening())),
                sph_solver: Some(SphSolver::new()),
                boundary: Boundary::None,
            }
        }
        _ => {
            let s = PlummerSphere { n, ..Default::default() };
            let p = s.generate();
            SimSetup {
                particles: p, dt: s.suggested_dt(), gravity: make_gravity(algorithm, s.suggested_softening()),
                sph_solver: None, boundary: Boundary::None,
            }
        }
    };

    let mut p = setup.particles;
    p.clear_accelerations();
    setup.gravity.compute_accelerations(&mut p);

    let mut sph_solver = setup.sph_solver;
    if let Some(sph) = &mut sph_solver {
        let _ = sph.compute(&mut p);
    }

    SimState {
        particles: p,
        gravity: setup.gravity,
        integrator: LeapfrogKDK,
        sph_solver,
        boundary: setup.boundary,
        dt: setup.dt,
        sim_time: 0.0,
        step: 0,
    }
}

fn make_gravity(algorithm: &str, softening: f64) -> Box<dyn GravitySolver> {
    match algorithm {
        "brute-force" => Box::new(BruteForce::new(softening)),
        _ => Box::new(BarnesHut::new(softening, 0.5)),
    }
}

fn camera_distance_for_scenario(scenario: &str) -> f32 {
    match scenario {
        "sod-shock" | "kelvin-helmholtz" => 3.0,
        "protoplanetary" => 8.0,
        "galaxy-collision" => 30.0,
        _ => 5.0,
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

    // Particle and axes pipelines target HDR format
    let particle_pipeline = ParticlePipeline::new(&device, HDR_FORMAT, DEPTH_FORMAT);

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

    let axes_pipeline = AxesPipeline::new(&device, HDR_FORMAT, DEPTH_FORMAT, &camera_bind_group_layout);
    let depth_view = bloom::create_depth_texture(&device, width, height);
    let mut camera = OrbitalCamera::new(width as f32 / height.max(1) as f32);

    let hdr_texture = bloom::create_hdr_texture(&device, width, height);
    let hdr_view = hdr_texture.create_view(&Default::default());
    let hdr_composited = bloom::create_hdr_texture(&device, width, height);
    let hdr_composited_view = hdr_composited.create_view(&Default::default());
    let bloom_pipeline = BloomPipeline::new(&device, &queue, width, height, &hdr_view);
    let tonemap_pipeline = ToneMapPipeline::new(&device, format, &hdr_composited_view);

    // Read initial config from DOM
    let scenario_name = read_select_value(&document, "scenario").unwrap_or("plummer".into());
    let algorithm_name = read_select_value(&document, "algorithm").unwrap_or("barnes-hut".into());
    let particle_count = read_range_value(&document, "particle-count").unwrap_or(5000) as usize;
    let speed = read_range_value(&document, "speed").unwrap_or(1) as f64;

    let sim = create_sim(&scenario_name, particle_count, &algorithm_name);
    camera.desired_distance = camera_distance_for_scenario(&scenario_name);

    // Pre-compute colors and masses
    let n = sim.particles.count;
    let cached_colors: Vec<[f32; 4]> = sim.particles.particle_type.iter().enumerate()
        .map(|(idx, &pt)| color::particle_color(pt, sim.particles.internal_energy[idx] as f32))
        .collect();
    let scratch_masses: Vec<f32> = sim.particles.mass.iter().map(|&m| m as f32).collect();

    let state = Rc::new(RefCell::new(AppState {
        device,
        queue,
        surface,
        surface_config,
        canvas: canvas.clone(),
        particle_pipeline,
        axes_pipeline,
        bloom_pipeline,
        tonemap_pipeline,
        hdr_texture,
        hdr_view,
        hdr_composited,
        hdr_composited_view,
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
        touch: TouchState::default(),
        pending_resize: false,
        scratch_positions: Vec::with_capacity(n),
        scratch_masses,
        cached_colors,
    }));

    // --- Input event listeners ---
    setup_input_handlers(&canvas, &state);
    setup_touch_handlers(&canvas, &state);
    setup_resize_observer(&canvas, &state);

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

            // Handle pending resize before rendering
            if s.pending_resize {
                handle_resize(&mut s);
                s.pending_resize = false;
            }

            s.last_frame_ms = dt_ms;
            s.fps = if dt_ms > 0.0 { 1000.0 / dt_ms } else { 0.0 };

            // Fixed timestep with capped accumulator ("Fix Your Timestep!" pattern).
            //
            // Accumulate real elapsed time, then drain it in fixed sim_dt chunks.
            // The cap prevents death spiral: if we fall behind, we drop time debt
            // rather than trying to catch up (which would make things worse).
            // The sim just runs slower than real-time at high N — visually fine.
            if !s.paused {
                let mut dt = s.sim.dt;

                // Accumulate real time (scaled by speed multiplier)
                s.accumulator += dt_s as f64 * s.speed_multiplier;

                // Cap at 10 steps worth — prevents spiral on slow frames or tab-away
                let max_accumulator = dt * 10.0;
                if s.accumulator > max_accumulator {
                    s.accumulator = max_accumulator;
                }

                // Drain in fixed-size chunks.
                let mut steps_taken = 0u32;
                while s.accumulator >= dt {
                    s.sim.step_once();
                    s.accumulator -= dt;
                    dt = s.sim.dt;
                    steps_taken += 1;
                }

                // Update gas particle colors (temperature changes each step)
                if steps_taken > 0 && s.sim.sph_solver.is_some() {
                    let n = s.sim.particles.count;
                    for idx in 0..n {
                        if s.sim.particles.particle_type[idx] == 4 {
                            s.cached_colors[idx] = color::particle_color(
                                4, s.sim.particles.internal_energy[idx] as f32
                            );
                        }
                    }
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

/// Refresh cached colors and masses after a scenario change.
fn refresh_sim_cache(s: &mut AppState) {
    s.cached_colors = s.sim.particles.particle_type.iter().enumerate()
        .map(|(idx, &pt)| color::particle_color(pt, s.sim.particles.internal_energy[idx] as f32))
        .collect();
    s.scratch_masses = s.sim.particles.mass.iter().map(|&m| m as f32).collect();
    s.scratch_positions = Vec::with_capacity(s.sim.particles.count);
}

fn render_frame(s: &mut AppState) {
    let camera_uniform = s.camera.build_uniform();

    // Write camera uniform
    s.queue.write_buffer(
        s.particle_pipeline.camera_buffer(),
        0,
        bytemuck::bytes_of(&camera_uniform),
    );

    // Update particle positions into scratch buffer (reused across frames)
    let n = s.sim.particles.count;
    s.scratch_positions.clear();
    for i in 0..n {
        s.scratch_positions.push([
            s.sim.particles.x[i] as f32,
            s.sim.particles.y[i] as f32,
            s.sim.particles.z[i] as f32,
        ]);
    }

    // Masses and colors are invariant within a run — precomputed at sim creation
    s.particle_pipeline.update_instances(
        &s.queue, &s.device,
        &s.scratch_positions, &s.scratch_masses, &s.cached_colors,
    );

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

    // Pass 1: Render axes + particles to HDR texture
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("HDR scene pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &s.hdr_view,
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

    // Pass 2: Bloom → hdr_composited
    s.bloom_pipeline
        .render(&mut encoder, &s.hdr_composited_view);

    // Pass 3: Tone map → LDR surface
    s.tonemap_pipeline.render(&mut encoder, &view);

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
                st.camera.apply_orbit_delta(dx, dy);
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
            st.camera.apply_zoom(1.0 - scroll);
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
// Resize handling
// ---------------------------------------------------------------------------

fn handle_resize(s: &mut AppState) {
    let w = s.canvas.client_width() as u32;
    let h = s.canvas.client_height() as u32;
    if w == 0 || h == 0 {
        return;
    }
    if w == s.surface_config.width && h == s.surface_config.height {
        return;
    }
    s.canvas.set_width(w);
    s.canvas.set_height(h);
    s.surface_config.width = w;
    s.surface_config.height = h;
    s.surface.configure(&s.device, &s.surface_config);

    // Recreate all size-dependent render targets (matches native-app resize)
    s.depth_view = bloom::create_depth_texture(&s.device, w, h);
    s.hdr_texture = bloom::create_hdr_texture(&s.device, w, h);
    s.hdr_view = s.hdr_texture.create_view(&Default::default());
    s.hdr_composited = bloom::create_hdr_texture(&s.device, w, h);
    s.hdr_composited_view = s.hdr_composited.create_view(&Default::default());
    s.bloom_pipeline.resize(&s.device, w, h);
    s.bloom_pipeline.update_source(&s.device, &s.hdr_view);
    s.tonemap_pipeline.update_source(&s.device, &s.hdr_composited_view);

    s.camera.set_aspect_ratio(w as f32 / h.max(1) as f32);
}

fn setup_resize_observer(canvas: &HtmlCanvasElement, state: &Rc<RefCell<AppState>>) {
    let s = state.clone();
    let cb = Closure::wrap(Box::new(move |_entries: js_sys::Array| {
        s.borrow_mut().pending_resize = true;
    }) as Box<dyn FnMut(js_sys::Array)>);

    let observer = web_sys::ResizeObserver::new(cb.as_ref().unchecked_ref())
        .expect("Failed to create ResizeObserver");
    observer.observe(canvas);
    cb.forget();
    // Leak the observer so it stays alive for the lifetime of the page.
    std::mem::forget(observer);
}

// ---------------------------------------------------------------------------
// Touch handlers
// ---------------------------------------------------------------------------

fn add_non_passive_listener(
    target: &web_sys::EventTarget,
    event: &str,
    cb: Closure<dyn FnMut(web_sys::TouchEvent)>,
) {
    let opts = web_sys::AddEventListenerOptions::new();
    opts.set_passive(false);
    target
        .add_event_listener_with_callback_and_add_event_listener_options(
            event,
            cb.as_ref().unchecked_ref(),
            &opts,
        )
        .unwrap();
    cb.forget();
}

fn touch_distance(a: &TrackedTouch, b: &TrackedTouch) -> f64 {
    let dx = a.pos[0] - b.pos[0];
    let dy = a.pos[1] - b.pos[1];
    (dx * dx + dy * dy).sqrt()
}

fn setup_touch_handlers(canvas: &HtmlCanvasElement, state: &Rc<RefCell<AppState>>) {
    // touchstart
    {
        let s = state.clone();
        add_non_passive_listener(
            canvas,
            "touchstart",
            Closure::wrap(Box::new(move |e: web_sys::TouchEvent| {
                e.prevent_default();
                let mut st = s.borrow_mut();
                let touches = e.changed_touches();
                for i in 0..touches.length() {
                    let t = touches.get(i).unwrap();
                    let tt = TrackedTouch {
                        id: t.identifier(),
                        pos: [t.client_x() as f64, t.client_y() as f64],
                    };
                    if st.touch.primary.is_none() {
                        st.touch.primary = Some(tt);
                    } else if st.touch.secondary.is_none()
                        && st.touch.primary.as_ref().map(|p| p.id) != Some(tt.id)
                    {
                        // Compute initial pinch distance
                        st.touch.pinch_distance =
                            Some(touch_distance(st.touch.primary.as_ref().unwrap(), &tt));
                        st.touch.secondary = Some(tt);
                    }
                }
            })),
        );
    }

    // touchmove
    {
        let s = state.clone();
        add_non_passive_listener(
            canvas,
            "touchmove",
            Closure::wrap(Box::new(move |e: web_sys::TouchEvent| {
                e.prevent_default();
                let mut st = s.borrow_mut();
                let touches = e.changed_touches();

                for i in 0..touches.length() {
                    let t = touches.get(i).unwrap();
                    let id = t.identifier();
                    let pos = [t.client_x() as f64, t.client_y() as f64];
                    if st.touch.primary.as_ref().map(|p| p.id) == Some(id) {
                        let old = st.touch.primary.as_ref().unwrap().pos;
                        st.touch.primary.as_mut().unwrap().pos = pos;
                        // Single-finger orbit (only if no pinch active)
                        if st.touch.secondary.is_none() {
                            let dx = (pos[0] - old[0]) as f32;
                            let dy = (pos[1] - old[1]) as f32;
                            st.camera.apply_orbit_delta(dx, dy);
                        }
                    } else if st.touch.secondary.as_ref().map(|p| p.id) == Some(id) {
                        st.touch.secondary.as_mut().unwrap().pos = pos;
                    }
                }

                // Two-finger pinch zoom
                if let (Some(p), Some(s2)) =
                    (st.touch.primary.as_ref(), st.touch.secondary.as_ref())
                {
                    let new_dist = touch_distance(p, s2);
                    if let Some(old_dist) = st.touch.pinch_distance {
                        if old_dist > 1.0 {
                            let ratio = new_dist / old_dist;
                            st.camera.apply_zoom(1.0 / ratio as f32);
                        }
                    }
                    st.touch.pinch_distance = Some(new_dist);
                }
            })),
        );
    }

    // touchend / touchcancel
    for event_name in &["touchend", "touchcancel"] {
        let s = state.clone();
        add_non_passive_listener(
            canvas,
            event_name,
            Closure::wrap(Box::new(move |e: web_sys::TouchEvent| {
                e.prevent_default();
                let mut st = s.borrow_mut();
                let touches = e.changed_touches();
                for i in 0..touches.length() {
                    let id = touches.get(i).unwrap().identifier();
                    if st.touch.primary.as_ref().map(|p| p.id) == Some(id) {
                        // Promote secondary to primary if it exists
                        st.touch.primary = st.touch.secondary.take();
                        st.touch.pinch_distance = None;
                    } else if st.touch.secondary.as_ref().map(|p| p.id) == Some(id) {
                        st.touch.secondary = None;
                        st.touch.pinch_distance = None;
                    }
                }
            })),
        );
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
                    st.camera.desired_distance = camera_distance_for_scenario(&st.scenario_name);
                    refresh_sim_cache(&mut st);
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
                    refresh_sim_cache(&mut st);
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
                    refresh_sim_cache(&mut st);
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
