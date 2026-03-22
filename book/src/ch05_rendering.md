# Rendering Particles

Visualization is the most underrated tool in computational physics. A simulation can pass every numerical test and still be wrong in ways that are instantly obvious when you see the particles move. A cluster that should be virialized but is visibly breathing. A disk with unexpected spiral structure. Rendering is a debugger with a bandwidth that no log file can match.

This chapter covers the [wgpu](https://wgpu.rs/) rendering pipeline, instanced billboard particles with WGSL shaders, orbital camera mathematics, the crate architecture that shares rendering code between native and web, and the constraints of targeting [WebGPU](https://en.wikipedia.org/wiki/WebGPU) in the browser.

## The wgpu Pipeline

[wgpu](https://github.com/gfx-rs/wgpu) is Rust's cross-platform [WebGPU](https://en.wikipedia.org/wiki/WebGPU) implementation. On macOS it talks to [Metal](https://en.wikipedia.org/wiki/Metal_(API)), on Windows to DX12/Vulkan, on Linux to Vulkan, and in the browser to the WebGPU API. One set of shaders, every platform.

GPU initialization follows a fixed progression: **Instance** (backend entry point) -> **Adapter** (physical GPU) -> **Device + Queue** (logical device and command submission) -> **Surface** (render target: a window or `<canvas>`). Each frame creates a command encoder, begins a render pass, issues draw calls, submits, and presents. The structure is boilerplate — the interesting part is what happens inside the render pass.

## Instanced Billboard Rendering

We use **instanced rendering**: define a single quad, then draw $N$ instances of it, each positioned at a particle's location. One draw call, $N$ particles.

### The Billboard Trick

A [billboard](https://en.wikipedia.org/wiki/Billboard_(computer_graphics)) is a flat quad that always faces the camera. Instead of rendering spheres, we render camera-facing quads and paint a soft circle in the fragment shader. The key is constructing the quad in world space using the camera's right and up vectors. Given a particle at position $\vec{p}$ and a quad vertex with offset $(o_x, o_y)$:

$$\vec{v}_\text{world} = \vec{p} + o_x \cdot s \cdot \hat{r}_\text{cam} + o_y \cdot s \cdot \hat{u}_\text{cam}$$

where $s$ is the particle's on-screen size, $\hat{r}_\text{cam}$ is the camera's right vector, and $\hat{u}_\text{cam}$ is its up vector. This places the quad in a plane perpendicular to the camera's view direction — it faces the camera regardless of where the particle is.

### The Particle Shader

The WGSL shader is embedded in [`crates/render-core/src/particles.rs`](https://github.com/jcorbettfrank/gravis/blob/m4/crates/render-core/src/particles.rs):

```wgsl
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
    if dist > 1.0 { discard; }
    let alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    let color = vec3<f32>(0.7, 0.85, 1.0) * in.brightness;
    return vec4<f32>(color * alpha, alpha);
}
```

The `VertexInput` takes a per-vertex `offset` (quad corner), per-instance `position` and `mass`. The camera uniform provides `view_proj`, `camera_right`, and `camera_up`. Let's walk through the key pieces.

**Sizing.** On-screen size scales as $M^{1/3}$: `pow(mass * 1000.0, 0.333)`. The `* 1000` rescales from N-body units (individual mass $\sim 1/N$) to a visible range. The $1/3$ exponent matches the physical radius-mass relation at constant density — a particle 8x more massive appears only 2x larger.

**Billboarding.** The world position offsets the particle by camera right/up vectors scaled by the quad corner offsets ($\pm 0.5$).

**Brightness.** Scales as $M^{0.5}$, clamped to $[0.3, 1.0]$ so even light particles are visible.

**Fragment shader.** Computes radial distance from quad center, applies [`smoothstep`](https://en.wikipedia.org/wiki/Smoothstep) for a smooth falloff, and discards fragments outside the unit circle. The output uses premultiplied alpha (`color * alpha`) for correct additive blending.

### Per-Vertex vs Per-Instance Data

The GPU receives two kinds of data ([`gpu_types.rs`](https://github.com/jcorbettfrank/gravis/blob/m4/crates/render-core/src/gpu_types.rs)): per-vertex `QuadVertex` (just a 2D offset, 4 entries, never changes) and per-instance `GpuParticle` (position + mass, $N$ entries, rewritten every frame). The draw call `draw_indexed(0..6, 0, 0..N)` draws two triangles for each of $N$ instances.

```rust
pub struct QuadVertex { pub offset: [f32; 2] }
pub struct GpuParticle { pub position: [f32; 3], pub mass: f32 }
```

### The f64-to-f32 Boundary

The simulation runs in f64 (Chapter 1 explained why). The GPU works in f32. The conversion happens when building the instance buffer — each particle's position and mass are cast to `f32`. This is safe for rendering: f32 precision of ~7 digits means positions are accurate to $\sim 10^{-7}$, which is sub-pixel at cluster scales. The loss of precision happens only in the visual output, never in the simulation.

### Additive Blending

The particle pipeline uses **additive blending** (`SrcAlpha + One`): when particles overlap, their colors are summed rather than composited. Dense regions glow brighter, naturally revealing cluster structure without post-processing. Depth writes are disabled because additive blending is order-independent.

<div class="physics-note">

**Additive blending and physical luminosity.**
Additive blending models optically thin emission — light from background stars passes through foreground stars unblocked. This is correct for star clusters where stars are tiny compared to inter-stellar distances. The visual result — bright cores, diffuse halos — matches long-exposure astrophotography. Optically thick media (gas, dust) would need alpha compositing with depth sorting.

</div>

## Camera Mathematics

The orbital camera ([`crates/render-core/src/camera.rs`](https://github.com/jcorbettfrank/gravis/blob/m4/crates/render-core/src/camera.rs)) lets the user orbit around a target point — typically the center of mass of the particle system. It has four parameters: target point, distance, yaw (horizontal angle), and pitch (vertical angle).

### Eye Position from Spherical Coordinates

The camera's eye position is computed from [spherical coordinates](https://en.wikipedia.org/wiki/Spherical_coordinate_system) relative to the target:

$$x = d \cos(\text{pitch}) \sin(\text{yaw})$$
$$y = d \sin(\text{pitch})$$
$$z = d \cos(\text{pitch}) \cos(\text{yaw})$$

$$\vec{p}_\text{eye} = \vec{p}_\text{target} + (x, y, z)$$

The pitch is clamped to $\pm 1.55$ radians ($\approx \pm 89^\circ$) to prevent the degenerate case where the view direction aligns with the up vector — this would cause the view matrix to flip or become undefined.

In code:

```rust
pub fn eye_position(&self) -> Vec3 {
    let x = self.distance * self.pitch.cos() * self.yaw.sin();
    let y = self.distance * self.pitch.sin();
    let z = self.distance * self.pitch.cos() * self.yaw.cos();
    self.target + Vec3::new(x, y, z)
}
```

### View and Projection Matrices

The [view matrix](https://en.wikipedia.org/wiki/Camera_matrix) is constructed with `look_at_rh` (right-handed coordinate system, which is the convention for WebGPU/wgpu):

```rust
let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
let proj = Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.near, self.far);
let view_proj = proj * view;
```

The combined view-projection matrix transforms from world space to clip space. The perspective projection uses a 45-degree vertical FOV and a near/far range of 0.001 to 1000.

### Extracting Camera Vectors for Billboarding

The billboard shader needs the camera's right and up vectors in world space. These live in the view matrix rows:

```rust
let camera_right = Vec3::new(view.col(0).x, view.col(1).x, view.col(2).x);
let camera_up = Vec3::new(view.col(0).y, view.col(1).y, view.col(2).y);
```

These vectors and the view-projection matrix are packed into a `CameraUniform` and uploaded to the GPU every frame.

### Exponential Smoothing

Raw mouse input produces jerky camera motion. We smooth using [exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing):

$$t = 1 - e^{-\lambda \cdot \Delta t}$$

where $\lambda$ is the smoothing factor (10.0). Each camera parameter is interpolated toward its desired value with $\text{lerp}(\text{current}, \text{desired}, t)$. The exponential form is frame-rate independent: at 30 fps or 144 fps, the camera converges at the same physical rate.

```rust
pub fn update(&mut self, dt: f32) {
    let t = 1.0 - (-self.smoothing * dt).exp();
    self.yaw = lerp(self.yaw, self.desired_yaw, t);
    self.pitch = lerp(self.pitch, self.desired_pitch, t);
    self.distance = lerp(self.distance, self.desired_distance, t);
    self.target = self.target.lerp(self.desired_target, t);
}
```

## The render-core / native-app / web-app Split

The GPU pipeline, shaders, camera math, and data types are identical between native and web — only the event loop and input handling differ. So the rendering code is split into three crates.

**render-core** ([`crates/render-core/`](https://github.com/jcorbettfrank/gravis/blob/m4/crates/render-core/src/lib.rs)) — depends only on wgpu and glam, no windowing:
- `ParticlePipeline` — the instanced billboard pipeline, shader, and instance buffer management
- `AxesPipeline` — coordinate axes rendered as colored lines
- `OrbitalCamera` — the camera model and uniform construction
- `GpuParticle`, `CameraUniform`, `QuadVertex` — GPU data types with `bytemuck::Pod` for zero-copy upload

**native-app** ([`crates/native-app/`](https://github.com/jcorbettfrank/gravis/blob/m4/crates/native-app/src/main.rs)) is the macOS desktop application:
- [winit](https://github.com/rust-windowing/winit) event loop and window management
- [egui](https://github.com/emilk/egui) HUD overlay for controls and diagnostics
- A dedicated simulation thread that communicates with the render thread via `mpsc` channels
- `NativeCamera` wraps `OrbitalCamera` and translates winit events (mouse drag, scroll wheel) into camera parameter changes

**web-app** ([`crates/web-app/`](https://github.com/jcorbettfrank/gravis/blob/m4/crates/web-app/src/lib.rs)) is the WASM browser target:
- `wasm-bindgen` entry point, no winit
- DOM event listeners (mousedown, mousemove, wheel) for camera input
- HTML control elements (dropdowns, sliders, buttons) for simulation parameters
- `requestAnimationFrame` loop for rendering and simulation stepping

The dependency structure looks like this:

```
sim-core  (pure simulation, no rendering)
   |
   v
render-core  (wgpu pipelines + camera, no windowing)
   |           |
   v           v
native-app   web-app
(winit+egui) (wasm-bindgen+DOM)
```

Both leaf crates depend on `sim-core` and `render-core` but never on each other. Adding a new platform means writing a thin crate that wires up the platform's event loop to the same `render-core` pipelines.

## Targeting the Browser with WebGPU

### WebGPU vs WebGL2

We target [WebGPU](https://en.wikipedia.org/wiki/WebGPU), not [WebGL2](https://en.wikipedia.org/wiki/WebGL). WebGPU maps cleanly to Metal/Vulkan/DX12, uses the [WGSL](https://www.w3.org/TR/WGSL/) shading language (same shaders everywhere), and provides compute shader support for future GPU-accelerated force computation. WebGL2 would require porting WGSL shaders to GLSL. Since wgpu targets WebGPU natively, the web build reuses the exact same shader source and pipeline configuration — zero porting. WebGPU is available in Chrome 113+, Edge 113+, and Safari 18+.

### Rust to WASM

The web build uses [wasm-pack](https://rustwasm.github.io/wasm-pack/) and [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) to compile Rust to WebAssembly. The entry point is a `#[wasm_bindgen(start)]` async function that initializes wgpu, sets up pipelines, attaches DOM event listeners, and starts the animation loop. All wgpu calls are identical to native.

### The Single-Threaded Constraint

WASM in the browser is single-threaded — no `std::thread::spawn`, no [rayon](https://github.com/rayon-rs/rayon). This is handled by feature-gating rayon in `sim-core`:

```toml
[features]
default = ["parallel"]
parallel = ["rayon"]
```

The Barnes-Hut tree walk has parallel (`#[cfg(feature = "parallel")]`) and sequential code paths. The web-app depends on sim-core with `default-features = false`, disabling rayon:

```toml
sim-core = { path = "../sim-core", default-features = false }
```

The WASM build compiles without rayon and uses sequential force computation. Native gets rayon by default.

### The requestAnimationFrame Loop

Without threads, simulation and rendering share a single `requestAnimationFrame` callback. A naive approach — compute however many steps the elapsed wall-clock time demands — creates a death spiral: a slow frame increases $\Delta t$, which demands more steps, which makes the next frame even slower.

The fix is the [fixed timestep with capped accumulator](https://gafferongames.com/post/fix_your_timestep/) pattern. Real elapsed time accumulates in a bucket, capped at a small multiple of $\Delta t_\text{sim}$ to discard time debt rather than trying to catch up. The bucket is then drained in fixed-size chunks:

```rust
// Accumulate real elapsed time (scaled by speed multiplier)
accumulator += dt_s as f64 * speed;

// Cap at 10 steps worth — prevents spiral on slow frames or tab-away
let max_accumulator = dt * 10.0;
if accumulator > max_accumulator {
    accumulator = max_accumulator;
}

// Drain in fixed-size chunks
while accumulator >= dt {
    sim.step_once();
    accumulator -= dt;
}
```

The cap is the key: if the simulation can't keep up with real-time at high $N$, it simply runs slower than wall-clock. The frame rate stays smooth because rendering is cheap — it's the CPU-side physics that's the bottleneck.

<div class="physics-note">

**Single-threaded performance.**
On WASM, the simulation is CPU-bound much earlier than on native. Without rayon, Barnes-Hut runs sequentially — at $N = 5{,}000$ with $\theta = 0.5$, a single force evaluation takes about 5 ms on WASM versus 2 ms native (single-threaded) versus 0.6 ms native (multi-threaded with rayon on 10 cores). The 10-step accumulator cap means the simulation gracefully slows down rather than freezing. For larger $N$, the user can reduce the speed multiplier to maintain smoother animation.

</div>

## Try It Yourself

Drag to orbit, scroll to zoom. The controls change scenario, particle count, and speed.

<div class="live-demo">
  <iframe src="demos/full.html" width="100%" height="550" loading="lazy"
          title="Interactive N-body simulation"></iframe>
  <p class="demo-fallback" style="display:none">
    <img src="images/m2_plummer.png" alt="Plummer sphere screenshot">
    <em>Live demo requires a WebGPU-enabled browser (Chrome 113+, Edge 113+, Safari 18+).</em>
  </p>
</div>

```bash
# Web: build WASM and serve
cd crates/web-app && wasm-pack build --target web --release
cd www && python3 -m http.server 8080

# Native: desktop renderer
cargo run -p native-app --release -- --scenario plummer -n 5000
```

## Performance Notes

Drawing 10,000 instanced billboards takes well under 1 ms on any recent GPU. The bottleneck is always the simulation, not the rendering.

**GPU-bound vs CPU-bound.** At low $N$, vsync is the limiter. At higher $N$, force computation dominates:

| Regime | Brute-force | Barnes-Hut |
|--------|-------------|------------|
| GPU-bound | $N < 1{,}000$ | $N < 5{,}000$ |
| CPU-bound | $N > 2{,}000$ | $N > 20{,}000$ |

**WASM overhead.** WASM simulation runs roughly 2-3x slower than native single-threaded, due to execution overhead and loss of SIMD auto-vectorization. The rendering itself has no WASM penalty — wgpu calls go straight to the browser's WebGPU implementation.

**Adaptive stepping.** Both native and WASM use adaptive step counts to keep frame rates smooth. If the CPU can't keep up, the simulation clock falls behind wall time rather than freezing.

## Further Reading

- [wgpu](https://wgpu.rs/) — the Rust WebGPU implementation powering our renderer
- [WebGPU specification](https://www.w3.org/TR/webgpu/) — the W3C spec that wgpu implements
- [WGSL specification](https://www.w3.org/TR/WGSL/) — the WebGPU shading language used for our particle and axes shaders
- [Billboard (computer graphics)](https://en.wikipedia.org/wiki/Billboard_(computer_graphics)) — the camera-facing quad technique for particle rendering
- [Instancing (computer graphics)](https://en.wikipedia.org/wiki/Geometry_instancing) — drawing many copies of the same geometry in a single draw call
- [Spherical coordinate system](https://en.wikipedia.org/wiki/Spherical_coordinate_system) — the coordinate system underlying the orbital camera
- [Exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) — the frame-rate-independent smoothing technique for camera input
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) — the Rust-to-JavaScript FFI layer for the web build

## What's Next

We have a simulation engine (Chapters 1-4) and a rendering pipeline (this chapter) that runs on both desktop and browser. The foundation is complete. Chapter 6 moves from toy scenarios to astrophysically motivated ones: generating equilibrium galaxy models with disk, bulge, and halo components, then smashing them together to watch the tidal tails form.
