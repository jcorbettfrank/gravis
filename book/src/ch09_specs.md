# Scenario Specs: Universes as Data

Every scenario so far has been a Rust struct. `GalaxyCollision` knows how to build two galaxies 15 units apart with an impact parameter of 2; `PlummerSphere` knows how to build one cluster. Trying "what if the second galaxy were a dwarf, tilted 60°, falling in from above the disk?" means writing code and recompiling, and the browser demo can only ever run what was compiled into it.

This chapter turns initial conditions into **data**. A scenario spec is a small JSON document that lists building blocks (disk galaxies, star clusters, point masses, gas clouds) and says where each one is and how it moves. Every runner accepts one: the headless CLI, the native renderer, and the browser. The format is also the contract for the next step of the project, **Text-to-Universe**: describe a scenario in plain English, have Claude write the spec, run it, check the physics, and revise.

We'll cover the format, how the simulator fills in the numbers a spec leaves out (timestep, softening, camera framing), and the orbital geometry that decides whether two galaxies throw out spectacular tidal tails or barely notice each other.

## A First Spec

Here is the complete spec for a tilted galaxy flyby ([`specs/tilted-flyby.json`](https://github.com/jcorbettfrank/gravis/blob/m8/specs/tilted-flyby.json)):

```json
{
  "name": "Tilted Flyby",
  "description": "Two Milky Way-like disks, one tilted 60 degrees, fall almost head-on into each other on a bound, nearly radial orbit. They pass through each other near t = 13, fall back and merge by t = 25-30, and both disks fling out a similar 5-10% of their stars: a near-radial plunge has no orbital sweep for either disk's spin to resonate with, so there are no Toomre-style prograde tails.",
  "components": [
    {
      "type": "disk_galaxy",
      "particles": 6000,
      "position": [-7.5, -1, 0],
      "velocity": [0.25, 0, 0]
    },
    {
      "type": "disk_galaxy",
      "particles": 6000,
      "spin_axis": [0, 0.866, 0.5],
      "position": [7.5, 1, 0],
      "velocity": [-0.25, 0, 0]
    }
  ]
}
```

Two galaxies, 15 units apart, approaching each other at a relative speed of 0.5. Every structural parameter (disk scale length, halo mass, bulge size) takes the same default as the `GalaxyModel` from [Chapter 6](ch06_scenarios.md). There is no timestep, no softening length, and no camera distance: the simulator derives all three.

Run it:

```bash
cargo run --release -p headless -- --spec specs/tilted-flyby.json --algorithm barnes-hut --steps 2000
```

```text
Scenario:   Tilted Flyby
Particles:  12000
Algorithm:  barnes-hut
dt:         1.477477e-2
Softening:  5.000000e-2
```

### Units

Specs use the same N-body units as the rest of the simulator, $G = 1$. For galactic scenes the natural reading is:

| Quantity | 1 N-body unit |
|---|---|
| Length | 1 kpc |
| Mass | $10^{10} \, M_\odot$ |
| Velocity | $\sqrt{G M / L} \approx 65.6$ km/s |
| Time | $L / V \approx 14.9$ Myr |

So the default galaxy has $10^{10} \, M_\odot$ of stars and $10^{11} \, M_\odot$ of dark matter, the flyby starts with the galaxies 15 kpc apart, and $t = 50$ is about 750 million years. Nothing in the code depends on this mapping; you could equally read a `plummer` component as a globular cluster in parsecs and solar masses, as long as you pick $G = 1$ consistently.

## Components

A spec has a list of `components`, each tagged by `"type"`. Each component reuses a generator that earlier chapters already built and tested, so the spec layer adds no new physics:

| Type | Generator | Required fields | Typical use |
|---|---|---|---|
| `disk_galaxy` | `GalaxyModel` (exponential disk + Plummer bulge + Hernquist halo, [Chapter 6](ch06_scenarios.md)) | `particles` | Spirals, dwarfs |
| `plummer` | `sample_plummer_particle` ([Chapter 1](ch01_gravity.md)) | `particles` | Star clusters, elliptical-ish blobs |
| `point_mass` | a single particle | `mass` | Black holes, stars, planets |
| `gas_cloud` | uniform SPH sphere ([Chapter 7](ch07_sph.md)) | `particles` | Cloud collapse |

Every component also takes `position` and `velocity` (default: origin, at rest). The full field reference, with every default, lives next to the examples in [`specs/README.md`](https://github.com/jcorbettfrank/gravis/blob/m8/specs/README.md); here we focus on the three ideas that make composition work.

### 1. Each component sits exactly where you put it

A sampled galaxy is never perfectly centered. With a few thousand random particles, the center of mass wanders by a fraction of a scale length and the net momentum is not quite zero. If we just added `position` and `velocity` to every particle, a galaxy placed at $x = -7.5$ would actually sit at $x = -7.4$ and drift slowly. That breaks the promise a spec makes.

So after generating each component, `ScenarioSpec::generate` shifts the component into its *own* center-of-mass frame, then adds the requested position and velocity ([`spec.rs`](https://github.com/jcorbettfrank/gravis/blob/m8/crates/sim-core/src/spec.rs)):

```rust
// Center the component on its own COM so it sits exactly at
// `position` and moves exactly at `velocity`.
let range = start..particles.count;
recenter(&mut particles, range.clone());
let (pos, vel) = (c.position(), c.velocity());
for j in range {
    particles.x[j] += pos[0];
    // ... y, z, vx, vy, vz
}
```

After all components are placed, the whole system is shifted to its center-of-mass frame. The positions you write are therefore *relative*: they fix the separations and relative velocities, and the barycenter ends up at the origin.

### 2. Orientation is one vector

A disk is axisymmetric, so its orientation is completely described by the direction of its angular momentum. That is the `spin_axis` field. `[0, 0, 1]` is a disk in the $xy$-plane rotating counter-clockwise when seen from $+z$; `[0, 0, -1]` is the same disk spinning the other way.

`GalaxyModel` always builds its disk around $+\hat z$. To orient it, we need the rotation $R$ that takes $\hat z$ onto the unit spin axis $\hat n$. Rodrigues' rotation formula, specialized to rotating $\hat z$ onto $\hat n = (x, y, z)$, collapses to a closed form with no trigonometry:

$$R = \begin{pmatrix} 1 - f x^2 & -f x y & x \\ -f x y & 1 - f y^2 & y \\ -x & -y & z \end{pmatrix}, \qquad f = \frac{1}{1 + z}$$

The third column is $\hat n$ itself, which is exactly what "takes $\hat z$ onto $\hat n$" means. The formula breaks down only when $\hat n = -\hat z$ ($f$ diverges), and there any half-turn about a horizontal axis works:

```rust
fn rotation_from_z(axis: [f64; 3]) -> [[f64; 3]; 3] {
    let n = norm(axis);
    let [x, y, z] = [axis[0] / n, axis[1] / n, axis[2] / n];
    if 1.0 + z < 1e-12 {
        // Antiparallel: half-turn about x
        return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    }
    let f = 1.0 / (1.0 + z);
    [
        [1.0 - f * x * x, -f * x * y, x],
        [-f * x * y, 1.0 - f * y * y, y],
        [-x, -y, z],
    ]
}
```

The same matrix rotates both positions and velocities, so the disk's rotation turns with it.

<div class="physics-note">

**Why not Euler angles?**
Three angles (inclination, position angle, and a third for rotation about the spin axis) are the traditional way to orient a galaxy, and observers quote inclinations that way. But for an axisymmetric disk the third angle does nothing, and the first two are easy to get wrong by a sign or a convention. A spin vector states the physically important fact directly: which way the disk rotates relative to the encounter. Whoever writes the spec (a person, a script, or a language model) can reason about "aligned with the orbital angular momentum" without a convention table.

</div>

### 3. One random stream per component

Each component draws its random numbers from its own stream of the same seeded generator:

```rust
for (i, c) in self.components.iter().enumerate() {
    let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
    rng.set_stream(i as u64);
    // ...
}
```

ChaCha20 supports $2^{64}$ independent streams per seed, so this costs nothing. The payoff is that editing one component never reshuffles another. Make the dwarf galaxy denser and the big galaxy's 8,000 particles stay exactly where they were. When you're iterating on a scenario, and especially when a program is iterating on it for you, changing one thing at a time is what lets you attribute a change in the result to a change in the input.

## Validation

A spec might be written by hand, by a script, or by a language model, and any of those can make mistakes. The parser is strict on purpose, and every error names the field that caused it.

**Unknown fields are errors.** Every struct uses `#[serde(deny_unknown_fields)]`, so a typo doesn't silently fall back to a default:

```text
invalid scenario spec: unknown field `spin_axes`, expected one of `particles`,
`position`, `velocity`, `spin_axis`, `baryon_mass`, ...
```

**Numbers are checked before generation.** `validate()` rejects non-finite positions, negative masses, a zero spin axis, or a `disk_fraction` outside $[0, 1]$, and it prefixes the component index and type:

```text
components[1] (gas_cloud): radius must be a positive number, got -1
```

Some checks guard against failures deep inside a sampler. The Hernquist halo is sampled by inverting its cumulative mass (Chapter 6), drawing $U$ from a range that ends at the truncation radius. If `halo_max_radius` is tiny compared with `halo_scale_radius`, that range is empty and the sampler would panic; validation turns that into a sentence instead.

**A total-particle cap** of 500,000 catches `"particles": 5e7` before it allocates gigabytes. Renderers may impose lower caps: the browser rescales anything over 50,000.

## Derived Defaults

A spec *may* set `dt`, `softening`, and `view_radius`. Usually it shouldn't: these depend on the physics of the components, and getting them right by hand is tedious. Here is how each is derived ([`spec.rs`](https://github.com/jcorbettfrank/gravis/blob/m8/crates/sim-core/src/spec.rs), [ADR 008](https://github.com/jcorbettfrank/gravis/blob/m8/docs/adr/008-scenario-specs.md)).

### Softening: the smallest anyone asks for

Each extended component proposes a softening length using the same rule as its built-in scenario, and the spec takes the minimum:

| Component | Proposed softening $\epsilon$ |
|---|---|
| `disk_galaxy` | $0.05 \, R_d$ |
| `plummer` | half the mean interparticle spacing at the half-mass radius $r_h \approx 1.305 \, a$ |
| `gas_cloud` | $0.3 \times$ the mean interparticle spacing |
| point masses only | $0.1\%$ of the closest Kepler pericenter |

Taking the minimum means the densest component is resolved properly. The cost is that the other components are softened less than they need, which slightly raises two-body relaxation in them; for the short encounters specs are meant for, that's the right trade.

### Timestep: the fastest thing that happens

The timestep must resolve the fastest orbit anywhere in the system. The spec computes a candidate for every source of fast motion and takes the minimum:

1. **Internal dynamics of each extended component.** A disk galaxy uses its own rule from Chapter 6 (orbital period at the disk scale length, divided by 500; about $7.4 / 500 \approx 0.015$ for the default galaxy). A Plummer sphere uses its dynamical time $\sqrt{a^3 / GM}$ divided by 100. A gas cloud uses its free-fall time divided by 500.
2. **Point mass + point mass.** The two-body Kepler orbit at pericenter, divided by 1000.
3. **Point mass + extended component.** This one is subtle. A star cluster orbiting a black hole has its center pass at pericenter $q$, but its nearest stars pass closer, at roughly $q - r_{\text{ext}}$, where $r_{\text{ext}}$ is the cluster's visible size. Those stars set the pace, so the candidate is the circular-orbit period at $q - r_{\text{ext}}$, divided by 50.

The pericenter in rules 2 and 3 comes from treating each pair as point masses on a Kepler orbit, using the energy and angular momentum of their initial separation $\mathbf r$ and relative velocity $\mathbf v$:

$$\mathcal{E} = \tfrac{1}{2} v^2 - \frac{\mu}{r}, \qquad h = |\mathbf r \times \mathbf v|, \qquad e = \sqrt{1 + \frac{2 \mathcal{E} h^2}{\mu^2}}, \qquad q = \frac{h^2}{\mu (1 + e)}$$

with $\mu = G(m_1 + m_2)$. This formula is valid for bound and unbound orbits alike, and handles the radial plunge ($h = 0 \Rightarrow q = 0$) without special cases.

<div class="physics-note">

**An earlier version used the softening length instead of $q - r_{\text{ext}}$.**
Using $\epsilon$ as the closest approach is the safe, lazy choice: no star can get closer than that in any meaningful sense. But for a dense cluster on a wide orbit around a black hole, $\epsilon$ is thousands of times smaller than the real closest approach, and the timestep came out about 100 times shorter than needed. Bounding the approach with the component's actual geometry fixed that. The general lesson: a "safe" default that makes a run 100 times slower is a default people (and programs) will learn to override, which defeats the point of having one.

</div>

### View radius: frame what matters

Renderers need to know how far back to put the camera. The spec computes the farthest component's distance from the center of mass plus its visible extent ($4 R_d$ for a disk, which holds about 90% of its mass; $3a$ for a Plummer sphere, about 85%). The native app and the browser hand this radius to `OrbitalCamera::frame_radius`, which picks a distance that fits it in the field of view.

### The examples, worked through

Here is what the rules produce for the three example specs (output of `headless --spec`):

| Spec | dt | Rule that won | Softening | Rule that won | View radius |
|---|---|---|---|---|---|
| `tilted-flyby` | 0.0148 | disk galaxy, period / 500 | 0.050 | $0.05 \, R_d$ | 11.6 |
| `minor-merger` | 0.0148 | the big disk | 0.030 | the dwarf's $0.05 \times 0.6$ | 10.3 |
| `cluster-and-black-hole` | 0.0035 | cluster dynamical time / 100 | 0.0036 | cluster spacing | 3.1 |

The cluster case is the interesting one. The cluster's center passes the black hole at $q \approx 0.47$, and its outer stars at about $0.47 - 0.15 = 0.32$. The orbit period there is $2\pi \sqrt{0.32^3} \approx 1.1$, so rule 3 asks for $\Delta t \approx 0.022$. But the cluster itself is tiny ($a = 0.05$) and light ($M = 0.001$), so its internal dynamical time is $\sqrt{0.05^3 / 0.001} \approx 0.35$, and rule 1 asks for $0.0035$, six times shorter. The internal dynamics win.

These are defaults, not guarantees. The way to check them is the way we've checked everything in this book: run the spec and read the energy drift. Over 200 steps at $N = 1000$ the three examples drift by $1.1 \times 10^{-4}$, $9.3 \times 10^{-5}$, and $7 \times 10^{-7}$ respectively. If a spec you wrote drifts by $10^{-2}$, halve `dt` and see whether the drift drops by about 4× (the leapfrog's second-order signature from [Chapter 2](ch02_integrators.md)); if it does, the timestep was the problem.

## Encounter Geometry

The derived numbers keep a simulation *stable*. Whether it's *interesting* depends on the encounter geometry, and that is the part of a spec that needs the most thought. This section is the practical core of Toomre & Toomre (1972), the paper that first explained tidal tails as the product of an ordinary gravitational flyby.

### Reduce it to two bodies

Before running anything, treat the two most massive components as point masses. A default disk galaxy has total mass $M = M_b (1 + 10) = 11$ (baryons plus a halo ten times heavier), so the tilted flyby is a two-body problem with $\mu = 22$. From the spec's positions and velocities:

- **Separation** $r = |(15, 2, 0)| \approx 15.1$
- **Relative speed** $v = 0.5$
- **Escape speed** $v_{\text{esc}} = \sqrt{2\mu / r} \approx 1.71$

Since $v < v_{\text{esc}}$, the galaxies are **bound**: after their first passage they'll turn around and come back, and eventually merge. An approach faster than $1.7$ gives a one-time flyby that never returns.

The Kepler eccentricity comes out at $e \approx 0.997$ and the point-mass pericenter at $q \approx 0.023$. Taken literally, that's a near head-on collision. It isn't, quite: these are extended galaxies whose halos reach 30 units, and once the halos overlap, the force between the centers is no longer $\mu / r^2$. The halos also exchange energy with each other (dynamical friction, Chapter 6), so the real orbit decays faster than Kepler predicts. The point-mass numbers are still the right first check: they tell you *bound or not*, *close or distant*, and, most importantly, *the orientation of the orbital plane*.

### Orbital angular momentum

The relative orbit has angular momentum

$$\mathbf L \propto \mathbf r \times \mathbf v = (15, 2, 0) \times (-0.5, 0, 0) = (0, 0, 1)$$

The encounter happens in the $xy$-plane, and the two galaxies swing around each other counter-clockwise as seen from $+z$. That matches the convention in [`specs/README.md`](https://github.com/jcorbettfrank/gravis/blob/m8/specs/README.md): put encounters in the $xy$-plane so the orbital angular momentum points along $\pm z$, and spin axes become easy to reason about.

### Prograde, retrograde, and in between

Now compare each disk's `spin_axis` with $\hat{\mathbf L}$:

- **Galaxy 1**, `spin_axis` $[0, 0, 1]$: parallel to $\hat{\mathbf L}$. Its stars orbit their center in the same sense as the companion orbits it. This is a **prograde** encounter.
- **Galaxy 2**, `spin_axis` $[0, 0.866, 0.5]$: $\cos\theta = 0.5$, so it's tilted $60°$ from $\hat{\mathbf L}$.

Why does this matter so much? Consider a star on the near side of a prograde disk while the companion swings past. In the star's own frame, the companion is moving slowly: both the star and the companion are sweeping around in the same direction. The tidal pull acts in a steady direction for a long time, and the star's orbit absorbs it coherently. It's a resonance: the star's orbital angular velocity roughly matches the companion's angular velocity during the passage. The near-side stars get pulled into a bridge toward the companion; the far-side stars, now orbiting too fast for the weakened binding, are flung outward into a long tail.

In a **retrograde** disk the same star sees the companion whip past in the opposite direction. The tidal force rotates rapidly in the star's frame and largely averages out. The disk thickens and warps, but it doesn't make tails.

<div class="verification">

**Measured: does the spin actually matter?**
We ran two orbits at 6,000 particles per galaxy (Barnes-Hut, $\theta = 0.5$), changing only galaxy 1's `spin_axis`, and counted the fraction of galaxy 1's disk stars farther than 8 units (two visible disk radii) from the center of its own bulge. Galaxy 2 keeps its 60° tilt throughout.

| Orbit | Galaxy 1 spin | $t = 20$ | $t = 30$ | $t = 50$ |
|---|---|---|---|---|
| **Wide**: offsets $\pm 4$ in $y$, velocities $\pm 0.7$ ($q \approx 3.0$, $e \approx 0.91$) | prograde `[0, 0, 1]` | 15.7% | 23.8% | 17.5% |
| | polar `[1, 0, 0]` | 0.0% | 1.0% | 1.0% |
| | retrograde `[0, 0, -1]` | 0.0% | 0.0% | 0.3% |
| **Close**: `tilted-flyby.json` ($q \approx 0.02$, $e \approx 0.997$) | prograde | 1.6% | 1.8% | 2.1% |
| | polar | 0.0% | 1.6% | 1.3% |
| | retrograde | 5.2% | 1.8% | 3.1% |

On the wide orbit the result is Toomre & Toomre's, unmistakably: the prograde disk loses about a quarter of its stars to tails, while the polar and retrograde disks lose almost none. On the close orbit, the differences vanish into the noise (about 380 disk stars per galaxy, so ±1%). That orbit is nearly a radial plunge: the galaxies fall straight through each other with very little orbital angular momentum, so there's no slow, sustained sweep for the disk's rotation to resonate with. The violent, nearly head-on passage disturbs every disk about equally, whatever its spin.

The lesson for spec writers: **prograde vs. retrograde only matters if there is an orbit to be prograde with.** A small impact parameter on a slow, bound approach gives a nearly radial orbit, and then the spin barely matters. To get Toomre-style tails, aim for a pericenter of a few disk scale lengths.

</div>

### Designing an encounter

Putting it together, a checklist for writing an encounter spec:

1. **Put the orbit in the $xy$-plane.** Positions and velocities with $z = 0$; then $\hat{\mathbf L} = \pm \hat{\mathbf z}$.
2. **Choose the spins on purpose.** `[0, 0, 1]` for prograde (when $\hat{\mathbf L} = +\hat{\mathbf z}$), `[0, 0, -1]` for retrograde, anything in between for a tilt. The tidal response falls off as the tilt grows.
3. **Check bound vs. unbound** against $v_{\text{esc}} = \sqrt{2 G (M_1 + M_2) / r}$. Remember that a default galaxy weighs 11, not 1.
4. **Check the pericenter, not just the offset.** The perpendicular offset sets the angular momentum, but a slow, strongly bound approach can still end up nearly radial. Aim for a Kepler pericenter $q$ of a few disk scale lengths if you want tails; $q \ll R_d$ is a plunge, where spin hardly matters.
5. **Scale smaller galaxies consistently.** A dwarf at a fraction $f$ of the default mass should shrink every radius by about $f^{1/3}$ to keep a similar density, as `minor-merger.json` does with $f = 0.2$ and radii scaled by $0.6$. A dwarf with the default radii but a fifth of the mass is fluffy and is shredded almost immediately.
6. **Estimate the timescale.** In the tilted flyby the disks overlap by $t \approx 25$ and have largely merged by $t \approx 50$. Slower approaches merge sooner.

The live demo below does steps 1–3 for you: it shows the two-body numbers and each disk's angle to $\hat{\mathbf L}$ every time you run a spec.

## Rescaling: Preview Small, Render Large

Every runner accepts a particle budget (`-n` on the command line, the particle slider in the browser) and rescales every extended component proportionally:

```rust
pub fn with_particle_budget(&self, total: usize) -> Self {
    // ...
    let scale = total as f64 / current as f64;
    let rescale = |n: &mut usize| *n = ((*n as f64 * scale).round() as usize).max(1);
    // disk galaxies, Plummer spheres, and gas clouds are rescaled;
    // point masses stay single particles.
}
```

Because component masses don't depend on particle counts, rescaling changes the resolution, not the physics. The workflow is to iterate at a few thousand particles, where a run takes seconds, then render the one you like at 40,000:

```bash
# Quick check: does the encounter happen, does energy hold?
cargo run --release -p headless -- --spec specs/tilted-flyby.json -n 2000 \
    --algorithm barnes-hut --steps 3400 --diag-interval 500

# Final frame at t = 25 (macOS)
cargo run --release -p native-app -- --spec specs/tilted-flyby.json -n 40000 \
    --algorithm barnes-hut --screenshot artifacts/media/flyby.png --screenshot-time 25
```

Softening is derived from the rescaled spec, so a Plummer cluster's $\epsilon$ shrinks as its particle count grows, as it should.

## Honest Limits

Specs compose existing models; they inherit those models' limits.

<div class="caveat">

**Equal-mass particles.** Every particle in a galaxy has the same mass, so dark matter gets about 91% of the particles. At 6,000 particles per galaxy, each disk has only about 380 stars. Lighter star particles would help, at the cost of heavy halo particles heating the disk; that trade-off is being measured separately before it becomes a spec field.

**Approximate equilibria.** The disk and halo velocities come from the approximations in Chapter 6, not a self-consistent distribution function. An isolated galaxy settles slightly during its first few dynamical times. For encounters, that's small next to the tidal response.

**Gas.** A `gas_cloud` turns on the SPH solver, and self-gravitating gas currently blows up at peak compression (the Evrard bounce, colliding clouds). Until that's fixed, no example spec uses gas, and gas specs are only trustworthy before strong compression. Even then, these runs show cloud collapse and gas compression, not star formation: there is no cooling, no sink particles, and no star-formation model.

</div>

## Try It Yourself

The playground loads the example specs and lets you edit them. **Wide prograde** and **Wide retrograde** are the wide orbit from the measurement above, identical except for galaxy 1's spin; run each to $t \approx 30$ and compare galaxy 1's tails. **Tilted flyby** is the close, nearly radial encounter from `specs/`. **Cluster + BH** shows a star cluster stripped into leading and trailing streams over a few orbits. Edit the JSON and press **Run spec**: errors appear below the editor, and the geometry readout shows the two-body numbers and each disk's angle to the orbital angular momentum. The **N** slider rescales the running spec, and **speed** sets sim time per second (the browser runs single-threaded, so large $N$ may not keep up).

<div class="live-demo live-demo--full">
  <iframe src="demos/spec-playground.html" width="100%" loading="lazy"
          title="Interactive scenario spec playground"></iframe>
  <p class="demo-fallback" style="display:none">
    <em>Live demo requires a WebGPU-enabled browser (Chrome 113+, Edge 113+, Safari 18+).</em>
  </p>
</div>

Some things to try:

- Raise the relative speed of the flyby above $v_{\text{esc}} \approx 1.7$ (e.g. velocities $\pm 1.0$) and watch the galaxies separate for good.
- Set the $y$-offsets to zero for a head-on collision. The geometry readout reports a radial plunge, and both disks respond symmetrically regardless of spin.
- Give the cluster a lower tangential velocity (`[0, 0.15, 0]`) so it plunges closer to the black hole, and watch how much faster it's disrupted. Note how `dt` shrinks in the status line.

## Further Reading

- [Toomre & Toomre (1972)](https://ui.adsabs.harvard.edu/abs/1972ApJ...178..623T/) -- "Galactic Bridges and Tails": prograde resonance and the origin of tidal tails
- [Barnes & Hernquist (1992)](https://ui.adsabs.harvard.edu/abs/1992ARA%26A..30..705B/) -- "Dynamics of Interacting Galaxies": self-consistent N-body mergers, including the role of halos
- [Binney & Tremaine, *Galactic Dynamics*](https://press.princeton.edu/books/paperback/9780691130279/galactic-dynamics) (2008), §8.3 -- tidal effects and encounters
- [Rodrigues' rotation formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula) -- the general form behind `rotation_from_z`
- [Kepler orbit](https://en.wikipedia.org/wiki/Kepler_orbit) -- energy, angular momentum, eccentricity, and pericenter
- [ADR 008: Declarative Scenario Specs](https://github.com/jcorbettfrank/gravis/blob/m8/docs/adr/008-scenario-specs.md) -- why composition, per-component streams, and derived defaults

## What's Next

A spec is easy for a person to write, but the point of the format is that it's also easy for a program to write, check, and revise. The `/universe` Claude Code skill ([ADR 013](https://github.com/jcorbettfrank/gravis/blob/main/docs/adr/013-universe-skill.md)) does exactly that: it takes a plain-English description ("a dwarf galaxy plunges through a Milky Way disk"), writes a spec, runs it headless, reads the energy drift and rendered frames, and revises until both the physics and the picture hold up.

The **[Text to Universe page](demos/universe.html)** runs the same loop in your browser. Type a description and Claude designs the orbit, test-runs each draft here (energy conservation, whether the encounter happens, a look at the frames), revises, and then the scene plays live. It needs your own Claude API key, which goes straight from the page to Anthropic; the pre-built examples work without one ([ADR 014](https://github.com/jcorbettfrank/gravis/blob/main/docs/adr/014-public-universe-page.md)).
