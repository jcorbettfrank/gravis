# Scenario specs

A spec describes initial conditions as JSON: a list of components (disk
galaxies, star clusters, point masses, gas clouds), each placed at a position
with a bulk velocity. Every runner accepts one:

```bash
cargo run --release -p headless -- --spec specs/tilted-flyby.json --algorithm barnes-hut --steps 2000
cargo run --release -p native-app -- --spec specs/tilted-flyby.json --algorithm barnes-hut
cargo run --release -p native-app -- --spec specs/tilted-flyby.json -n 40000 --algorithm barnes-hut \
    --screenshot artifacts/media/flyby.png --screenshot-time 25
```

In the browser demo, **Spec…** opens a panel to paste and run one; from
JavaScript, `load_spec(json)` does the same and returns a JSON summary.

`-n`/`--particles` (or the browser's particle slider) rescales every component
proportionally, so a spec can be previewed small and rendered large. The
browser caps specs at 50,000 particles.

## Units

N-body units, G = 1. For galactic scenes: 1 length unit = 1 kpc,
1 mass unit = 10¹⁰ M☉, 1 time unit ≈ 14.9 Myr, 1 velocity unit ≈ 65.6 km/s.
In the default disk galaxy (baryon mass 1, halo ratio 10) the circular
velocity at the disk scale radius is ~0.85 and the orbital period there ~7.4,
which sets the default timestep (period / 500).

## Top level

| Field | Default | Meaning |
|---|---|---|
| `components` | required | One or more components (below) |
| `name` | `"Custom Scenario"` | Display name |
| `description` | none | What the scenario should show, e.g. the prompt it came from |
| `seed` | `42` | RNG seed. Each component uses its own stream, so editing one never reshuffles another |
| `dt` | derived | Timestep. Default: the fastest internal or orbital timescale among components |
| `softening` | derived | Plummer softening. Default: the smallest any component asks for |
| `view_radius` | derived | Radius to frame around the center of mass. Default: farthest component + its size |

Unknown fields are rejected, and errors name the offending field, e.g.
`components[1] (gas_cloud): radius must be a positive number, got -1`.
The system is shifted to its center-of-mass frame after it is assembled.

## Components

Every component takes `position` and `velocity` (`[x, y, z]`, default origin
and at rest). Each component is centered on its own center of mass, so it sits
exactly at `position` and moves exactly at `velocity`.

### `disk_galaxy`

Exponential stellar disk, Plummer bulge, and Hernquist dark matter halo
([ADR 006](../docs/adr/006-galaxy-model.md)). Particles have equal mass, so
most of them (10 of every 11 at the default halo ratio) are dark matter.

| Field | Default | Meaning |
|---|---|---|
| `particles` | required | Total across disk, bulge, and halo |
| `spin_axis` | `[0, 0, 1]` | Direction of the disk's angular momentum. The disk lies perpendicular to it |
| `baryon_mass` | `1.0` | Disk + bulge mass |
| `disk_fraction` | `0.7` | Share of baryon mass in the disk (rest is bulge), in [0, 1] |
| `halo_mass_ratio` | `10.0` | Halo mass as a multiple of baryon mass, ≥ 0 |
| `disk_scale_radius` | `1.0` | Exponential disk scale length |
| `disk_scale_height` | `0.1` | sech² vertical scale |
| `bulge_scale_radius` | `0.3` | Plummer bulge scale |
| `halo_scale_radius` | `5.0` | Hernquist halo scale |
| `disk_max_radius` | `6.0` | Disk truncation |
| `halo_max_radius` | `30.0` | Halo truncation (≥ 10% of `halo_scale_radius`) |

A smaller galaxy should shrink with its mass: at a fraction *f* of the default
mass, scaling every radius by roughly *f*^⅓ keeps a similar density (see
`minor-merger.json`).

### `plummer`

A star cluster (or any spheroid) in virial equilibrium.

| Field | Default | Meaning |
|---|---|---|
| `particles` | required | |
| `mass` | `1.0` | Total mass |
| `scale_radius` | `1.0` | Plummer scale; ~85% of the mass lies within 3× this |

### `point_mass`

A single particle: a black hole, a star, a planet.

| Field | Default | Meaning |
|---|---|---|
| `mass` | required | |

### `gas_cloud`

A uniform-density sphere of SPH gas. Any gas in a spec turns on the SPH solver.
With one cloud and default fields this is the Evrard collapse setup.

| Field | Default | Meaning |
|---|---|---|
| `particles` | required | |
| `mass` | `1.0` | |
| `radius` | `1.0` | |
| `internal_energy` | `0.05` | Specific internal energy, a temperature proxy. 0.05 is cold enough to collapse |

**Known issue:** self-gravitating gas currently blows up at peak compression
(the Evrard bounce, colliding clouds). This predates specs and affects the
built-in gas scenarios too; until it is fixed, gas specs are only trustworthy
before strong compression.

## Setting up encounters

- **Orbital plane.** Put the encounter in the xy-plane (positions and
  velocities with z = 0). The orbital angular momentum then points along ±z,
  and a disk with `spin_axis` `[0, 0, 1]` lies in the orbital plane.
- **Prograde vs. retrograde.** A disk whose spin is aligned with the orbital
  angular momentum resonates with the passing galaxy and throws out long tidal
  tails; a retrograde disk barely responds (Toomre & Toomre 1972). Tilting the
  spin axis out of the orbital plane weakens the response.
- **Approach speed.** Two default galaxies (total mass 22) starting 15 apart
  are bound if their relative speed is below ~1.7, the two-body escape speed;
  slower approaches merge sooner.
- **Timescale.** In `tilted-flyby.json` (15 apart, relative speed 0.5) the
  disks overlap by t ≈ 25 and have largely merged by t ≈ 50.

## Examples

| Spec | What it shows |
|---|---|
| [`tilted-flyby.json`](tilted-flyby.json) | Prograde encounter; one disk tilted 60° |
| [`minor-merger.json`](minor-merger.json) | A dwarf galaxy shredded by a Milky Way–like disk |
| [`cluster-and-black-hole.json`](cluster-and-black-hole.json) | A star cluster stripped into streams by a black hole |

`cargo test -p sim-core --test spec` checks that every example parses, builds,
and stays physically sane.
