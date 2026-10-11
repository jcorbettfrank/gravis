# Worked example: "A dwarf galaxy plunges through a Milky Way disk"

A real run of the loop (2026-10-11, 4-core cloud container), kept short. The
numbers are what the tools printed.

## Translate

- Milky Way disk: default `disk_galaxy` (total mass 11).
- Dwarf: `disk_galaxy` with `baryon_mass` 0.2 (total 2.2, a 1:5 mass ratio),
  every radius × 0.2^⅓ ≈ 0.585.
- "Plunges through": the dwarf must cross the disk, not orbit in its plane.
  The orbit is in the xy-plane and the dwarf falls in roughly along x, so the
  Milky Way's `spin_axis` is `[1, 0, 0]`: its disk lies in the yz-plane and
  the dwarf hits it face-on. Render `xy` (disk edge-on, shows the path) and
  `yz` (disk face-on).

## v1: parabolic, pericenter 0.3

```bash
python3 $U/orbit.py design --m1 11 --m2 2.2 --pericenter 0.3 --separation 12
```

gave body 1 at `[-2, 0, 0]` moving `[0.244097, -0.039087, 0]` and body 2 at
`[10, 0, 0]` moving `[-1.220485, 0.195434, 0]`, Kepler pericenter at t = 5.6.
Rendered at 0, 8, 12, 16, 24 with N = 4,000 (20 s):

```
[0]-[1] start 12.4, closest 0.524 at t=6.62, end 17.4
  first within reach: t=6.15; completed passages: t=6.62 d=0.524
WARN  energy: max |dE/E| = 0.00197
PASS  encounter [0]-[1]
```

The frames show the dwarf crossing the edge-on disk and leaving with a
trailing stream; the Milky Way disk survives (half-mass radius ×1.02) and the
dwarf is half disrupted (×1.98). The first passage came at 6.6, 1.18× Kepler's
time. The story works; the energy doesn't quite.

## v2: halve dt

Set `"dt": 0.0074` (derived was 0.0148). Max |dE/E| went only to 1.3e-3 for
twice the cost. Probing v1 at θ = 0.3 (3.6e-4) and brute force (7e-5) showed
the drift is Barnes-Hut force error, not the timestep: back out the `dt`
change.

## v3: θ = 0.3, pericenter 0.1

Two changes: `--theta 0.3`, and a pericenter of 0.1 so the dwarf goes through
the middle of the disk rather than its inner edge. N = 4,000, 42 s:

```
[0]-[1] start 12.4, closest 0.196 at t=6.5, end 15.4
  completed passages: t=6.5 d=0.196
PASS  energy: max |dE/E| = 0.000299
PASS  encounter [0]-[1]: closest 0.196 at t=6.5 <= reach 1.91
```

At t = 6.5 the `xy` frame shows the dwarf exactly in the disk plane; by t = 10
it is through, with a faint bridge of stripped stars back toward the disk, and the Milky Way disk
has thickened and warped (×1.05). Accepted.

## Final run and report

Same spec at N = 20,000, θ = 0.3, rendered at 0, 6.5, 12, 18 (8 min):

```
[0] disk_galaxy  N=16000  r_half 1.14 -> 1.01  x0.88
[1] disk_galaxy  N=4000   r_half 0.667 -> 0.891  x1.33
[0]-[1] closest 0.485 at t=6.25; completed passages: t=6.25 d=0.485
PASS  energy: max |dE/E| = 0.000154
PASS  encounter [0]-[1]
```

The details moved, which is why the final run gets checked again: a larger N
resamples every component, and this passage came 0.49 from the center rather
than 0.2, so the dwarf lost less (×1.33). The story held: the `xy` frames
show the dwarf inside the edge-on disk at t = 6.5 and a long stream behind it
at t = 18, and at this resolution the face-on (`yz`) frames show spiral and
ring-like waves spreading through the disk after the passage, which N = 4,000
was too sparse to show.

What the run honestly shows: a satellite galaxy on a parabolic orbit passing
through the center of a disk galaxy five times its mass, losing part of its
stars into a stream, and sending spiral and ring-like density waves through
the disk. What it doesn't show: no gas, so no shocks or star formation; and
the waves are not a clean collisional ring like the Cartwheel's (Lynds &
Toomre 1976), so the report shouldn't call it a ring galaxy.
