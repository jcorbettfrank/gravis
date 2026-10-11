---
name: universe
description: Turn a plain-English description of a gravitational scene (galaxy encounters, star clusters, black holes) into a gravis scenario spec, run it headless, check energy conservation, whether the encounter happens and the rendered frames, and revise until it holds up. Use for "/universe <description>" or any request to simulate a described scenario.
---

# /universe: description → spec → run → check → revise

The user describes a scene ("a dwarf galaxy plunges through a Milky Way disk").
You write a JSON scenario spec, run it with `headless`, read the numbers and
the frames, and revise until **both** the physics and the picture match the
description. Then report what you built, what it shows, and what it doesn't.

Read `specs/README.md` once per session: it is the spec format reference
(fields, defaults, units). This file is the method; it doesn't repeat the
format.

Works anywhere `cargo` does, including cloud containers with no GPU or window:
everything here is headless. Python helpers use the standard library only.

## 0. Setup

```bash
cargo build --release -p headless        # once; then call the binary directly
H=./target/release/headless
U=.claude/skills/universe/scripts
```

Work in `artifacts/universe/<slug>/` (gitignored), one directory per scenario,
`<slug>` a few words of the description in kebab case. Each revision gets its
own subdirectory (`v1/`, `v2/`, ...) holding `spec.json`, the frames and
`summary.json`, so earlier attempts stay comparable.

## 1. Translate the description

Map each thing in the description to a component. Put the user's sentence in
the spec's `description` field verbatim, and give it a short `name`.

| Words | Component | Starting point |
|---|---|---|
| Milky Way, spiral galaxy, disk galaxy | `disk_galaxy` | defaults (total mass 11: baryons 1 + halo 10; 1 mass unit = 10¹⁰ M☉) |
| dwarf, satellite, small galaxy | `disk_galaxy` | `baryon_mass` 0.05–0.3, every radius × (mass fraction)^⅓, as in `specs/minor-merger.json` |
| giant / massive galaxy | `disk_galaxy` | `baryon_mass` 2–5, radii × (mass)^⅓; keep `halo_mass_ratio` × f/(1−f) ≈ 10 (see README) |
| elliptical, spheroid | `plummer` | not a true elliptical profile; say so |
| globular / star cluster | `plummer` | `mass` ~1e-4–1e-3 relative to a host, small `scale_radius` |
| black hole, massive star, perturber | `point_mass` | |
| gas cloud, nebula | `gas_cloud` | see the gas warning below |

Not modeled, so never claimed: star formation, supernovae or feedback,
cooling, magnetic fields, relativity, cosmic expansion, dark energy,
collisions between individual stars. If the description needs one, build the
closest gravitational scene and say plainly what is missing ("the merger
compresses the disks; this sim has no star formation, so there is no
starburst"). Follow the honesty rules in `CLAUDE.md` § Astrophysics Scope.

**Gas.** Self-gravitating SPH gas currently blows up at peak compression
(known issue, being fixed separately; don't edit `crates/sim-core/src/sph/`).
Avoid `gas_cloud` unless the description is about gas; if it is, keep the run
short of strong compression and say that only the early phase is
trustworthy.

## 2. Design the geometry

Encounters are where descriptions succeed or fail: "plunges through" needs a
close pericenter, "flyby" needs one that doesn't merge, "merger" needs a bound
orbit. Use the orbit helper rather than guessing velocities:

```bash
# Body 2 relative to body 1, both at their spec positions, in the xy-plane,
# orbiting counterclockwise (orbital angular momentum +z):
python3 $U/orbit.py design --m1 11 --m2 2.2 --pericenter 1 --eccentricity 1 --separation 12
# Check any spec, including one you edited by hand:
python3 $U/orbit.py inspect artifacts/universe/<slug>/v1/spec.json
```

`design` prints `body1`/`body2` positions and velocities ready to paste, the
escape speed, and the Kepler time to pericenter. Masses are component totals
(`inspect` computes them from a spec).

Rules of thumb (more in `specs/README.md` § Setting up encounters):

- **Orbital plane xy.** Positions and velocities with z = 0. The orbital
  angular momentum then points along +z (or −z with `--retrograde-orbit`).
- **Prograde tails.** A disk whose `spin_axis` is parallel to the orbital
  angular momentum throws out long tidal tails; antiparallel (retrograde)
  barely responds; tilted is in between. Use this deliberately: tails when the
  description wants drama, retrograde when it wants a disk to survive.
- **Plunge vs. graze.** Pericenter below the larger disk's scale radius (1 by
  default) goes through the disk; 2–4 disk scale radii raises tides without
  collision; beyond ~3× the halo scale radius (15) is a distant flyby.
- **Bound or not.** Eccentricity < 1 is bound and merges within a few
  passages; ≈ 1 (parabolic) is the classic Toomre setup; > 1 flies apart.
  Extended halos plus dynamical friction mean even parabolic galaxy pairs
  usually merge eventually.
- **Kepler is only a first guess for galaxies.** Halos (scale 5, reaching 30)
  overlap long before the disks touch, so the attraction is weaker early on
  and the first passage comes **later** than `design` predicts; see
  [calibration](#calibration). For point masses and compact clusters it is
  accurate.
- **Start separation.** Far enough that the disks begin undisturbed (≳ 10 for
  default galaxies), close enough that the run is short. Every unit of travel
  is wall time.
- **Spin axes.** `spin_axis` is the disk's angular momentum; `[0, 0, 1]` lies
  in the orbital plane, `[1, 0, 0]` is edge-on to it, `[0, 0.866, 0.5]` is
  tilted 60°.

## 3. Run a preview

Pick the run length from the design: render at the start, around the expected
first passage, and after it. Keep previews small so a revision costs about a
minute or two.

```bash
D=artifacts/universe/<slug>/v1
$H --spec $D/spec.json -n 4000 --algorithm barnes-hut \
   --render-at 0,20,35,50 --render-dir $D --render-view xy,xz \
   --diag-interval 250 > $D/run.log 2>&1
```

- `-n` rescales every component proportionally. 3,000–6,000 particles is
  plenty to see tails and disruption. Always pass `--algorithm barnes-hut`
  above ~2,000 particles.
- Accuracy: θ = 0.5 (the default) is fine to see whether the story works; a
  galaxy encounter at N = 4,000 typically drifts 5e-4 to 2e-3 by t ≈ 25–50. Use
  `--theta 0.3` for the final run, or as soon as the check WARNs.
- Cost: on a 4-core cloud container, N = 4,000 runs about 80 Barnes-Hut steps
  per second and N = 6,000 about 47. The default `dt` for galaxies is ≈ 0.015,
  so t = 50 is ~3,400 steps. A point-mass scene's `dt` is set by its tightest
  orbit, which can be much smaller; check the `dt:` line `headless` prints
  first, and `wall_seconds` in each summary, before going longer. θ = 0.3
  costs about 2×, brute force about 5× at N = 4,000.
- `--render-at` without `--steps` runs until the last requested time, and it
  turns on `summary.json`. The summary's maximum energy drift is sampled at
  each frame and every `--diag-interval` steps (an O(N²) potential each time,
  cheap at preview sizes), so keep the interval on; the log goes to
  `run.log` so it doesn't fill your context. `--diag-interval 0` without a
  summary skips the potential energy entirely, and the printed energy is then
  meaningless.
- Add `xz` to `--render-view` whenever anything is tilted or out of the xy-plane.
- Add `--render-autoexpose` for late, diffuse structure (a stream from a
  disrupted cluster) that the first frame's exposure leaves too faint.
- A spec error exits immediately and names the field; fix it and rerun.

## 4. Check

```bash
python3 $U/check.py $D/summary.json --expect-encounter 0,1
```

It prints the energy drift, each component's growth in half-mass radius, each
pair's separation history and passages, and the frame paths, then
PASS/WARN/FAIL lines (exit 1 on any FAIL). `--expect-encounter A,B` (repeat
for more pairs) demands that A and B came within the sum of their initial
half-mass radii: use it for every pair the description says should meet. For
scenes where the bodies shouldn't touch, give the distance: `0,1,0.6` for a
cluster meant to pass 0.5 from a black hole.

The separation history (`separation: t:d ...`, also `encounters[].history` in
the summary) is the quickest way to see when things happened; use it to move
render times onto the interesting moments.

Then **look at every frame** with the Read tool. Numbers can pass while the
picture is wrong (tails pointing the wrong way, everything out of frame, a
galaxy that never appears to move). Describe to yourself what each frame
shows before deciding.

What "holds up" means:

1. **Energy**: max |dE/E| ≤ 1e-3 (PASS). Up to 1e-2 (WARN) is acceptable for a
   preview if you say so, but fix it before the final run; worse means the
   integration is wrong and its pictures can't be trusted.
2. **Events happen**: every encounter the description implies happened
   (`--expect-encounter`), at a time inside the run, and nothing the
   description rules out happened (a "flyby" that merged, a "survives" galaxy
   with its radius tripled).
3. **Frames**: the described structure is visible: tails, a shredded dwarf, a
   stream, a ring. It sits inside the field of view and isn't saturated or too
   faint.

`r_half ... x3.50` in the check output means that component's visible
half-mass radius grew 3.5×: it has been disrupted. Under ~1.3× it has
survived largely intact (disks thicken a little on their own).

## 5. Revise

Change one or two things per revision, write `vN+1/spec.json`, and rerun.

| Symptom | Fix |
|---|---|
| Components never meet | Lower the pericenter or the relative speed (`orbit.py design`); start closer; run longer (the first passage may be after your last frame: look at the separation history) |
| Merged when it should be a flyby | Raise the eccentricity above 1 or the pericenter; raise the relative speed above the escape speed |
| No tidal tails | Make the disk prograde (spin parallel to the orbital angular momentum); lower the pericenter; give the partner more mass |
| Too violent, everything shredded | Raise the pericenter; make the disk retrograde or tilted; lower the partner's mass |
| Energy WARN/FAIL in a galaxy encounter | `--theta 0.3` (about 2× the cost). The drift there is Barnes-Hut force error, not the timestep: in the worked example halving `dt` only took 2.0e-3 to 1.3e-3, while θ = 0.3 gave 3e-4 and brute force 7e-5 |
| Energy WARN/FAIL with a point mass or tight cluster | Set `dt` explicitly (half the derived one), or raise `softening`; `dt` and `softening` are top-level spec fields. Try `--theta 0.3` too |
| Structure off the edge of the frame | Set the spec's `view_radius` (or pass `--render-radius`) |
| Everything a tiny blob | Lower `view_radius`; the default frames the starting positions, which can be far apart |
| Late structure invisible | `--render-autoexpose` |
| Disks look sparse | More particles in the final run; `star_particle_fraction` (default 0.5) already puts half of them in stars |

Stop when all three checks hold, or after about five revisions: then report
what still falls short and what you would change, rather than looping on.

## 6. Final run and report

Optionally rerun the accepted spec at a larger N (10,000–40,000) for nicer
frames; check it again, since a larger N changes softening and can change
the details. Then:

- If the user wants to keep it, copy the spec to `specs/<slug>.json` and add
  a row to the examples table in `specs/README.md`;
  `cargo test -p sim-core --test spec` checks every example parses, builds,
  and stays sane.
- Reply with: the spec (or its path), the frames that show the story, the
  check results in a sentence (energy drift, when the closest approach
  happened), what the run honestly shows and what it doesn't model, and how
  to watch it live:

```bash
cargo run -p native-app --release -- --spec specs/<slug>.json --algorithm barnes-hut     # macOS
```

  or paste the spec into the browser demo's **Spec…** panel.

A worked example of the whole loop is in [`example.md`](example.md).

## Calibration

Kepler (`orbit.py inspect`) against the measured separation history
(`check.py`, N = 4,000, Barnes-Hut θ = 0.5) for the example specs:

| Spec | Kepler: pericenter, time | Measured first passage | Later passages |
|---|---|---|---|
| `cluster-and-black-hole.json` | 0.47 at t = 7.2, period 14.3 | 0.47 at t = 7.2 | 0.46 at 20.9, 0.41 at 34.4 |
| `tilted-flyby.json` | 0.02 at t = 10.1 | 0.23 at t = 12.4 | 0.13 at 20.8, then merged |
| `minor-merger.json` | 0.62 at t = 7.8 | 1.5 at t = 11.0 | 0.43 at 21.6, 0.26 at 26.0, then merged |

So for a point mass and a compact cluster, Kepler is exact. For galaxies,
expect the first passage ~1.2–1.4× later than Kepler's time, and wider when
the smaller galaxy is much lighter. Each later passage is closer (dynamical
friction), and a bound galaxy pair merges within two or three. Plan render
times and run length with that margin, and read the real times off the
separation history rather than trusting the prediction.
