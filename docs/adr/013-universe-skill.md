# ADR 013: The /universe Skill

**Date**: 2026-10-11
**Status**: Accepted

## Decision

M8 step 2 is a Claude Code skill, `.claude/skills/universe/`, not a program. `SKILL.md` gives the method (describe → spec → headless run → check → revise) with acceptance criteria and a symptom → fix table. Two standard-library Python helpers do the parts that shouldn't be done by eye or by mental arithmetic: `orbit.py` designs and inspects two-body orbits, and `check.py` condenses a headless `summary.json` into PASS/WARN/FAIL checks. The headless summary gains two additive fields per component pair, `pericenters` and `history`, so the loop can see *when* things happened.

## Context

The loop has three kinds of work. Translating a description into components and judging whether a frame "looks like" the description is language and vision work, which is what Claude is for. Orbital geometry (what velocity gives a pericenter of 0.3 from 12 units away?) is arithmetic that Claude gets subtly wrong when it does it in its head. And deciding whether energy held or an encounter happened needs fixed thresholds, so two runs are judged the same way.

### 1. A skill, not an executable pipeline

A Rust or Python driver that calls a model API would duplicate what Claude Code already does (read files, run commands, look at images, revise), and would need its own credentials and prompt plumbing. A skill is a document plus helpers, runs wherever Claude Code does (including cloud containers with no GPU, via [ADR 010](010-headless-renderer.md)'s renderer), and is easy to improve as failure modes turn up. Step 3 (a public page) will need a server-side driver anyway; it can follow the same method.

### 2. Orbit helper in Python, not a spec field

An `orbit: {pericenter, eccentricity}` field in the spec would put the geometry in `sim-core`, but it would also freeze one parameterization into the format and make specs depend on a solver for their initial conditions. Instead `orbit.py design` prints plain positions and velocities to paste, and `orbit.py inspect` reports the Kepler orbit of every pair in any spec, including hand-edited ones. The spec format ([ADR 008](008-scenario-specs.md)) stays positions and velocities.

Kepler treats each component as a point of its total mass. That is exact for point masses and compact clusters, but extended galaxy halos overlap long before the disks meet, so the first passage comes later and wider than Kepler says. Measured on the example specs at N = 4,000 (Barnes-Hut θ = 0.5):

| Spec | Kepler pericenter, time | Measured first passage |
|---|---|---|
| `cluster-and-black-hole.json` | 0.47 at t = 7.2 | 0.47 at t = 7.2 |
| `tilted-flyby.json` | 0.02 at t = 10.1 | 0.23 at t = 12.4 |
| `minor-merger.json` | 0.62 at t = 7.8 | 1.5 at t = 11.0 |

The skill tells Claude to plan with ~1.2–1.4× Kepler's time for galaxies and read the real times off the separation history. A better effective-mass model (enclosed halo mass) was not worth its complexity while the history makes the correction visible after one preview.

### 3. Thresholds

`check.py` passes energy at max |dE/E| ≤ 1e-3, warns up to 1e-2, and fails beyond, or on a non-finite energy. An encounter passes if the two tracked centers came within the sum of their initial visible half-mass radii, or within an explicit distance for scenes that shouldn't touch. These are loose next to the verification matrix in `PLAN.md` (1e-4 for test problems) because the loop's question is whether a picture can be trusted, not whether the integrator is correct.

While tuning the worked example (a dwarf plunging through a disk, N = 4,000), drift at θ = 0.5 was 2.0e-3 by t = 24. Halving `dt` only reduced it to 1.3e-3; θ = 0.3 gave 3e-4, and brute force 7e-5. So in galaxy encounters the drift is Barnes-Hut force error, not the timestep, and the skill's first remedy is `--theta 0.3` (about 2× the cost).

### 4. Close passages and separation history in the summary

ADR 010's summary records only each pair's minimum separation. That answers "did they meet" but not "when first", which sets render times, and on a merger the minimum is the final coalescence, not the first passage. Two additive fields per pair:

- `pericenters`: each completed separation minimum, found with hysteresis (a minimum counts once the separation climbs 10% of the initial separation above it), so a merged remnant's jitter isn't counted.
- `history`: `[time, separation]` samples, thinned evenly to at most 128 by halving when full, plus the last step. That is enough to see the infall and every passage without extra frames, and stays small for any run length.

Both are O(pairs) per step. Existing fields are unchanged, per ADR 010.

## Consequences

- `/universe <description>` works in a cloud thread: the worked example in `.claude/skills/universe/example.md` went from description to an accepted spec in three previews, about 2 minutes of simulation in total.
- Run directories go to `artifacts/universe/` (gitignored). Specs worth keeping are copied to `specs/`, where `cargo test -p sim-core --test spec` covers them.
- The skill's numbers (costs, calibration, thresholds) were measured on a 4-core cloud container and should be refreshed when the solver changes.
- Gas scenes remain unreliable past peak compression until the SPH energy fix lands; the skill says so and steers away from `gas_cloud`.
