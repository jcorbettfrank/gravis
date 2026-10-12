# ADR 014: The Public Text-to-Universe Page

**Date**: 2026-10-12
**Status**: Accepted

## Decision

M8 step 3 is a static page, `crates/web-app/static/demos/universe.html` (published with the book as `demos/universe.html`). A visitor types a description; Claude, called from the visitor's browser with the visitor's own API key, runs the `/universe` method ([ADR 013](013-universe-skill.md)) against two tools the page implements: `design_orbit` (a JavaScript port of `orbit.py design`) and `run_preview` (a short simulation in a Web Worker that returns `check.py`'s numbers and verdicts plus rendered frames). Each previewed spec plays live in the WebGPU view while Claude revises. Four pre-built examples from `specs/` work with no key.

Supporting changes:

- `sim_core::tracking`: component tracking (centers, half-mass radii, pairwise encounters with passages and separation history) moved out of `headless`'s summary so both runners share one implementation. `headless` output is unchanged.
- `sim_core::preview::Preview`: a gravity-only check run that is built from a spec, advanced in caller-sized chunks, and reports energy drift (with `check.py`'s 1e-3 / 1e-2 thresholds) and encounters. Gas specs are rejected while the SPH energy fix is pending.
- `web-app`: `SpecPreview` exposes `Preview` to JavaScript, and the WASM entry point returns early when there is no `window`, so the same module loads in a worker.

## Context

The handoff plan's step 3 is "a public page where visitors type their own descriptions". The site is GitHub Pages: static files, no server. Turning text into a spec needs a model call, and judging the spec needs a run.

### 1. The visitor's key, called from the browser

Alternatives:

| Option | Cost to the project | Abuse exposure | Works today |
|---|---|---|---|
| Visitor's key, browser → api.anthropic.com | none | none (each visitor pays for their own) | yes |
| Hosted proxy holding a project key | every visitor's tokens | needs rate limits, quotas, bot defenses | needs a deployment and a secret |
| Pre-generated gallery only | none | none | yes, but no typing |

The visitor's key is the only option that is public, free to run, and needs nothing deployed. The API allows direct browser calls when the client opts in (`dangerouslyAllowBrowser`, which sends `anthropic-dangerous-direct-browser-access`); the risk that flag names is shipping *your* key to visitors, which doesn't apply when visitors bring their own. The key stays in the page's memory unless the visitor ticks "remember", which puts it in `localStorage`.

The page also accepts an API proxy URL (a settings field, `?proxy=URL`, or `DEFAULT_PROXY` in the page). The SDK just uses it as its base URL, so a later deployment that holds a key behind a rate-limited proxy (a Cloudflare Worker, say) needs no page changes. Building that proxy is left until there is a reason to pay for visitors.

### 2. No code from a CDN on a page that holds a key

The Anthropic TypeScript SDK (pinned 0.130.0) is vendored next to the page at build time by `scripts/vendor_anthropic_sdk.sh`, which checks the npm tarball against its registry sha512 and copies only the ES modules. A CDN import would let a compromised CDN read keys. The SDK's single bare import (`standardwebhooks`, for server-side webhook verification) is mapped to a stub with an import map. Nothing else is fetched from other origins.

### 3. The same loop as the skill, with tools instead of a shell

The skill's value is the check, not the spec writing: Kepler arithmetic done by a script rather than in Claude's head, fixed thresholds, and Claude looking at frames. The page keeps all three:

- `design_orbit` is `orbit.py design` ported line for line (`universe-orbit.js`); on three designs and an `inspect` of `minor-merger.json` the outputs are identical JSON. `run_preview` results also start with `orbit.py inspect`'s Kepler table.
- `run_preview` prints the same blocks and PASS/WARN/FAIL lines as `check.py` (`formatReport` in `universe-agent.js`), from the same tracking code `headless` uses.
- Frames are a CPU density splat in the worker (`OffscreenCanvas`), a reduced copy of `headless`'s renderer: bilinear splat, type colors, dark matter hidden, asinh stretch with the first frame's exposure, point masses as white dots, no blur. Each frame is one 514×256 PNG with the face-on (xy) and edge-on (xz) views side by side, sent to Claude as an image block in the tool result.

The system prompt condenses `SKILL.md` and `specs/README.md` (component table, encounter rules, calibration, symptom table, honesty rules). It is long and stable, so the request uses automatic prompt caching. Limits: 4 previews, 12 model turns.

Model defaults to Claude Opus 5.5 with adaptive thinking at `effort: "medium"`; Sonnet 5.5 and Haiku 5.5 are offered for visitors who want it cheaper. Opus and Sonnet requests opt into server-side refusal fallbacks (`fallbacks: "default"`); a refusal that survives is shown as such.

### 4. Preview size

Previews rescale every spec to 1,500 particles at Barnes-Hut θ = 0.5, capped at 6,000 steps and t ≤ 100. Measured natively on one thread (wasm is slower), the dwarf plunge to t = 30 (2,031 steps) took 28 s at N = 2,000 and 9 s at N = 1,000; θ = 0.3 cost 1.7× and didn't help at this size (2.5e-3 vs 2.3e-3; brute force 2.3e-3 at N = 1,000), so the drift there is set by particle number and softening, not tree error. The prompt therefore tells Claude that WARN (≤ 1e-2) is acceptable for a preview if it says so, while FAIL still has to be fixed. The live view runs the spec at its own particle count (up to 50,000).

The preview's step count and the live view's particle count trade against the visitor's patience; both are constants in `universe-agent.js` and worth revisiting with in-browser timings on real hardware.

## Consequences

- The page works for anyone with a WebGPU browser and an Anthropic API key, at no cost to the project. Without a key, it is a gallery of four scenes.
- Each scene costs the visitor a few model calls with up to four small images.
- The orbit helper and the check now exist in Python (the skill) and JavaScript (the page). `universe-orbit.js` says to keep them in step; a change to `check.py`'s thresholds should change `sim_core::preview` and `universe-agent.js` too.
- `book.yml` and `scripts/build_book.sh` copy the page's scripts, the vendored SDK, and `specs/*.json` into `demos/`; `specs/**` now triggers a book deploy.
- Verified in headless Chromium (SwiftShader WebGPU) against a mock Messages API: the loop designs an orbit, previews the dwarf plunge, shows verdicts and frames, and finishes with Claude's answer and the spec running live. Not yet verified against the real API from a deployed page, or for timing on real GPUs.
