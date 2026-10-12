// Preview runs for the text-to-universe page (universe.html), off the main
// thread so the live view keeps rendering. Each run builds a sim_core
// `Preview` (via the WASM `SpecPreview`), steps it in chunks, draws frames at
// the requested times, and posts back a report: energy drift and encounters,
// the same numbers the /universe skill reads from headless. See ADR 014.
//
// in:  {type: 'run', id, spec, particles, until, maxSteps, frameTimes}
//      {type: 'cancel', id}
// out: {type: 'ready'} | {type: 'progress', id, progress, time}
//      {type: 'frame', id, time, blob} | {type: 'done', id, report}
//      {type: 'error', id, message}

import init, { SpecPreview } from './pkg/web_app.js';

const ready = init().then(() => postMessage({ type: 'ready' }));
let current = null; // id of the run in progress
const CHUNK_STEPS = 20;

onmessage = async (ev) => {
    const msg = ev.data;
    if (msg.type === 'cancel') {
        if (current === msg.id) current = null;
        return;
    }
    if (msg.type !== 'run') return;
    await ready;
    current = msg.id;
    try {
        await run(msg);
    } catch (e) {
        postMessage({ type: 'error', id: msg.id, message: e.message ?? String(e) });
    }
};

async function run({ id, spec, particles, until, maxSteps, frameTimes }) {
    const p = new SpecPreview(spec, particles, until, maxSteps);
    try {
        const dt = p.dt();
        const radius = p.view_radius();
        const times = [...new Set(frameTimes)].filter((t) => t >= 0).sort((a, b) => a - b);
        const exposure = { value: null }; // fixed by the first frame, as in headless
        let done = false;
        while (!done) {
            if (current !== id) return; // cancelled, or superseded by a newer run
            // Step up to the next frame time exactly, then draw it.
            const next = times[0];
            let steps = CHUNK_STEPS;
            if (next !== undefined) {
                steps = Math.min(steps, Math.max(0, Math.round((next - p.time()) / dt)));
            }
            if (steps === 0 && next !== undefined) {
                times.shift();
                const blob = await drawFrame(p, radius, exposure);
                postMessage({ type: 'frame', id, time: p.time(), blob });
                continue;
            }
            done = p.advance(steps);
            postMessage({ type: 'progress', id, progress: p.progress(), time: p.time() });
            await new Promise((r) => setTimeout(r, 0)); // let a cancel through
        }
        // Frames asked for past the end (the step cap cut the run short, or
        // a time beyond `until`) show the final state, once.
        if (times.length > 0) {
            const blob = await drawFrame(p, radius, exposure);
            postMessage({ type: 'frame', id, time: p.time(), blob });
        }
        postMessage({ type: 'done', id, report: JSON.parse(p.report()) });
    } finally {
        p.free();
        if (current === id) current = null;
    }
}

// ---------------------------------------------------------------------------
// Frames: a CPU density splat like headless's renderer (crates/headless/src/
// render.rs), small enough to send to Claude: two 256-px views, face-on (xy)
// and edge-on (xz), dark matter hidden, point masses as white dots.
// ---------------------------------------------------------------------------

const SIZE = 256;
const DARK_MATTER = 3;
const POINT_MASS = 5;
// Linear light per particle type, as in render-core/src/color.rs
const COLORS = {
    0: [0.7, 0.85, 1.0], // default
    1: [0.6, 0.75, 1.0], // disk star
    2: [1.0, 0.8, 0.4], // bulge star
    4: [0.5, 0.7, 1.0], // gas
};

function splat(pos, types, radius, axis) {
    const n = SIZE;
    const light = new Float32Array(3 * n * n);
    const scale = n / (2 * radius);
    for (let i = 0; i < types.length; i++) {
        const t = types[i];
        if (t === DARK_MATTER || t === POINT_MASS) continue;
        const px = (pos[3 * i] + radius) * scale - 0.5;
        const py = (radius - pos[3 * i + axis]) * scale - 0.5;
        if (!(px > -1 && py > -1 && px < n && py < n)) continue;
        const x0 = Math.floor(px), y0 = Math.floor(py);
        const fx = px - x0, fy = py - y0;
        const c = COLORS[t] ?? COLORS[0];
        for (const [dx, dy, w] of [
            [0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)],
            [0, 1, (1 - fx) * fy], [1, 1, fx * fy],
        ]) {
            const x = x0 + dx, y = y0 + dy;
            if (x < 0 || y < 0 || x >= n || y >= n) continue;
            const k = 3 * (y * n + x);
            light[k] += w * c[0];
            light[k + 1] += w * c[1];
            light[k + 2] += w * c[2];
        }
    }
    return light;
}

// 99.5th percentile of lit pixels' peak channel maps to white.
function autoExposure(light) {
    const lit = [];
    for (let k = 0; k < light.length; k += 3) {
        const l = Math.max(light[k], light[k + 1], light[k + 2]);
        if (l > 0) lit.push(l);
    }
    if (lit.length === 0) return 1;
    lit.sort((a, b) => a - b);
    return Math.max(lit[Math.floor((lit.length - 1) * 0.995)], 1e-6);
}

// asinh stretch (Lupton et al. 2004): faint tails and bright cores both show.
function toneMap(light, img, x0, exposure) {
    const beta = exposure / 100;
    const norm = Math.asinh(exposure / beta);
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const k = 3 * (y * SIZE + x);
            const l = Math.max(light[k], light[k + 1], light[k + 2]);
            const o = 4 * (y * img.width + x0 + x);
            img.data[o + 3] = 255;
            if (l <= 0) continue;
            const s = Math.min(Math.asinh(l / beta) / norm, 1) / l;
            for (let ch = 0; ch < 3; ch++) {
                img.data[o + ch] = Math.round(255 * Math.min(light[k + ch] * s, 1));
            }
        }
    }
}

async function drawFrame(p, radius, exposure) {
    const pos = p.positions();
    const types = p.types();
    const canvas = new OffscreenCanvas(2 * SIZE + 2, SIZE);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const views = [[1, 0], [2, SIZE + 2]]; // [axis drawn vertically, x offset]: xy, xz
    const lights = views.map(([axis]) => splat(pos, types, radius, axis));
    exposure.value ??= Math.max(...lights.map(autoExposure));
    views.forEach(([, x0], v) => toneMap(lights[v], img, x0, exposure.value));
    ctx.putImageData(img, 0, 0);

    ctx.fillStyle = '#fff';
    const scale = SIZE / (2 * radius);
    views.forEach(([axis, x0]) => {
        for (let i = 0; i < types.length; i++) {
            if (types[i] !== POINT_MASS) continue;
            const px = x0 + (pos[3 * i] + radius) * scale;
            const py = (radius - pos[3 * i + axis]) * scale;
            ctx.beginPath();
            ctx.arc(px, py, 2, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
    ctx.fillStyle = '#334';
    ctx.fillRect(SIZE, 0, 2, SIZE);
    ctx.fillStyle = '#99a';
    ctx.font = '11px sans-serif';
    ctx.fillText(`xy  t = ${p.time().toFixed(1)}`, 6, 14);
    ctx.fillText('xz', SIZE + 8, 14);
    ctx.fillText(`field ±${radius.toPrecision(3)}`, 6, SIZE - 6);
    return canvas.convertToBlob({ type: 'image/png' });
}
