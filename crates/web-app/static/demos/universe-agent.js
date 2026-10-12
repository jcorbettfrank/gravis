// The text-to-universe loop for universe.html: Claude turns a description into
// a scenario spec, previews it (a short run in the visitor's browser: energy
// drift, encounters, frames), reads the result, and revises until it holds
// up. This is the /universe skill's method (.claude/skills/universe/SKILL.md)
// with the shell commands swapped for two tools the page implements. See
// ADR 014.

import { design, inspect } from './universe-orbit.js';

export const MAX_PREVIEWS = 4;
export const MAX_TURNS = 12;
export const PREVIEW_PARTICLES = 1500;
export const PREVIEW_MAX_STEPS = 6000;
export const MAX_UNTIL = 100;
export const ENERGY_PASS = 1e-3; // as in check.py and sim_core::preview
export const ENERGY_WARN = 1e-2;

export const SYSTEM_PROMPT = `You build gravitational N-body scenes for gravis, a galaxy and star-cluster simulator that runs in the visitor's web browser. The visitor describes a scene in plain English. You write a JSON scenario spec, preview it with the run_preview tool, read the numbers and frames, and revise until the physics and the picture both match the description. Then you answer the visitor.

# Method
1. Translate: map each thing in the description to a component (table below). Put the visitor's sentence in the spec's "description" field verbatim and give the spec a short "name".
2. Design the geometry with design_orbit rather than guessing velocities. Encounters are where descriptions succeed or fail: "plunges through" needs a close pericenter, "flyby" one that doesn't merge, "merger" a bound orbit.
3. Preview with run_preview. Choose "until" to cover the story (usually past the first close passage) and 2-4 frame_times at the start, around the expected passage, and after it. List every pair the description says should meet in "expect".
4. Check the result: energy verdict, whether the expected encounters happened and when, and what the frames show. Describe each frame to yourself before deciding. Numbers can pass while the picture is wrong.
5. Revise one or two things at a time (symptom table below), at most ${MAX_PREVIEWS} previews in total. Stop as soon as the scene holds up.

# What "holds up" means
- Energy: max |dE/E| <= 1e-3 passes; up to 1e-2 warns, acceptable for these small previews (${PREVIEW_PARTICLES} particles) as long as you say so; worse fails, and the run can't be trusted: fix it.
- Events: every encounter the description implies happened inside the run, and nothing it rules out did (a "flyby" that merged; a "survives" galaxy whose half-mass radius tripled).
- Frames: the described structure is visible (tails, a shredded dwarf, a stream, a ring) and inside the field of view.

# Spec format (N-body units, G = 1)
Galactic scale: 1 length unit = 1 kpc, 1 mass unit = 1e10 solar masses, 1 time unit ~ 14.9 Myr, 1 velocity unit ~ 65.6 km/s.
Top level: "components" (required, list), "name", "description", "seed" (default 42), "dt", "softening", "view_radius" (derived when omitted; view_radius frames the scene around the center of mass). Unknown fields are rejected and errors name the field. The system is shifted to its center-of-mass frame.
Every component takes "position" and "velocity" ([x, y, z], default origin and at rest); it sits exactly at its position and moves at its velocity.
- disk_galaxy: exponential disk + Plummer bulge + Hernquist dark matter halo. "particles" (required, total), "spin_axis" (disk angular momentum, default [0,0,1]), "baryon_mass" 1.0, "disk_fraction" 0.7, "halo_mass_ratio" 10, "disk_scale_radius" 1.0, "disk_scale_height" 0.1, "bulge_scale_radius" 0.3, "halo_scale_radius" 5.0, "disk_max_radius" 6.0, "halo_max_radius" 30.0, "star_particle_fraction" 0.5. Total mass = baryon_mass x (1 + halo_mass_ratio), 11 by default. A smaller galaxy should shrink with its mass: at a fraction f of the default mass scale every radius by about f^(1/3).
- plummer: a star cluster or spheroid in equilibrium. "particles" (required), "mass" 1.0, "scale_radius" 1.0.
- point_mass: one particle (a black hole, a star). "mass" (required).
- gas_cloud: not available on this page (self-gravitating gas is unreliable past peak compression, a known issue being fixed). Build the closest gravitational scene instead and say so.
The browser runs at most 50,000 particles; previews rescale every spec to ${PREVIEW_PARTICLES}. Write particle counts for the live view: 10,000-30,000 in total is a good range, split roughly in proportion to mass (a dwarf needs at least ~2,000 to be visible).

| Words | Component | Starting point |
|---|---|---|
| Milky Way, spiral galaxy, disk galaxy | disk_galaxy | defaults |
| dwarf, satellite, small galaxy | disk_galaxy | baryon_mass 0.05-0.3, radii x (mass fraction)^(1/3) |
| giant / massive galaxy | disk_galaxy | baryon_mass 2-5, radii x mass^(1/3); lower star_particle_fraction so halo_mass_ratio x f/(1-f) stays near 10 |
| elliptical, spheroid | plummer | not a true elliptical profile; say so |
| globular / star cluster | plummer | mass 1e-4 to 1e-3 of the host, small scale_radius (e.g. 0.05) |
| black hole, massive star, perturber | point_mass | |

# Encounter design
- Put the orbit in the xy-plane (z = 0 positions and velocities); its angular momentum then points along +z (design_orbit's default) or -z (retrograde_orbit).
- Prograde tails: a disk whose spin_axis is parallel to the orbital angular momentum throws out long tidal tails; antiparallel (retrograde) barely responds; tilted is in between. Use tails for drama, retrograde for a disk that should survive.
- Plunge vs graze: a pericenter below the larger disk's scale radius (1 by default) goes through the disk; 2-4 scale radii raises tides without collision; beyond ~15 is a distant flyby.
- Bound or not: eccentricity < 1 is bound and merges within a few passages; ~1 (parabolic) is the classic Toomre setup; > 1 flies apart. Extended halos plus dynamical friction mean even parabolic galaxy pairs usually merge eventually.
- Kepler is exact for point masses and compact clusters. For galaxies the first passage comes 1.2-1.4x later than design_orbit's time_to_pericenter, and wider when the smaller galaxy is much lighter; each later passage is closer and bound pairs merge within two or three. Read the real times off the preview's separation history.
- Start separation: ~10-12 for default galaxies so disks begin undisturbed, but every unit of travel costs preview time.
- spin_axis [0,0,1] puts the disk in the orbital plane; [1,0,0] is edge-on to it (a dwarf moving along x then hits the disk face-on); [0,0.866,0.5] is tilted 60 degrees.
- Default galaxy dt is ~0.015, so until = 40 is ~2,700 steps. A point mass with a tight orbit makes dt much smaller; the preview stops at ${PREVIEW_MAX_STEPS} steps and says so. Keep until <= ${MAX_UNTIL}.

# Symptoms and fixes
- Components never meet: lower the pericenter or relative speed; start closer; run longer (look at the separation history).
- Merged when it should be a flyby: eccentricity above 1, a larger pericenter, relative speed above the escape speed.
- No tidal tails: make the disk prograde; lower the pericenter; give the partner more mass.
- Everything shredded: raise the pericenter; make the disk retrograde or tilted; lower the partner's mass.
- Energy FAIL with a point mass or tight cluster: set dt explicitly to half the derived one, or raise softening.
- Structure off the frame: set view_radius. Everything a tiny blob: lower view_radius.

# Honesty
Gravity only. Not modeled, so never claimed: star formation, starbursts, supernovae or feedback, gas and cooling, magnetic fields, relativity (no accretion, mergers of black holes are just two point masses), cosmic expansion, dark energy, collisions between individual stars. If the description needs one, build the closest gravitational scene and say plainly what is missing. If the request isn't a physical scene at all, say what this page can do instead and don't call the tools.

# Your final answer
After your last preview (always of the spec you are presenting), reply in plain text, no Markdown headings or tables, under 150 words: what the visitor is watching and when to look (in sim time units), the check results in a sentence (energy drift, when the closest approach happened), and what the simulation does not model. The page shows the last previewed spec live; don't paste the JSON.`;

export const TOOLS = [
    {
        name: 'design_orbit',
        description: 'Kepler two-body orbit design (each body a point of its total mass; G = 1). ' +
            'Returns body1/body2 positions and velocities to paste into the two components (body 2 starts on +x of body 1, ' +
            'orbit in the xy-plane, angular momentum +z unless retrograde_orbit), the escape speed, and the orbit: ' +
            'pericenter, eccentricity, bound, time_to_pericenter, period. Use component total masses ' +
            '(a default disk_galaxy is 11).',
        input_schema: {
            type: 'object',
            additionalProperties: false,
            required: ['m1', 'm2', 'pericenter', 'eccentricity', 'separation', 'retrograde_orbit'],
            properties: {
                m1: { type: 'number', description: 'total mass of body 1' },
                m2: { type: 'number', description: 'total mass of body 2' },
                pericenter: { type: 'number', description: 'closest approach (Kepler)' },
                eccentricity: { type: 'number', description: '1 = parabolic, < 1 bound, > 1 unbound' },
                separation: { type: 'number', description: 'starting separation' },
                retrograde_orbit: { type: 'boolean', description: 'orbit clockwise (angular momentum -z)' },
            },
        },
        strict: true,
    },
    {
        name: 'run_preview',
        description: `Validate a scenario spec and run a short preview of it in the visitor's browser ` +
            `(rescaled to ${PREVIEW_PARTICLES} particles, Barnes-Hut). Returns the Kepler orbit of each pair, ` +
            'the energy drift with a PASS/WARN/FAIL verdict, each component\'s growth in visible half-mass radius ' +
            '(x1.3 or less: intact; x3: disrupted), each pair\'s closest approach, completed passages and separation ' +
            'history, PASS/FAIL for every expected encounter, and one image per frame time (left: xy view, ' +
            'right: xz view, dark matter hidden, point masses as white dots). An invalid spec returns the error, ' +
            'naming the field. The visitor sees each previewed spec running live.',
        input_schema: {
            type: 'object',
            additionalProperties: false,
            required: ['spec', 'until', 'frame_times', 'expect'],
            properties: {
                spec: { type: 'object', description: 'the full scenario spec' },
                until: { type: 'number', description: `sim time to run to, at most ${MAX_UNTIL}` },
                frame_times: {
                    type: 'array', items: { type: 'number' },
                    description: '1-4 sim times to render, within [0, until]',
                },
                expect: {
                    type: 'array',
                    description: 'component pairs that must pass close to each other (indices into components)',
                    items: {
                        type: 'object',
                        additionalProperties: false,
                        required: ['a', 'b', 'within'],
                        properties: {
                            a: { type: 'integer' },
                            b: { type: 'integer' },
                            within: {
                                type: 'number',
                                description: 'required closest approach; 0 means their reach, the sum of their ' +
                                    'initial visible half-mass radii (use an explicit distance for scenes that ' +
                                    "shouldn't touch)",
                            },
                        },
                    },
                },
            },
        },
    },
];

const fmt = (x) => (x === null || x === undefined ? 'nan' : Number(x).toPrecision(3).replace(/\.?0+(e|$)/, '$1'));

/**
 * Condense a preview report into the text Claude reads: the same content and
 * verdicts as the skill's check.py prints for a headless summary.
 * Returns {text, verdicts: [[level, message]], ok}.
 */
export function formatReport(report, spec, expect) {
    const lines = [];
    const verdicts = [];
    const r = report;
    lines.push(`${r.name}: ${r.particles} particles` +
        (r.requested_particles > r.particles ? ` (rescaled from ${r.requested_particles})` : '') +
        `, Barnes-Hut theta ${r.theta}, dt ${fmt(r.dt)}, softening ${fmt(r.softening)}, ` +
        `ran to t = ${fmt(r.sim_time)} (${r.steps} steps)` +
        (r.truncated ? `, stopped by the ${PREVIEW_MAX_STEPS}-step cap short of t = ${fmt(r.until)}` : ''));
    if (r.truncated) {
        verdicts.push(['WARN', `step cap reached at t = ${fmt(r.sim_time)}: dt is small (a tight orbit?) or until is long`]);
    }

    const drift = r.max_abs_energy_drift;
    if (drift === null || !Number.isFinite(drift)) {
        verdicts.push(['FAIL', 'energy is not finite: the run blew up']);
    } else {
        const level = drift <= ENERGY_PASS ? 'PASS' : drift <= ENERGY_WARN ? 'WARN' : 'FAIL';
        verdicts.push([level, `energy: max |dE/E| = ${fmt(drift)}`]);
    }

    lines.push('', 'kepler orbits (each component as a point mass):');
    try {
        for (const o of inspect(spec)) {
            if (o.note) {
                lines.push(`  [${o.a}]-[${o.b}] ${o.note}`);
                continue;
            }
            lines.push(`  [${o.a}]-[${o.b}] masses ${o.masses.join(', ')}, separation ${fmt(o.separation)}, ` +
                `speed ${fmt(o.relative_speed)} (escape ${fmt(o.escape_speed)}), ${o.bound ? 'bound' : 'unbound'}, ` +
                `e ${fmt(o.eccentricity)}, pericenter ${fmt(o.pericenter)} at t ${fmt(o.time_to_pericenter)}` +
                (o.period ? `, period ${fmt(o.period)}` : ''));
        }
    } catch (e) {
        lines.push(`  (${e.message})`);
    }

    const first = new Map(r.initial.map((c) => [c.index, c]));
    lines.push('', 'components (visible half-mass radius, start -> end):');
    for (const c of r.components) {
        const r0 = first.get(c.index)?.half_mass_radius;
        const growth = c.kind === 'point_mass' || !r0 ? '' : `  x${(c.half_mass_radius / r0).toFixed(2)}`;
        lines.push(`  [${c.index}] ${c.kind} N=${c.particles} r_half ${r0 ? fmt(r0) + ' -> ' : ''}` +
            `${fmt(c.half_mass_radius)}${growth}`);
    }

    const reach = (e) => {
        const s = [e.a, e.b].reduce((acc, i) => acc + (first.get(i)?.half_mass_radius ?? 0), 0);
        return s > 0 ? s : 0.1 * e.initial_separation;
    };
    if (r.encounters.length) lines.push('', 'pairs (reach = sum of initial half-mass radii):');
    for (const e of r.encounters) {
        const passes = e.pericenters.map((p) => `t=${fmt(p.time)} d=${fmt(p.separation)}`).join(', ') ||
            'none completed';
        const contact = e.history.find(([, d]) => d <= reach(e));
        lines.push(`  [${e.a}]-[${e.b}] start ${fmt(e.initial_separation)}, closest ${fmt(e.min_separation)} ` +
            `at t=${fmt(e.time_of_min_separation)}, end ${fmt(e.final_separation)}`);
        lines.push(`    reach ${fmt(reach(e))}, first within reach: ${contact ? 't=' + fmt(contact[0]) : 'never'}; ` +
            `completed passages: ${passes}`);
        const h = e.history;
        if (h.length) {
            const k = Math.max(1, Math.floor(h.length / 12));
            const pts = h.filter((_, i) => i % k === 0);
            if ((h.length - 1) % k) pts.push(h[h.length - 1]);
            lines.push('    separation: ' + pts.map(([t, d]) => `${fmt(t)}:${fmt(d)}`).join(' '));
        }
    }

    for (const x of expect ?? []) {
        const [a, b] = [x.a, x.b].sort((p, q) => p - q);
        const e = r.encounters.find((e) => e.a === a && e.b === b);
        if (!e) {
            verdicts.push(['FAIL', `encounter [${a}]-[${b}]: no such pair`]);
            continue;
        }
        const [limit, label] = x.within > 0 ? [x.within, 'limit'] : [reach(e), 'reach'];
        const ok = e.min_separation <= limit;
        verdicts.push([ok ? 'PASS' : 'FAIL', `encounter [${a}]-[${b}]: closest ${fmt(e.min_separation)} ` +
            `at t=${fmt(e.time_of_min_separation)} ${ok ? '<=' : '>'} ${label} ${fmt(limit)}`]);
    }

    lines.push('', ...verdicts.map(([l, m]) => `${l}  ${m}`));
    return { text: lines.join('\n'), verdicts, ok: !verdicts.some(([l]) => l === 'FAIL') };
}

/** Check run_preview's input before spending a run on it; returns an error string or null. */
export function checkPreviewInput(input) {
    if (!input || typeof input.spec !== 'object' || input.spec === null || Array.isArray(input.spec)) {
        return 'spec must be a JSON object';
    }
    if (!(input.until > 0 && input.until <= MAX_UNTIL)) return `until must be in (0, ${MAX_UNTIL}]`;
    if (!Array.isArray(input.frame_times) || input.frame_times.length < 1 || input.frame_times.length > 4) {
        return 'frame_times must list 1-4 times';
    }
    if (input.frame_times.some((t) => !(t >= 0 && t <= input.until))) return 'frame_times must lie in [0, until]';
    if (!Array.isArray(input.expect)) return 'expect must be a list (it may be empty)';
    for (const c of input.spec.components ?? []) {
        if (c?.type === 'gas_cloud') {
            return 'gas_cloud is not available on this page: self-gravitating gas is unreliable past peak ' +
                'compression (a known issue being fixed). Build the closest gravitational scene and say so.';
        }
    }
    return null;
}

/**
 * Run the loop. `client` is an Anthropic SDK client; `preview(input)` runs a
 * preview and resolves to {report, frames: [{time, base64}]} or throws with
 * the spec error. `emit(event)` reports progress to the page:
 *   {kind: 'thinking'} | {kind: 'text', text} | {kind: 'design', input, result}
 *   {kind: 'preview-start', n, input} | {kind: 'preview', n, input, report, check, frames}
 *   {kind: 'preview-error', n, input, message} | {kind: 'done', text, spec, check}
 * Resolves to the final event's payload.
 */
export async function runAgent({ client, model, description, preview, emit, signal }) {
    const messages = [{ role: 'user', content: description }];
    let previews = 0;
    let last = null; // {spec, check} of the last successful preview
    const fallbacks = /^claude-(opus|sonnet)-/.test(model);

    for (let turn = 0; turn < MAX_TURNS; turn++) {
        emit({ kind: 'thinking' });
        const params = {
            model,
            max_tokens: 16000,
            system: SYSTEM_PROMPT,
            tools: TOOLS,
            thinking: { type: 'adaptive' },
            output_config: { effort: 'medium' },
            cache_control: { type: 'ephemeral' },
            messages,
        };
        if (fallbacks) {
            params.betas = ['server-side-fallback-2026-07-01'];
            params.fallbacks = 'default';
        }
        const response = await client.beta.messages.create(params, { signal });
        if (response.stop_reason === 'refusal') {
            throw new Error('Claude declined this description' +
                (response.stop_details?.explanation ? `: ${response.stop_details.explanation}` : '.'));
        }
        if (response.stop_reason === 'max_tokens') throw new Error('Claude ran out of output tokens.');
        messages.push({ role: 'assistant', content: response.content });

        const text = response.content.filter((b) => b.type === 'text').map((b) => b.text).join('\n').trim();
        const calls = response.content.filter((b) => b.type === 'tool_use');
        if (calls.length === 0) {
            const done = { kind: 'done', text, spec: last?.spec ?? null, check: last?.check ?? null };
            emit(done);
            return done;
        }
        if (text) emit({ kind: 'text', text });

        const results = [];
        for (const call of calls) {
            results.push(await runTool(call));
        }
        messages.push({ role: 'user', content: results });
    }
    throw new Error(`Claude didn't finish within ${MAX_TURNS} turns.`);

    async function runTool(call) {
        const error = (message) => ({ type: 'tool_result', tool_use_id: call.id, is_error: true, content: message });
        if (call.name === 'design_orbit') {
            try {
                const result = design(call.input);
                emit({ kind: 'design', input: call.input, result });
                return { type: 'tool_result', tool_use_id: call.id, content: JSON.stringify(result, null, 1) };
            } catch (e) {
                return error(e.message);
            }
        }
        if (call.name !== 'run_preview') return error(`unknown tool ${call.name}`);
        if (previews >= MAX_PREVIEWS) {
            return error(`The preview budget (${MAX_PREVIEWS}) is used up. Answer the visitor now; the page ` +
                'shows your last previewed spec.');
        }
        const input = call.input;
        const bad = checkPreviewInput(input);
        if (bad) return error(bad);
        previews += 1;
        const n = previews;
        emit({ kind: 'preview-start', n, input });
        let result;
        try {
            result = await preview(input);
        } catch (e) {
            if (signal?.aborted) throw e;
            emit({ kind: 'preview-error', n, input, message: e.message });
            return error(e.message);
        }
        const check = formatReport(result.report, input.spec, input.expect);
        last = { spec: input.spec, check };
        emit({ kind: 'preview', n, input, report: result.report, check, frames: result.frames });
        const left = MAX_PREVIEWS - previews;
        const content = [{
            type: 'text',
            text: `${check.text}\n\n${left > 0 ? `${left} preview(s) left.` : 'That was the last preview: answer the visitor now.'}`,
        }];
        for (const f of result.frames) {
            content.push({ type: 'text', text: `frame at t = ${fmt(f.time)}:` });
            content.push({ type: 'image', source: { type: 'base64', media_type: 'image/png', data: f.base64 } });
        }
        return { type: 'tool_result', tool_use_id: call.id, content };
    }
}
