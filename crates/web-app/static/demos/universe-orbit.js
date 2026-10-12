// Two-body orbit helper (N-body units, G = 1): a port of the /universe skill's
// .claude/skills/universe/scripts/orbit.py for the text-to-universe page, so
// Claude gets the same geometry arithmetic in the browser that it gets from
// the script in Claude Code. Keep the two in step.
//
// Both treat each component as a point of its total mass. Galaxies are
// extended, so their first passage comes ~1.2-1.4x later than Kepler says.

const DISK_DEFAULTS = { baryon_mass: 1.0, halo_mass_ratio: 10.0 };

export function componentMass(c) {
    switch (c.type) {
        case 'disk_galaxy':
            return (c.baryon_mass ?? DISK_DEFAULTS.baryon_mass) *
                (1 + (c.halo_mass_ratio ?? DISK_DEFAULTS.halo_mass_ratio));
        case 'plummer':
        case 'gas_cloud':
            return c.mass ?? 1.0;
        case 'point_mass':
            return c.mass;
        default:
            throw new Error(`unknown component type ${JSON.stringify(c.type)}`);
    }
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const round = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

function timeToPericenter(mu, q, e, r, rv) {
    if (e < 1e-9) return 0;
    const p = q * (1 + e);
    const cosF = Math.max(-1, Math.min(1, (p / r - 1) / e));
    const f = Math.acos(cosF);
    const approaching = rv < 0;
    let t;
    if (Math.abs(e - 1) < 1e-9) {
        const d = Math.tan(f / 2);
        t = 0.5 * Math.sqrt(p ** 3 / mu) * (d + d ** 3 / 3);
    } else if (e < 1) {
        const a = q / (1 - e);
        const n = Math.sqrt(mu / a ** 3);
        const bigE = Math.acos(Math.max(-1, Math.min(1, (e + cosF) / (1 + e * cosF))));
        t = (bigE - e * Math.sin(bigE)) / n;
        if (!approaching) return 2 * Math.PI / n - t; // next pericenter, one period on
    } else {
        const a = q / (e - 1);
        const n = Math.sqrt(mu / a ** 3);
        const bigF = Math.acosh(Math.max(1, (e + cosF) / (1 + e * cosF)));
        t = (e * Math.sinh(bigF) - bigF) / n;
    }
    return approaching ? t : 0;
}

/** Orbital elements of a relative orbit. */
export function elements(mu, r, v) {
    const rr = Math.sqrt(dot(r, r));
    const h = [r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2], r[0] * v[1] - r[1] * v[0]];
    const h2 = dot(h, h);
    const energy = 0.5 * dot(v, v) - mu / rr;
    const e = Math.sqrt(Math.max(0, 1 + 2 * energy * h2 / (mu * mu)));
    const q = h2 / mu / (1 + e);
    return { q, e, energy, h, tPeri: timeToPericenter(mu, q, e, rr, dot(r, v)) };
}

function describe(mu, { q, e, energy, h, tPeri }) {
    const d = {
        pericenter: round(q, 6),
        eccentricity: round(e, 6),
        bound: energy < 0,
        time_to_pericenter: round(tPeri, 4),
        orbital_angular_momentum: h.map((x) => round(x, 6)),
    };
    if (energy < 0 && e < 0.999) {
        const a = q / (1 - e);
        d.apocenter = round(a * (1 + e), 6);
        d.period = round(2 * Math.PI * Math.sqrt(a ** 3 / mu), 4);
    }
    return d;
}

/**
 * Positions and velocities for two bodies on a Kepler orbit in the xy-plane,
 * body 2 starting on +x of body 1, orbiting counterclockwise (angular
 * momentum +z) unless `retrograde_orbit`. Throws on impossible geometry.
 */
export function design({ m1, m2, pericenter: q, eccentricity: e = 1, separation: r0, retrograde_orbit = false }) {
    const mu = m1 + m2;
    if (!(m1 > 0 && m2 > 0)) throw new Error('masses must be > 0');
    if (!(q > 0 && r0 > 0 && e >= 0)) throw new Error('pericenter and separation must be > 0, eccentricity >= 0');
    if (r0 < q) throw new Error(`separation ${r0} is inside the pericenter ${q}`);
    if (e > 0 && e < 1) {
        const apo = q * (1 + e) / (1 - e);
        if (r0 > apo * (1 + 1e-12)) {
            throw new Error(`separation ${r0} is beyond the apocenter ${apo.toPrecision(4)}; ` +
                'raise the eccentricity or start closer');
        }
    }
    const p = q * (1 + e);
    const f = e < 1e-9 ? 0 : -Math.acos(Math.max(-1, Math.min(1, (p / r0 - 1) / e))); // incoming
    const [rx, ry] = [r0 * Math.cos(f), r0 * Math.sin(f)];
    const s = Math.sqrt(mu / p);
    const [vx, vy] = [-s * Math.sin(f), s * (e + Math.cos(f))];
    const [c, sn] = [Math.cos(-f), Math.sin(-f)];
    const relR = [rx * c - ry * sn, rx * sn + ry * c, 0];
    const relV = [vx * c - vy * sn, vx * sn + vy * c, 0];
    if (retrograde_orbit) {
        relR[1] = -relR[1];
        relV[1] = -relV[1];
    }
    const el = elements(mu, relR, relV);
    const [f1, f2] = [m2 / mu, m1 / mu];
    // `+ 0` turns -0 into 0, as Python's json prints it
    const scaled = (vec, k) => vec.map((x) => round(k * x, 6) + 0);
    return {
        body1: { position: scaled(relR, -f1), velocity: scaled(relV, -f1) },
        body2: { position: scaled(relR, f2), velocity: scaled(relV, f2) },
        relative_speed: round(Math.sqrt(dot(relV, relV)), 6),
        escape_speed: round(Math.sqrt(2 * mu / r0), 6),
        orbit: describe(mu, el),
    };
}

/** Kepler orbit of every pair of components in a spec. */
export function inspect(spec) {
    const comps = spec.components ?? [];
    const rows = [];
    for (let i = 0; i < comps.length; i++) {
        for (let j = i + 1; j < comps.length; j++) {
            const [a, b] = [comps[i], comps[j]];
            const [ma, mb] = [componentMass(a), componentMass(b)];
            const [ra, rb] = [a.position ?? [0, 0, 0], b.position ?? [0, 0, 0]];
            const [va, vb] = [a.velocity ?? [0, 0, 0], b.velocity ?? [0, 0, 0]];
            const r = rb.map((x, k) => x - ra[k]);
            const v = vb.map((x, k) => x - va[k]);
            const sep = Math.sqrt(dot(r, r));
            if (sep === 0) {
                rows.push({ a: i, b: j, note: 'same position' });
                continue;
            }
            rows.push({
                a: i, b: j,
                masses: [round(ma, 6), round(mb, 6)],
                separation: round(sep, 6),
                relative_speed: round(Math.sqrt(dot(v, v)), 6),
                escape_speed: round(Math.sqrt(2 * (ma + mb) / sep), 6),
                ...describe(ma + mb, elements(ma + mb, r, v)),
            });
        }
    }
    return rows;
}
