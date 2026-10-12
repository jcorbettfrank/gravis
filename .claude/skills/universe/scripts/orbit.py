#!/usr/bin/env python3
"""Two-body orbit helper for scenario specs (N-body units, G = 1).

design:  pick an orbit (pericenter, eccentricity, starting separation) and get
         the relative position and velocity to put in a spec, plus the time
         to first pericenter.
inspect: read a spec and report the Keplerian orbit of every pair of
         components (pericenter, eccentricity, bound or not, time to
         pericenter), so a hand-written spec can be checked before a run.

Both treat each component as a point of its total mass. Galaxies are extended
(the default halo reaches r = 30), so once halos overlap, the real orbit
decays faster than Kepler says (dynamical friction): treat the pericenter
time as the first close passage, and expect bound orbits to merge within a
few passages. Standard library only.
"""

import argparse
import json
import math
import sys

# Defaults from specs/README.md, for components that omit fields
DISK_DEFAULTS = {"baryon_mass": 1.0, "halo_mass_ratio": 10.0}


def component_mass(c):
    kind = c.get("type")
    if kind == "disk_galaxy":
        mb = c.get("baryon_mass", DISK_DEFAULTS["baryon_mass"])
        ratio = c.get("halo_mass_ratio", DISK_DEFAULTS["halo_mass_ratio"])
        return mb * (1.0 + ratio)
    if kind in ("plummer", "gas_cloud"):
        return c.get("mass", 1.0)
    if kind == "point_mass":
        return c["mass"]
    raise ValueError(f"unknown component type {kind!r}")


def elements(mu, r_vec, v_vec):
    """Orbital elements of a relative orbit: (q, e, energy, h_z, t_peri).

    t_peri is the time until the next pericenter (0 if moving away from it
    on an unbound orbit, i.e. the closest approach is already behind).
    """
    r = math.sqrt(sum(x * x for x in r_vec))
    v2 = sum(x * x for x in v_vec)
    rv = sum(a * b for a, b in zip(r_vec, v_vec))
    hx = r_vec[1] * v_vec[2] - r_vec[2] * v_vec[1]
    hy = r_vec[2] * v_vec[0] - r_vec[0] * v_vec[2]
    hz = r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]
    h2 = hx * hx + hy * hy + hz * hz
    energy = 0.5 * v2 - mu / r
    e = math.sqrt(max(0.0, 1.0 + 2.0 * energy * h2 / (mu * mu)))
    p = h2 / mu
    q = p / (1.0 + e)
    return q, e, energy, (hx, hy, hz), time_to_pericenter(mu, q, e, r, rv)


def time_to_pericenter(mu, q, e, r, rv):
    if e < 1e-9:
        return 0.0
    p = q * (1.0 + e)
    cos_f = max(-1.0, min(1.0, (p / r - 1.0) / e))
    f = math.acos(cos_f)  # magnitude of the true anomaly
    approaching = rv < 0.0
    if abs(e - 1.0) < 1e-9:
        d = math.tan(f / 2.0)
        t = 0.5 * math.sqrt(p ** 3 / mu) * (d + d ** 3 / 3.0)
    elif e < 1.0:
        a = q / (1.0 - e)
        n = math.sqrt(mu / a ** 3)
        big_e = math.acos(max(-1.0, min(1.0, (e + cos_f) / (1.0 + e * cos_f))))
        t = (big_e - e * math.sin(big_e)) / n
        if not approaching:
            return 2.0 * math.pi / n - t  # next pericenter, one period on
    else:
        a = q / (e - 1.0)
        n = math.sqrt(mu / a ** 3)
        big_f = math.acosh(max(1.0, (e + cos_f) / (1.0 + e * cos_f)))
        t = (e * math.sinh(big_f) - big_f) / n
    return t if approaching else 0.0


def design(args):
    m1, m2 = args.m1, args.m2
    mu = m1 + m2
    q, e, r0 = args.pericenter, args.eccentricity, args.separation
    if q <= 0 or r0 <= 0 or e < 0:
        sys.exit("pericenter and separation must be > 0, eccentricity >= 0")
    if r0 < q:
        sys.exit(f"separation {r0} is inside the pericenter {q}")
    if e < 1.0 and e > 0.0:
        apo = q * (1.0 + e) / (1.0 - e)
        if r0 > apo * (1 + 1e-12):
            sys.exit(f"separation {r0} is beyond the apocenter {apo:.4g}; "
                     "raise the eccentricity or start closer")
    p = q * (1.0 + e)
    if e < 1e-9:
        f = 0.0
    else:
        f = -math.acos(max(-1.0, min(1.0, (p / r0 - 1.0) / e)))  # incoming
    # Perifocal frame (pericenter on +x), then rotate so the start is on +x.
    rx, ry = r0 * math.cos(f), r0 * math.sin(f)
    s = math.sqrt(mu / p)
    vx, vy = -s * math.sin(f), s * (e + math.cos(f))
    c, sn = math.cos(-f), math.sin(-f)
    rel_r = [rx * c - ry * sn, rx * sn + ry * c, 0.0]
    rel_v = [vx * c - vy * sn, vx * sn + vy * c, 0.0]
    if args.retrograde_orbit:
        rel_r[1], rel_v[1] = -rel_r[1], -rel_v[1]
    _, _, energy, h, t_peri = elements(mu, rel_r, rel_v)

    # Body 2 relative to body 1; split about the center of mass.
    f1, f2 = m2 / mu, m1 / mu
    out = {
        "body1": {"position": [round(-f1 * x, 6) for x in rel_r],
                  "velocity": [round(-f1 * x, 6) for x in rel_v]},
        "body2": {"position": [round(f2 * x, 6) for x in rel_r],
                  "velocity": [round(f2 * x, 6) for x in rel_v]},
        "relative_speed": round(math.sqrt(sum(x * x for x in rel_v)), 6),
        "escape_speed": round(math.sqrt(2 * mu / r0), 6),
        "orbit": describe(mu, q, e, energy, h, t_peri),
    }
    print(json.dumps(out, indent=2))


def describe(mu, q, e, energy, h, t_peri):
    d = {
        "pericenter": round(q, 6),
        "eccentricity": round(e, 6),
        "bound": energy < 0,
        "time_to_pericenter": round(t_peri, 4),
        "orbital_angular_momentum": [round(x, 6) for x in h],
    }
    if energy < 0 and e < 0.999:  # skip the huge apocenter of a ~parabolic orbit
        a = q / (1 - e)
        d["apocenter"] = round(a * (1 + e), 6)
        d["period"] = round(2 * math.pi * math.sqrt(a ** 3 / mu), 4)
    return d


def inspect(args):
    with open(args.spec) as fh:
        spec = json.load(fh)
    comps = spec["components"]
    rows = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            ma, mb = component_mass(a), component_mass(b)
            ra, rb = a.get("position", [0, 0, 0]), b.get("position", [0, 0, 0])
            va, vb = a.get("velocity", [0, 0, 0]), b.get("velocity", [0, 0, 0])
            r = [y - x for x, y in zip(ra, rb)]
            v = [y - x for x, y in zip(va, vb)]
            sep = math.sqrt(sum(x * x for x in r))
            if sep == 0:
                rows.append({"a": i, "b": j, "note": "same position"})
                continue
            q, e, energy, h, t = elements(ma + mb, r, v)
            row = {"a": i, "b": j, "masses": [round(ma, 6), round(mb, 6)],
                   "separation": round(sep, 6),
                   "relative_speed": round(math.sqrt(sum(x * x for x in v)), 6),
                   "escape_speed": round(math.sqrt(2 * (ma + mb) / sep), 6)}
            row.update(describe(ma + mb, q, e, energy, h, t))
            rows.append(row)
    print(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design", help="relative position/velocity for an orbit")
    d.add_argument("--m1", type=float, required=True, help="mass of body 1 (e.g. 11 for a default disk galaxy)")
    d.add_argument("--m2", type=float, required=True, help="mass of body 2")
    d.add_argument("--pericenter", type=float, required=True, help="closest approach (Kepler)")
    d.add_argument("--eccentricity", type=float, default=1.0, help="1 = parabolic (default), <1 bound, >1 unbound")
    d.add_argument("--separation", type=float, required=True, help="starting separation")
    d.add_argument("--retrograde-orbit", action="store_true",
                   help="orbit clockwise (angular momentum along -z) instead of +z")
    d.set_defaults(func=design)
    i = sub.add_parser("inspect", help="Kepler orbit of every component pair in a spec")
    i.add_argument("spec")
    i.set_defaults(func=inspect)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
