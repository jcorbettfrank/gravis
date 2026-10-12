#!/usr/bin/env python3
"""Condense a headless summary.json into the checks the /universe loop needs.

Prints energy conservation, each component's size growth (a disruption
signal), every pair's close passages, and the rendered frames, then a verdict
line per check: PASS, WARN or FAIL. Exit status is 1 if any check FAILs.

    python3 check.py artifacts/universe/<slug>/summary.json [--expect-encounter 0,1]

--expect-encounter A,B[,D] (repeatable) fails the run unless the centers of
components A and B came within D of each other: "the dwarf actually reached
the disk". D defaults to their reach, the sum of their initial visible
half-mass radii; give it for scenes where the bodies shouldn't touch, such as
a cluster on a pericenter of 0.5 around a black hole. Standard library only.
"""

import argparse
import json
import math
import sys

# |dE/E| thresholds. Leapfrog + Barnes-Hut (theta 0.5) at a few thousand
# particles typically lands near 1e-4..1e-3 over a galaxy encounter.
DRIFT_PASS = 1e-3
DRIFT_WARN = 1e-2


def fmt(x):
    return f"{x:.3g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary")
    ap.add_argument("--expect-encounter", action="append", default=[],
                    metavar="A,B[,D]", help="components that must pass close to each other")
    args = ap.parse_args()
    with open(args.summary) as fh:
        s = json.load(fh)

    verdicts = []

    def verdict(level, msg):
        verdicts.append((level, msg))

    print(f"{s['scenario']}: N={s['particles']} {s['algorithm']} "
          f"dt={fmt(s['dt'])} eps={fmt(s['softening'])} "
          f"t={fmt(s['sim_time'])} ({s['steps']} steps, {s['wall_seconds']:.0f} s wall)")

    total = s["final_state"]["total"]
    drift = s["max_abs_energy_drift"]
    if total is None or drift is None or not math.isfinite(total):
        # serde_json writes NaN and infinity as null
        verdict("FAIL", "energy is not finite: the run blew up")
        drift = math.inf
    level = "PASS" if drift <= DRIFT_PASS else "WARN" if drift <= DRIFT_WARN else "FAIL"
    if math.isfinite(drift):
        verdict(level, f"energy: max |dE/E| = {fmt(drift)} "
                       f"(final {fmt(s['final_state']['relative_drift'])}), "
                       f"|dL/L| = {fmt(s['angular_momentum_drift'])}")
    if s["particle_types"].get("gas", 0) > 0:
        verdict("WARN", "gas present: self-gravitating SPH is known to blow up at "
                        "peak compression; trust only the run before it")

    frames = s.get("frames", [])
    first = {c["index"]: c for c in frames[0]["components"]} if frames else {}
    print("\ncomponents (visible half-mass radius, first frame -> end):")
    for c in s["components"]:
        r0 = first.get(c["index"], {}).get("half_mass_radius")
        r1 = c["half_mass_radius"]
        if c["kind"] == "point_mass" or not r0:
            growth = ""
        else:
            growth = f"  x{r1 / r0:.2f}"
        start = f"{fmt(r0)} -> " if r0 else ""
        print(f"  [{c['index']}] {c['kind']:<12} N={c['particles']:<7} "
              f"r_half {start}{fmt(r1)}{growth}")

    def reach(e):
        r = sum(first.get(i, {}).get("half_mass_radius", 0.0) for i in (e["a"], e["b"]))
        return r if r > 0 else 0.1 * e["initial_separation"]

    print("\npairs (reach = sum of initial half-mass radii):")
    for e in s.get("encounters", []):
        a, b = e["a"], e["b"]
        passes = ", ".join(f"t={fmt(p['time'])} d={fmt(p['separation'])}"
                           for p in e.get("pericenters", [])) or "none completed"
        hist = e.get("history", [])
        contact = next((t for t, d in hist if d <= reach(e)), None)
        print(f"  [{a}]-[{b}] start {fmt(e['initial_separation'])}, "
              f"closest {fmt(e['min_separation'])} at t={fmt(e['time_of_min_separation'])}, "
              f"end {fmt(e['final_separation'])}")
        print(f"    reach {fmt(reach(e))}, first within reach: "
              f"{'t=' + fmt(contact) if contact is not None else 'never'}; "
              f"completed passages: {passes}")
        if hist:
            k = max(1, len(hist) // 12)
            pts = hist[::k] + ([hist[-1]] if (len(hist) - 1) % k else [])
            print("    separation: " + " ".join(f"{fmt(t)}:{fmt(d)}" for t, d in pts))

    for spec in args.expect_encounter:
        parts = spec.split(",")
        a, b = sorted(int(x) for x in parts[:2])
        limit = float(parts[2]) if len(parts) > 2 else None
        e = next((e for e in s["encounters"] if e["a"] == a and e["b"] == b), None)
        if e is None:
            verdict("FAIL", f"encounter [{a}]-[{b}]: no such pair")
            continue
        if limit is None:
            limit, label = reach(e), "reach"
        else:
            label = "limit"
        ok = e["min_separation"] <= limit
        verdict("PASS" if ok else "FAIL",
                f"encounter [{a}]-[{b}]: closest {fmt(e['min_separation'])} "
                f"at t={fmt(e['time_of_min_separation'])} "
                f"{'<=' if ok else '>'} {label} {fmt(limit)}")

    if frames:
        print("\nframes:")
        for f in frames:
            d = f["energy"]["relative_drift"]
            print(f"  t={fmt(f['time'])} dE/E={fmt(d) if d is not None else 'nan'}  " + "  ".join(f["images"]))
    else:
        verdict("WARN", "no frames rendered: pass --render-at")

    print()
    for level, msg in verdicts:
        print(f"{level}  {msg}")
    sys.exit(1 if any(level == "FAIL" for level, _ in verdicts) else 0)


if __name__ == "__main__":
    main()
