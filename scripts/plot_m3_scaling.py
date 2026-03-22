#!/usr/bin/env python3
"""Generate M3 log-log scaling plot from benchmark CSV.

Usage:
    python3 scripts/plot_m3_scaling.py

Reads:  artifacts/benchmarks/m3_scaling.csv
Writes: artifacts/plots/m3_scaling.png
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

def main():
    csv_path = "artifacts/benchmarks/m3_scaling.csv"
    plot_dir = "artifacts/plots"
    os.makedirs(plot_dir, exist_ok=True)

    bf_n, bf_t = [], []
    bh_n, bh_t = [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            t = float(row["wall_time_ms"])
            if row["algorithm"] == "brute-force":
                bf_n.append(n)
                bf_t.append(t)
            else:
                bh_n.append(n)
                bh_t.append(t)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.loglog(bf_n, bf_t, "o-", color="#e74c3c", linewidth=2, markersize=8, label="Brute-force O(N\u00b2)")
    ax.loglog(bh_n, bh_t, "s-", color="#2ecc71", linewidth=2, markersize=8, label="Barnes-Hut O(N log N)")

    # Reference slopes
    n_ref = np.array([1e3, 1e6])
    # N^2 reference (normalized to brute-force data)
    if bf_n and bf_t:
        c2 = bf_t[0] / (bf_n[0] ** 2)
        ax.loglog(n_ref, c2 * n_ref**2, "--", color="#e74c3c", alpha=0.3, label="N\u00b2 reference")
    # N log N reference (normalized to BH data)
    if bh_n and bh_t:
        c1 = bh_t[0] / (bh_n[0] * np.log2(bh_n[0]))
        ax.loglog(n_ref, c1 * n_ref * np.log2(n_ref), "--", color="#2ecc71", alpha=0.3, label="N log N reference")

    ax.set_xlabel("Number of particles (N)", fontsize=13)
    ax.set_ylabel("Time per force evaluation (ms)", fontsize=13)
    ax.set_title("M3: Brute-Force vs Barnes-Hut Scaling", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=11)

    out_path = os.path.join(plot_dir, "m3_scaling.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
