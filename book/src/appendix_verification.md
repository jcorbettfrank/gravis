# Verification Tests

This appendix collects all test problems with their analytical solutions and acceptance criteria. Every test can be reproduced with `cargo test`.

## M1 Tests (Brute-Force Gravity + Leapfrog)

### Two-Body Kepler Orbit

**Setup**: Two masses $m_1 = m_2 = 0.5$ in elliptical orbit, eccentricity $e$, semi-major axis $a = 1$.

**Analytical**: Period $T = 2\pi\sqrt{a^3/(G M_{\text{total}})} = 2\pi$ ([Kepler's third law](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Third_law)). Total energy $E = -G m_1 m_2 / (2a) = -0.125$ for zero softening (from the [specific orbital energy](https://en.wikipedia.org/wiki/Specific_orbital_energy) of a Keplerian orbit). Angular momentum $L = \mu \sqrt{G M_{\text{total}} a (1-e^2)}$ where $\mu = m_1 m_2 / M_{\text{total}} = 0.25$ is the [reduced mass](https://en.wikipedia.org/wiki/Reduced_mass).

| Test | Command | Criterion |
|------|---------|-----------|
| Energy conservation (1000 orbits, e=0.5) | `cargo test -p sim-core --release -- energy_conservation_1000_orbits` | $\|\Delta E/E\| < 10^{-4}$ |
| Angular momentum (100 orbits, e=0.7) | `cargo test -p sim-core --release -- angular_momentum_conservation` | $\|\Delta L/L\| < 10^{-6}$ |
| Orbital period (10 orbits, e=0.3) | `cargo test -p sim-core --release -- kepler_third_law_period` | $\|T_{\text{meas}} - 2\pi\|/2\pi < 10^{-4}$ |

### Plummer Sphere Equilibrium

**Setup**: $N$ particles sampled from the [Plummer distribution function](https://en.wikipedia.org/wiki/Plummer_model) ([Aarseth, Henon & Wielen 1974](https://ui.adsabs.harvard.edu/abs/1974A%26A....37..183A)). Scale radius $a = 1$, total mass $M = 1$.

**Analytical**: [Virial equilibrium](https://en.wikipedia.org/wiki/Virial_theorem) $2K/|U| = 1$. Half-mass radius $r_h \approx 1.305 a$ (from inverting the Plummer cumulative mass profile $M(r) = M r^3 / (r^2 + a^2)^{3/2}$ at $M(r_h) = M/2$).

| Test | Command | Criterion |
|------|---------|-----------|
| Virial equilibrium (N=500, 20 $t_{\text{dyn}}$) | `cargo test -p sim-core --release -- virial_equilibrium` | Mean $2K/\|U\| \in [0.85, 1.15]$ |
| Energy conservation (N=300, 50 $t_{\text{dyn}}$) | `cargo test -p sim-core --release -- energy_conservation` | $\|\Delta E/E\| < 10^{-3}$ |

### Conservation Laws

| Test | Command | Criterion |
|------|---------|-----------|
| Linear momentum (Plummer N=200) | `cargo test -p sim-core --release -- linear_momentum_conservation_plummer` | $\|\vec{p}\| < 10^{-12}$ |
| Linear momentum (two-body) | `cargo test -p sim-core --release -- linear_momentum_conservation_two_body` | $\|\vec{p}\| < 10^{-14}$ |
| Angular momentum (two-body e=0.8) | `cargo test -p sim-core --release -- angular_momentum_conservation_two_body` | $\|\Delta L/L\| < 10^{-8}$ |
| COM drift (Plummer N=200) | `cargo test -p sim-core --release -- com_drift` | $\|\vec{r}_{\text{COM}}\| < 10^{-12}$ |

### Snapshot Restart

| Test | Command | Criterion |
|------|---------|-----------|
| Restore + initialize matches fresh run | `cargo test -p sim-core --release -- restore_then_initialize` | Position difference $< 10^{-14}$ |

## M3 Tests (Barnes-Hut + Rayon)

### Force Accuracy

**Setup**: Plummer sphere with $N = 1000$, $\epsilon = 0.05$. Compute accelerations with both brute-force (exact reference) and Barnes-Hut at various $\theta$.

**Analytical**: At $\theta = 0$, Barnes-Hut opens every node and must match brute-force exactly. At $\theta > 0$, the [monopole approximation](https://en.wikipedia.org/wiki/Multipole_expansion) introduces error that grows with $\theta$.

| Test | Command | Criterion |
|------|---------|-----------|
| Force accuracy ($\theta = 0.5$) | `cargo test -p sim-core --release -- force_accuracy_theta_05` | RMS relative error $< 1\%$ |
| Force accuracy ($\theta = 0.3$) | `cargo test -p sim-core --release -- force_accuracy_theta_03` | RMS relative error $< 0.5\%$ |
| Monotonic error vs $\theta$ | `cargo test -p sim-core --release -- force_accuracy_monotonic` | Error at $\theta_{i+1} \geq$ error at $\theta_i$ |
| $\theta = 0$ matches brute-force | `cargo test -p sim-core --release -- theta_zero_matches_brute_force` | Per-particle difference $< 10^{-12}$ |

### Energy Conservation (Barnes-Hut)

**Setup**: Same Kepler and Plummer setups as M1, but using the Barnes-Hut solver. Tolerances are wider than brute-force because the approximate forces introduce a small non-conservative component.

| Test | Command | Criterion |
|------|---------|-----------|
| Kepler orbit (100 orbits, $\theta = 0.3$) | `cargo test -p sim-core --release -- energy_conservation_kepler_bh` | $\|\Delta E/E\| < 10^{-3}$ |
| Plummer virial (10 $t_{\text{dyn}}$, $\theta = 0.5$) | `cargo test -p sim-core --release -- virial_equilibrium_plummer_bh` | $\|2K/\|U\| - 1\| < 0.5$, $\|\Delta E/E\| < 1\%$ |

### Scaling

**Setup**: Barnes-Hut force evaluation at $N = 1000, 2000, 5000, 10000, 20000$ with $\theta = 0.5$. Measure wall-clock time and fit [log-log slope](https://en.wikipedia.org/wiki/Log%E2%80%93log_plot).

| Test | Command | Criterion |
|------|---------|-----------|
| $O(N \log N)$ scaling | `cargo test -p sim-core --release -- barnes_hut_scaling_is_nlogn` | Log-log slope $\in [0.8, 1.8]$ |

## Future Tests

Tests added in later milestones will be documented here:

- **M7**: Sod shock tube, Sedov-Taylor blast wave, Evrard collapse, Kelvin-Helmholtz instability
