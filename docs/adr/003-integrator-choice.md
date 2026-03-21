# ADR 003: Leapfrog (KDK) Integrator

**Date**: 2026-03-21
**Status**: Accepted

## Decision

Use the Kick-Drift-Kick leapfrog (velocity Verlet) as the primary time integrator. Yoshida 4th-order is a planned future addition.

## Context

The choice of integrator for gravitational N-body simulation is critical. The key property is **symplecticity**: a symplectic integrator exactly preserves a "shadow Hamiltonian" close to the true Hamiltonian, so energy errors are bounded and oscillating rather than growing secularly.

**RK4** (Runge-Kutta 4th order) is 4th-order accurate per step but not symplectic. Over long integrations, it shows linear energy drift proportional to time. This makes it unsuitable for simulations that need to run for thousands of dynamical times.

**Leapfrog/Verlet** is only 2nd-order accurate per step but symplectic. Energy errors oscillate with bounded amplitude proportional to dt². For a simulation running indefinitely, bounded errors beat drifting errors regardless of order.

The KDK (Kick-Drift-Kick) form keeps positions and velocities synchronized at integer timesteps, which simplifies diagnostics, snapshot output, and comparison between runs.

## Verification

The two-body Kepler orbit test confirms:
- Energy conserved to dE/E < 10^-14 after one full orbit (returns to exact initial energy)
- Energy conserved to dE/E < 10^-4 after 1000 orbits with e=0.5
- Angular momentum conserved to machine precision
- Orbital period matches Kepler's third law to <0.01%

## Future

Yoshida 4th-order composes three leapfrog steps with specific coefficients to achieve 4th-order accuracy while preserving symplecticity. This is planned for M8 (stretch goals).
