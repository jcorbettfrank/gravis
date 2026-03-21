use crate::particle::Particles;

/// A scenario generates initial conditions for an N-body simulation.
///
/// All scenarios must be deterministic: the same parameters always produce
/// the same initial conditions, enabling reproducible experiments and
/// artifact generation.
pub trait Scenario {
    /// Human-readable name for this scenario.
    fn name(&self) -> &str;

    /// Generate initial conditions.
    fn generate(&self) -> Particles;

    /// Suggested timestep for this scenario.
    fn suggested_dt(&self) -> f64;

    /// Suggested softening for this scenario.
    fn suggested_softening(&self) -> f64;
}
