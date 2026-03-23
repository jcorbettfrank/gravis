//! Scenario factory: maps CLI scenario names to configured scenario instances.
//!
//! This centralizes the name → scenario mapping so that headless, native-app,
//! and web-app don't each maintain their own copy.

use crate::particle::Particles;
use crate::scenario::Scenario;
use crate::scenarios::cold_collapse::ColdCollapse;
use crate::scenarios::evrard_collapse::EvrardCollapse;
use crate::scenarios::galaxy_collision::GalaxyCollision;
use crate::scenarios::kelvin_helmholtz::KelvinHelmholtz;
use crate::scenarios::plummer_sphere::PlummerSphere;
use crate::scenarios::protoplanetary::Protoplanetary;
use crate::scenarios::sedov_blast::SedovBlast;
use crate::scenarios::sod_shock::SodShockTube;
use crate::scenarios::two_body::TwoBody;

/// Configuration passed from CLI or UI to the scenario factory.
pub struct ScenarioConfig {
    pub particles: Option<usize>,
    pub seed: u64,
    pub eccentricity: f64,
}

/// Result of building a scenario: particles + simulation parameters.
pub struct BuiltScenario {
    pub particles: Particles,
    pub dt: f64,
    pub softening: f64,
    pub name: String,
}

/// Create a scenario by name, generate particles, and return simulation parameters.
///
/// `dt_override` and `softening_override` allow CLI flags to take precedence
/// over the scenario's suggested values.
pub fn build(
    scenario_name: &str,
    config: &ScenarioConfig,
    dt_override: Option<f64>,
    softening_override: Option<f64>,
) -> Result<BuiltScenario, String> {
    let scenario: Box<dyn Scenario> = match scenario_name {
        "plummer" => {
            let n = config.particles.unwrap_or(1000);
            Box::new(PlummerSphere { n, seed: config.seed, ..Default::default() })
        }
        "two-body" | "kepler" => {
            Box::new(TwoBody { eccentricity: config.eccentricity, ..Default::default() })
        }
        "cold-collapse" => {
            let n = config.particles.unwrap_or(5000);
            Box::new(ColdCollapse { n, seed: config.seed, ..Default::default() })
        }
        "cold-collapse-gas" => {
            let n = config.particles.unwrap_or(5000);
            Box::new(ColdCollapse { n, sph: true, seed: config.seed, ..Default::default() })
        }
        "galaxy-collision" => {
            let n = config.particles.unwrap_or(10000);
            Box::new(GalaxyCollision { n_per_galaxy: n / 2, seed: config.seed, ..Default::default() })
        }
        "sod-shock" => Box::new(SodShockTube::default()),
        "sedov-blast" => {
            let n = config.particles.unwrap_or(5000);
            Box::new(SedovBlast { n_particles: n, seed: config.seed, ..Default::default() })
        }
        "evrard-collapse" => {
            let n = config.particles.unwrap_or(5000);
            Box::new(EvrardCollapse { n_particles: n, seed: config.seed, ..Default::default() })
        }
        "kelvin-helmholtz" | "kh" => Box::new(KelvinHelmholtz::default()),
        "protoplanetary" => {
            let n = config.particles.unwrap_or(5000);
            Box::new(Protoplanetary { n_gas: n, seed: config.seed, ..Default::default() })
        }
        other => {
            return Err(format!(
                "Unknown scenario: {other}. Available: plummer, two-body, cold-collapse, \
                 cold-collapse-gas, galaxy-collision, sod-shock, sedov-blast, evrard-collapse, \
                 kelvin-helmholtz, protoplanetary"
            ));
        }
    };

    let particles = scenario.generate();
    let dt = dt_override.unwrap_or_else(|| scenario.suggested_dt());
    let softening = softening_override.unwrap_or_else(|| scenario.suggested_softening());
    let name = scenario.name().to_string();

    Ok(BuiltScenario { particles, dt, softening, name })
}

/// Returns true for SPH scenarios that are purely hydrodynamic (no gravity needed).
///
/// Sod shock, Sedov blast, and Kelvin-Helmholtz are pure hydro tests.
/// Evrard collapse, cold-collapse-gas, and protoplanetary use SPH but require gravity.
pub fn is_pure_hydro(scenario_name: &str) -> bool {
    matches!(scenario_name, "sod-shock" | "sedov-blast" | "kelvin-helmholtz" | "kh")
}
