use clap::Parser;
use winit::event_loop::EventLoop;

mod app;
mod render;
mod sim_thread;

#[derive(Parser, Clone)]
#[command(name = "nbody-native")]
#[command(about = "Real-time N-body simulation renderer")]
pub struct Cli {
    /// Scenario to run
    #[arg(short, long, default_value = "plummer")]
    pub scenario: String,

    /// Number of particles (overrides scenario default)
    #[arg(short = 'n', long)]
    pub particles: Option<usize>,

    /// Timestep size (overrides scenario suggestion)
    #[arg(long)]
    pub dt: Option<f64>,

    /// Softening length (overrides scenario suggestion)
    #[arg(long)]
    pub softening: Option<f64>,

    /// RNG seed for scenario generation
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Eccentricity for two-body scenario (must be in [0, 1))
    #[arg(long, default_value_t = 0.5)]
    pub eccentricity: f64,

    /// Initial speed multiplier (sim time units per wall second)
    #[arg(long, default_value_t = 1.0)]
    pub speed: f64,

    /// Take a screenshot after settling and exit (path to PNG)
    #[arg(long)]
    pub screenshot: Option<String>,

    /// Force calculation algorithm
    #[arg(long, default_value = "brute-force", value_parser = ["brute-force", "barnes-hut"])]
    pub algorithm: String,

    /// Barnes-Hut opening angle (lower = more accurate, higher = faster)
    #[arg(long, default_value_t = 0.5)]
    pub theta: f64,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    // Validate
    if let Some(n) = cli.particles
        && n == 0
    {
        eprintln!("Error: --particles must be > 0");
        std::process::exit(1);
    }
    if cli.eccentricity < 0.0 || cli.eccentricity >= 1.0 {
        eprintln!(
            "Error: --eccentricity must be in [0, 1), got {}",
            cli.eccentricity
        );
        std::process::exit(1);
    }
    if let Some(dt) = cli.dt
        && dt <= 0.0
    {
        eprintln!("Error: --dt must be positive, got {dt}");
        std::process::exit(1);
    }
    if let Some(s) = cli.softening
        && s < 0.0
    {
        eprintln!("Error: --softening must be non-negative, got {s}");
        std::process::exit(1);
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = app::App::new(cli);
    event_loop.run_app(&mut app).expect("Event loop error");
}
