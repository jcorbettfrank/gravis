/// Map particle type to RGBA color for rendering.
///
/// Color scheme:
/// - Default (0): blue-white (backward compatible with original color)
/// - DiskStar (1): blue-white (cool stellar population)
/// - BulgeStar (2): yellow-orange (old stellar population)
/// - DarkMatter (3): dim translucent purple
/// - Gas (4): use `gas_temperature_color` instead for dynamic coloring
pub fn particle_type_to_color(particle_type: u8) -> [f32; 4] {
    match particle_type {
        0 => [0.7, 0.85, 1.0, 1.0],        // Default: blue-white
        1 => [0.6, 0.75, 1.0, 1.0],         // DiskStar: blue-white
        2 => [1.0, 0.8, 0.4, 1.0],          // BulgeStar: yellow-orange
        3 => [0.3, 0.2, 0.5, 0.15],         // DarkMatter: dim translucent
        4 => [0.3, 0.6, 1.0, 0.8],          // Gas: default blue (overridden by temperature)
        _ => [0.7, 0.85, 1.0, 1.0],         // Fallback: blue-white
    }
}

/// Map gas internal energy to an RGBA color using a temperature colormap.
///
/// Uses log(u) as a temperature proxy:
/// - Cold (low u): blue
/// - Warm (moderate u): yellow-orange
/// - Hot (high u): white
///
/// The range is calibrated to u ∈ [0.01, 100] covering typical SPH problems.
pub fn gas_temperature_color(internal_energy: f32) -> [f32; 4] {
    let log_u = internal_energy.max(1e-6).log10();
    // Map log10(u) from [-2, 2] to [0, 1]
    let t = ((log_u + 2.0) / 4.0).clamp(0.0, 1.0);

    // Cool (blue) → warm (yellow) → hot (white)
    let r = t.powf(0.5);
    let g = t.powf(1.2);
    let b = (1.0 - t).powf(0.7) * 0.9 + t * 0.5;
    let a = 0.7 + 0.3 * t; // Hotter particles more opaque

    [r, g, b, a]
}
