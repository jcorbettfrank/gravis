/// Map particle type to RGBA color for rendering.
///
/// Color scheme:
/// - Default (0): blue-white (backward compatible with original color)
/// - DiskStar (1): blue-white (cool stellar population)
/// - BulgeStar (2): yellow-orange (old stellar population)
/// - DarkMatter (3): dim translucent purple
pub fn particle_type_to_color(particle_type: u8) -> [f32; 4] {
    match particle_type {
        0 => [0.7, 0.85, 1.0, 1.0],        // Default: blue-white
        1 => [0.6, 0.75, 1.0, 1.0],         // DiskStar: blue-white
        2 => [1.0, 0.8, 0.4, 1.0],          // BulgeStar: yellow-orange
        3 => [0.3, 0.2, 0.5, 0.15],         // DarkMatter: dim translucent
        _ => [0.7, 0.85, 1.0, 1.0],         // Fallback: blue-white
    }
}
