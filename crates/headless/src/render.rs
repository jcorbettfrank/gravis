//! CPU density renderer: projects particles onto a plane, splats them into an
//! RGB grid colored by particle type, and tone-maps with an asinh stretch.
//!
//! Needs no GPU or window, so cloud sessions and CI can look at frames. Dark
//! matter is hidden by default, as a telescope would see it. Brightness is per
//! particle (by type), not per unit mass, so it stays meaningful when star and
//! halo particles have different masses.

use sim_core::particle::{ParticleType, Particles};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Projection plane. The first two letters are the image's horizontal and
/// vertical axes; the view looks down the remaining axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Xy,
    Xz,
    Yz,
}

impl View {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "xy" => Ok(View::Xy),
            "xz" => Ok(View::Xz),
            "yz" => Ok(View::Yz),
            other => Err(format!("unknown view '{other}', expected xy, xz, or yz")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            View::Xy => "xy",
            View::Xz => "xz",
            View::Yz => "yz",
        }
    }

    fn project(self, p: &Particles, i: usize) -> (f64, f64) {
        match self {
            View::Xy => (p.x[i], p.y[i]),
            View::Xz => (p.x[i], p.z[i]),
            View::Yz => (p.y[i], p.z[i]),
        }
    }
}

pub struct RenderOptions {
    /// Image width and height in pixels.
    pub size: usize,
    /// Half-width of the square field of view, centered on the origin (the
    /// system's center of mass).
    pub radius: f64,
    pub show_dark_matter: bool,
}

/// An 8-bit RGB image.
pub struct Image {
    pub size: usize,
    pub rgb: Vec<u8>,
}

impl Image {
    #[cfg(test)]
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let k = 3 * (y * self.size + x);
        [self.rgb[k], self.rgb[k + 1], self.rgb[k + 2]]
    }

    pub fn write_png(&self, path: &Path) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut encoder =
            png::Encoder::new(BufWriter::new(file), self.size as u32, self.size as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer
            .write_image_data(&self.rgb)
            .map_err(|e| e.to_string())
    }
}

/// Linear color and weight each particle type contributes. Mirrors the hues in
/// `render-core/src/color.rs`; dark matter is faint when shown at all.
fn type_color(particle_type: u8, internal_energy: f64) -> [f32; 3] {
    const DISK: [f32; 3] = [0.6, 0.75, 1.0];
    const BULGE: [f32; 3] = [1.0, 0.8, 0.4];
    const DEFAULT: [f32; 3] = [0.7, 0.85, 1.0];
    const DARK: [f32; 3] = [0.3 * 0.15, 0.2 * 0.15, 0.5 * 0.15];
    match particle_type {
        t if t == ParticleType::DiskStar as u8 => DISK,
        t if t == ParticleType::BulgeStar as u8 => BULGE,
        t if t == ParticleType::DarkMatter as u8 => DARK,
        t if t == ParticleType::Gas as u8 => gas_color(internal_energy),
        _ => DEFAULT,
    }
}

/// Temperature colormap for gas, as in `render-core::color::gas_temperature_color`:
/// log10(u) from -2 to 2 runs blue → yellow → white.
fn gas_color(u: f64) -> [f32; 3] {
    let t = ((u.max(1e-6).log10() as f32 + 2.0) / 4.0).clamp(0.0, 1.0);
    [
        t.powf(0.5),
        t.powf(1.2),
        (1.0 - t).powf(0.7) * 0.9 + t * 0.5,
    ]
}

/// Accumulate linear RGB light from every visible particle except `markers`
/// (point masses, drawn separately), with bilinear (cloud-in-cell) weights,
/// then blur by a small Gaussian to suppress shot noise.
pub fn splat(
    particles: &Particles,
    markers: &[usize],
    view: View,
    opts: &RenderOptions,
) -> Vec<f32> {
    let n = opts.size;
    let mut light = vec![0.0f32; 3 * n * n];
    let scale = n as f64 / (2.0 * opts.radius);
    let dm = ParticleType::DarkMatter as u8;

    for i in 0..particles.count {
        if (!opts.show_dark_matter && particles.particle_type[i] == dm) || markers.contains(&i) {
            continue;
        }
        let (u, v) = view.project(particles, i);
        // Pixel centers sit at integer + 0.5; image y runs downward.
        let px = (u + opts.radius) * scale - 0.5;
        let py = (opts.radius - v) * scale - 0.5;
        if !(px > -1.0 && py > -1.0 && px < n as f64 && py < n as f64) {
            continue;
        }
        let (x0, y0) = (px.floor(), py.floor());
        let (fx, fy) = ((px - x0) as f32, (py - y0) as f32);
        let c = type_color(particles.particle_type[i], particles.internal_energy[i]);
        for (dx, dy, w) in [
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ] {
            let (x, y) = (x0 as i64 + dx, y0 as i64 + dy);
            if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
                continue;
            }
            let k = 3 * (y as usize * n + x as usize);
            for ch in 0..3 {
                light[k + ch] += w * c[ch];
            }
        }
    }

    // One pixel of blur at 800 px; scale with resolution so a frame looks the
    // same at any size.
    let sigma = (n as f32 / 800.0).max(0.7);
    gaussian_blur(&mut light, n, sigma);
    light
}

/// Separable Gaussian blur of an RGB float image in place.
fn gaussian_blur(img: &mut [f32], n: usize, sigma: f32) {
    let radius = (3.0 * sigma).ceil() as i64;
    let mut kernel: Vec<f32> = (-radius..=radius)
        .map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= sum);

    let mut tmp = vec![0.0f32; img.len()];
    for pass in 0..2 {
        let (src, dst): (&[f32], &mut [f32]) = if pass == 0 {
            (img, &mut tmp)
        } else {
            (&tmp, img)
        };
        dst.iter_mut().for_each(|v| *v = 0.0);
        for y in 0..n {
            for x in 0..n {
                let mut acc = [0.0f32; 3];
                for (j, w) in kernel.iter().enumerate() {
                    let d = j as i64 - radius;
                    let (sx, sy) = if pass == 0 {
                        (x as i64 + d, y as i64)
                    } else {
                        (x as i64, y as i64 + d)
                    };
                    if sx < 0 || sy < 0 || sx >= n as i64 || sy >= n as i64 {
                        continue;
                    }
                    let k = 3 * (sy as usize * n + sx as usize);
                    for ch in 0..3 {
                        acc[ch] += w * src[k + ch];
                    }
                }
                let k = 3 * (y * n + x);
                dst[k..k + 3].copy_from_slice(&acc);
            }
        }
    }
}

/// The brightness that maps to full white: the 99.5th percentile of lit
/// pixels' peak channel. Robust to a few saturated cores.
pub fn auto_exposure(light: &[f32]) -> f32 {
    let mut lit: Vec<f32> = light
        .chunks_exact(3)
        .map(peak)
        .filter(|&l| l > 0.0)
        .collect();
    if lit.is_empty() {
        return 1.0;
    }
    let k = ((lit.len() - 1) as f64 * 0.995) as usize;
    let (_, v, _) = lit.select_nth_unstable_by(k, f32::total_cmp);
    v.max(1e-6)
}

fn peak(c: &[f32]) -> f32 {
    c[0].max(c[1]).max(c[2])
}

/// Tone-map linear light with an asinh stretch (Lupton et al. 2004), which is
/// linear for faint pixels and logarithmic for bright ones, so a lone tidal-tail
/// star and a galaxy core both show. Hue is preserved by scaling all channels
/// by the same factor.
pub fn tone_map(light: &[f32], n: usize, exposure: f32) -> Image {
    // Softening: about 100 × fainter than full white still separates from black
    let beta = exposure / 100.0;
    let norm = (exposure / beta).asinh();
    let mut rgb = vec![0u8; 3 * n * n];
    let (pixels, _) = rgb.as_chunks_mut::<3>();
    let (texels, _) = light.as_chunks::<3>();
    for (out, c) in pixels.iter_mut().zip(texels) {
        let l = peak(c);
        if l <= 0.0 {
            continue;
        }
        let s = ((l / beta).asinh() / norm).min(1.0) / l;
        for ch in 0..3 {
            out[ch] = (255.0 * (c[ch] * s).min(1.0)).round() as u8;
        }
    }
    Image { size: n, rgb }
}

/// Draw each marker particle (a point mass, such as a black hole) as a small
/// white dot with a soft edge, on top of the stars.
pub fn draw_markers(
    img: &mut Image,
    particles: &Particles,
    markers: &[usize],
    view: View,
    radius: f64,
) {
    let n = img.size;
    let scale = n as f64 / (2.0 * radius);
    let r = (n as f64 / 300.0).max(2.0);
    for &i in markers {
        let (u, v) = view.project(particles, i);
        let (cx, cy) = ((u + radius) * scale, (radius - v) * scale);
        let reach = r.ceil() as i64 + 1;
        for y in (cy as i64 - reach)..=(cy as i64 + reach) {
            for x in (cx as i64 - reach)..=(cx as i64 + reach) {
                if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
                    continue;
                }
                let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
                let a = (r + 0.5 - d).clamp(0.0, 1.0);
                if a == 0.0 {
                    continue;
                }
                let k = 3 * (y as usize * n + x as usize);
                for ch in 0..3 {
                    let v = img.rgb[k + ch] as f64;
                    img.rgb[k + ch] = (v + a * (255.0 - v)).round() as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(size: usize, show_dark_matter: bool) -> RenderOptions {
        RenderOptions {
            size,
            radius: 1.0,
            show_dark_matter,
        }
    }

    fn one_particle(x: f64, y: f64, particle_type: ParticleType) -> Particles {
        let mut p = Particles::new(1);
        p.add_typed(x, y, 0.0, 0.0, 0.0, 0.0, 1.0, particle_type as u8);
        p
    }

    #[test]
    fn particle_lands_at_projected_pixel() {
        // (0.5, 0.5) in a [-1, 1]² field at 64 px is pixel (48, 16): right and up
        let p = one_particle(0.5, 0.5, ParticleType::DiskStar);
        let o = opts(64, false);
        let light = splat(&p, &[], View::Xy, &o);
        let img = tone_map(&light, o.size, auto_exposure(&light));
        let brightest = (0..64 * 64)
            .max_by_key(|&k| peak_u8(img.pixel(k % 64, k / 64)))
            .unwrap();
        assert_eq!((brightest % 64, brightest / 64), (48, 16));
        assert_eq!(img.pixel(5, 60), [0, 0, 0]);
    }

    #[test]
    fn splat_conserves_light() {
        let p = one_particle(0.1, -0.2, ParticleType::Default);
        let light = splat(&p, &[], View::Xy, &opts(64, false));
        let total: f32 = light.chunks_exact(3).map(|c| c[2]).sum();
        assert!((total - 1.0).abs() < 1e-4, "blue channel sum {total}");
    }

    #[test]
    fn dark_matter_hidden_unless_asked() {
        let p = one_particle(0.0, 0.0, ParticleType::DarkMatter);
        assert!(
            splat(&p, &[], View::Xy, &opts(32, false))
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(
            splat(&p, &[], View::Xy, &opts(32, true))
                .iter()
                .any(|&v| v > 0.0)
        );
    }

    #[test]
    fn markers_are_drawn_white_not_splatted() {
        let p = one_particle(0.0, 0.0, ParticleType::Default);
        let o = opts(64, false);
        let light = splat(&p, &[0], View::Xy, &o);
        assert!(light.iter().all(|&v| v == 0.0));
        let mut img = tone_map(&light, o.size, 1.0);
        draw_markers(&mut img, &p, &[0], View::Xy, o.radius);
        assert_eq!(img.pixel(32, 32), [255, 255, 255]);
    }

    #[test]
    fn views_project_the_right_axes() {
        let mut p = Particles::new(1);
        p.add(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(View::Xz.project(&p, 0), (0.1, 0.3));
        assert_eq!(View::Yz.project(&p, 0), (0.2, 0.3));
        assert!(View::parse("zz").is_err());
    }

    fn peak_u8(c: [u8; 3]) -> u8 {
        c[0].max(c[1]).max(c[2])
    }
}
