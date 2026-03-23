//! Wendland C2 smoothing kernel for 3D SPH.
//!
//! The Wendland C2 kernel avoids the pairing instability present in the cubic
//! spline (whose flat central region causes particles to clump in pairs).
//! Wendland C2 has a non-zero second derivative at the origin, which stabilizes
//! the particle distribution. It has compact support at q = r/h = 2.
//!
//! Reference: Wendland (1995), Dehnen & Aly (2012).

use std::f64::consts::PI;

/// Normalization constant for Wendland C2 in 3D with compact support at q = r/h = 2.
///
/// The standard Wendland C2 has σ = 21/(2π) with support at q = 1. Rescaling to
/// support at q = 2 (substituting q → q/2 in the kernel) introduces a factor of
/// 1/8 from the volume element, giving σ = 21/(16π).
const NORM_3D: f64 = 21.0 / (16.0 * PI);

/// Wendland C2 kernel value W(r, h) in 3D.
///
/// W(q) = σ/h³ · (1 - q/2)⁴ · (1 + 2q)   for 0 ≤ q ≤ 2
///      = 0                                  for q > 2
///
/// where q = r/h and σ = 21/(2π).
#[inline]
pub fn w(r: f64, h: f64) -> f64 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let h3 = h * h * h;
    let s = 1.0 - 0.5 * q; // (1 - q/2)
    let s4 = s * s * s * s;
    NORM_3D / h3 * s4 * (1.0 + 2.0 * q)
}

/// Gradient of the Wendland C2 kernel: ∇W(r_vec, h).
///
/// Returns a 3-vector pointing from j toward i (same direction as r_vec = r_i - r_j).
///
/// ∇W = (dW/dr) · r̂ = (1/h)(dW/dq) · r_vec/|r_vec|
///
/// dW/dq = σ/h³ · (-5q)(1 - q/2)³   for 0 ≤ q ≤ 2
///       = 0                           for q > 2
///
/// Returns [0, 0, 0] when r = 0 (self-interaction has zero gradient by symmetry).
#[inline]
pub fn grad_w(dx: f64, dy: f64, dz: f64, r: f64, h: f64) -> [f64; 3] {
    if r < 1e-30 {
        return [0.0; 3];
    }
    let q = r / h;
    if q >= 2.0 {
        return [0.0; 3];
    }
    let h3 = h * h * h;
    let s = 1.0 - 0.5 * q; // (1 - q/2)
    let s3 = s * s * s;
    // dW/dq = σ/h³ · (-5q) · (1-q/2)³
    let dwdq = NORM_3D / h3 * (-5.0 * q) * s3;
    // ∇W = (1/h)(dW/dq) · r_vec / r
    let factor = dwdq / (h * r);
    [factor * dx, factor * dy, factor * dz]
}

/// Partial derivative of W with respect to h: ∂W/∂h.
///
/// Used for grad-h correction terms (Springel & Hernquist 2002).
///
/// ∂W/∂h = -(3/h)W(q) - (q/h)(dW/dq)
///
/// This follows from differentiating W(r/h, h) = σ/h³ · f(q) with respect to h,
/// where q = r/h.
#[inline]
pub fn dw_dh(r: f64, h: f64) -> f64 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let h3 = h * h * h;
    let s = 1.0 - 0.5 * q;
    let s3 = s * s * s;
    let s4 = s3 * s;
    let w_val = NORM_3D / h3 * s4 * (1.0 + 2.0 * q);
    let dwdq = NORM_3D / h3 * (-5.0 * q) * s3;
    -3.0 / h * w_val - q / h * dwdq
}

/// Magnitude of the kernel gradient |∇W|.
///
/// This is cheaper than computing the full gradient vector when only the
/// magnitude is needed (e.g., for artificial conductivity).
#[inline]
pub fn abs_grad_w(r: f64, h: f64) -> f64 {
    if r < 1e-30 {
        return 0.0;
    }
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let h3 = h * h * h;
    let s = 1.0 - 0.5 * q;
    let s3 = s * s * s;
    let dwdq = NORM_3D / h3 * (-5.0 * q) * s3;
    // |∇W| = |dW/dq| / h  (since dW/dq is negative, negate it)
    -dwdq / h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_support() {
        let h = 1.0;
        assert_eq!(w(2.0, h), 0.0);
        assert_eq!(w(2.5, h), 0.0);
        assert!(w(1.99, h) > 0.0);
    }

    #[test]
    fn kernel_positive() {
        let h = 0.5;
        for i in 0..20 {
            let r = i as f64 * 0.05;
            assert!(w(r, h) >= 0.0, "W({}, {}) = {} < 0", r, h, w(r, h));
        }
    }

    #[test]
    fn kernel_monotone_decreasing() {
        let h = 1.0;
        let mut prev = w(0.0, h);
        for i in 1..20 {
            let r = i as f64 * 0.1;
            let cur = w(r, h);
            assert!(cur <= prev, "W not monotone at r={}", r);
            prev = cur;
        }
    }

    #[test]
    fn normalization_3d() {
        // Numerically integrate W over 3D space using spherical shells.
        // ∫ W(r,h) dV = ∫₀^{2h} W(r,h) · 4πr² dr ≈ 1
        let h = 1.0;
        let n = 10_000;
        let dr = 2.0 * h / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let r = (i as f64 + 0.5) * dr;
            integral += w(r, h) * 4.0 * PI * r * r * dr;
        }
        assert!(
            (integral - 1.0).abs() < 0.001,
            "Normalization integral = {} (expected 1.0)",
            integral
        );
    }

    #[test]
    fn gradient_antisymmetry() {
        let h = 0.8;
        let (dx, dy, dz): (f64, f64, f64) = (0.3, -0.2, 0.5);
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let g_fwd = grad_w(dx, dy, dz, r, h);
        let g_rev = grad_w(-dx, -dy, -dz, r, h);
        for d in 0..3 {
            assert!(
                (g_fwd[d] + g_rev[d]).abs() < 1e-14,
                "Gradient not antisymmetric in component {}",
                d
            );
        }
    }

    #[test]
    fn gradient_zero_at_origin() {
        let g = grad_w(0.0, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(g, [0.0; 3]);
    }

    #[test]
    fn gradient_zero_outside_support() {
        let g = grad_w(3.0, 0.0, 0.0, 3.0, 1.0);
        assert_eq!(g, [0.0; 3]);
    }

    #[test]
    fn dw_dh_finite_difference() {
        // Verify ∂W/∂h against numerical finite difference.
        let r = 0.7;
        let h = 1.0;
        let eps = 1e-6;
        let numerical = (w(r, h + eps) - w(r, h - eps)) / (2.0 * eps);
        let analytical = dw_dh(r, h);
        assert!(
            (numerical - analytical).abs() < 1e-5,
            "dW/dh: numerical={}, analytical={}",
            numerical,
            analytical
        );
    }

    #[test]
    fn dw_dh_zero_outside_support() {
        assert_eq!(dw_dh(3.0, 1.0), 0.0);
    }

    #[test]
    fn abs_grad_w_matches_gradient_magnitude() {
        let h = 0.6;
        let (dx, dy, dz): (f64, f64, f64) = (0.2, 0.3, -0.1);
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let g = grad_w(dx, dy, dz, r, h);
        let mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        let abs_g = abs_grad_w(r, h);
        assert!(
            (mag - abs_g).abs() < 1e-14,
            "|grad_w|={}, abs_grad_w={}",
            mag,
            abs_g
        );
    }
}
