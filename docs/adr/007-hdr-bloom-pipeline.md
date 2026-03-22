# ADR 007: HDR Rendering and Bloom Pipeline

**Date**: 2026-03-22
**Status**: Accepted

## Decision

Render particles to an Rgba16Float intermediate texture, apply a 3-pass separable Gaussian bloom at half resolution, composite bloom back onto the HDR scene, and tone map to the LDR surface using the ACES filmic curve.

## Context

M5 galaxy scenarios produce dense particle concentrations where thousands of additively blended particles overlap. The additive blending from Chapter 5 (SrcAlpha + One) accumulates color values that far exceed 1.0 in the galaxy core -- values of 10-100 are common with N >= 10,000. Rendering directly to the 8-bit surface clamps everything above 1.0 to solid white, destroying all structural detail in the densest regions.

Three design decisions address this:

### 1. HDR intermediate (Rgba16Float)

The particle render pass writes to an Rgba16Float texture instead of the swap chain surface. This 16-bit half-precision float format stores values up to ~65,504 with ~3 significant digits, which is sufficient for accumulating additive blends of any particle count we target. The alternative -- Rgba32Float -- would double memory bandwidth for negligible visual benefit, since the bloom and tone map stages introduce far more than 3 digits of error.

The HDR texture is allocated at full resolution and recreated on window resize. Memory cost is 8 bytes/pixel (vs. 4 bytes/pixel for Bgra8Unorm), doubling the render target memory. At 1920x1080, this is ~16 MB -- negligible on any GPU from the last decade.

### 2. Separable Gaussian bloom at half resolution

Bloom simulates the optical scattering of bright light. The pipeline has three passes:

**Threshold pass**: Extracts pixels with luminance above 0.8 (configurable). Uses standard perceptual luminance weights (0.2126, 0.7152, 0.0722). Output is at half resolution, providing an implicit 2x downscale that widens the effective blur radius.

**Horizontal blur pass**: 9-tap Gaussian (sigma ~ 2.0) along the x-axis at half resolution. Reads from the threshold output, writes to a blur intermediate texture.

**Vertical blur pass**: Same 9-tap kernel along the y-axis. Reads from the blur intermediate, writes to the final bloom texture.

The separable decomposition is key: a 2D Gaussian kernel of width k requires k^2 texture samples per pixel. The Gaussian function is separable (G(x,y) = G(x) * G(y)), so we can compute the same result with two 1D passes of k samples each -- O(k) instead of O(k^2). For our k=9 kernel, that's 9+9=18 samples instead of 81.

Operating at half resolution further reduces cost by 4x (quarter the pixels) and doubles the effective blur radius in screen space. The bilinear texture sampler handles the resolution mismatch during the composite step.

**Composite pass**: Adds bloom back to the original HDR scene: output = scene + intensity * bloom (intensity = 0.3 by default). The output remains in Rgba16Float for the subsequent tone mapping step.

### 3. ACES filmic tone mapping

The composited HDR image is mapped to the [0, 1] LDR range using the ACES (Academy Color Encoding System) filmic curve:

    f(x) = (x * (2.51x + 0.03)) / (x * (2.43x + 0.59) + 0.14)

ACES was chosen over alternatives:

- **Reinhard (x / (1+x))**: Too gentle -- galaxy cores still appear washed out because Reinhard compresses the high end too aggressively while leaving midtones too dark.
- **Linear clamp**: Loses all HDR information. The entire motivation for this pipeline is to avoid clamping.
- **Filmic Uncharted 2**: Similar S-curve to ACES but with more parameters to tune. ACES achieves comparable results with a single fixed curve.

ACES provides a natural-looking shoulder (bright values compress gracefully), a slight toe lift (dark regions aren't crushed to pure black), and a near-linear midtone region. The result resembles a properly exposed photograph.

The tone map is a single fullscreen pass that reads the Rgba16Float composite and writes to the swap chain surface (typically Bgra8Unorm).

## Consequences

- **Performance**: The bloom pipeline adds 4 fullscreen passes (threshold + blur H + blur V + composite) at half resolution, plus 1 fullscreen tone map pass at full resolution. On the M5 Pro GPU, total bloom + tonemap overhead is < 0.5 ms at 1080p, well within budget for 60fps rendering. The simulation (CPU) remains the bottleneck.
- **Memory**: Three additional Rgba16Float textures at half resolution, plus the full-resolution HDR render target. Total additional GPU memory at 1920x1080: ~16 MB (HDR target) + ~12 MB (3 half-res textures) = ~28 MB.
- **Web compatibility**: WebGPU supports Rgba16Float as a render target on all shipping implementations (Chrome, Edge, Safari). The bloom pipeline runs identically on native and WASM with no code changes. WASM performance is GPU-bound during bloom (same GPU, same shaders), so there is no WASM-specific overhead.
- **Texture recreation on resize**: All bloom textures and their bind groups must be recreated when the window resizes. The BloomPipeline::resize() method handles this, but the bind group recreation pattern adds complexity. Caching bind group layouts (stored as fields on BloomPipeline) avoids recreating those.
- **Fixed bloom parameters**: The threshold (0.8) and intensity (0.3) are hardcoded. Future work could expose these as egui sliders for user tuning, or make them scenario-dependent (e.g., lower threshold for sparse scenarios, higher for dense galaxy cores).
