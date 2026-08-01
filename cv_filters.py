"""
Classic image-processing styles using OpenCV/PIL — zero API cost, zero rate
limits, zero network calls. Every style here runs instantly on your own
server using nothing but pixel math. No AI models, no external requests.
"""

import cv2
import numpy as np


def _decode(image_bytes: bytes, max_dim: int = 1024) -> np.ndarray:
    """Shared decode + downscale step used by every filter below."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def _encode(img: np.ndarray, quality: int = 92) -> bytes:
    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Failed to encode processed image")
    return encoded.tobytes()


def aesthetic_blur(image_bytes: bytes, intensity: int = 50, angle: float = 90.0, grayscale: bool = False) -> bytes:
    """
    Directional motion-blur "dreamy drift" effect — like a subject caught
    mid-movement in a field, streaked softly in one direction, rather than
    an even all-over Gaussian blur. Intensity 1-100 controls streak length
    and how much of the original sharp image blends back in.
    If grayscale=True, the image is converted to black & white first, then
    the same directional blur/drift is applied on top.
    """
    intensity = max(1, min(100, int(intensity)))
    img = _decode(image_bytes)

    if grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]

    # Streak length scales with intensity — longer streak = stronger drift.
    # Capped relative to image size so it never looks like total mush.
    streak_len = max(3, int((intensity / 100) * (min(h, w) * 0.06)))
    if streak_len % 2 == 0:
        streak_len += 1

    # Build a directional motion-blur kernel (a line at the given angle)
    kernel = np.zeros((streak_len, streak_len), dtype=np.float32)
    kernel[streak_len // 2, :] = 1.0
    center = (streak_len / 2 - 0.5, streak_len / 2 - 0.5)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_matrix, (streak_len, streak_len))
    kernel = kernel / max(kernel.sum(), 1e-6)

    streaked = cv2.filter2D(img, -1, kernel)

    # Slight brightness/soft lift so the streaked areas feel airy, not muddy
    streaked = cv2.convertScaleAbs(streaked, alpha=1.05, beta=6)

    # Blend: at low intensity, mostly sharp with a hint of drift; at high
    # intensity, the streak dominates but a touch of the original shows
    # through so edges don't fully dissolve.
    alpha = intensity / 100.0
    result = cv2.addWeighted(img, 1 - alpha * 0.85, streaked, alpha * 0.85 + 0.15, 0)

    return _encode(result)


def aesthetic_glow(image_bytes: bytes, intensity: int = 100) -> bytes:
    """
    Full moody color-grade pipeline: brightness/contrast/saturation/sharpen
    adjustments, exposure/highlight/shadow/temperature tuning, a soft muted
    filter pass, film grain, and a light vignette. `intensity` (1-100) scales
    how strongly all of the above are applied, with 100 matching the full
    reference settings.
    """
    intensity = max(1, min(100, int(intensity)))
    strength = intensity / 100.0

    img = _decode(image_bytes).astype(np.float32)

    # --- Exposure (-5) + Brightness (-5 to -10, use -8 as the reference midpoint) ---
    brightness_shift = -8 * strength
    exposure_shift = -5 * strength
    img = img + brightness_shift + exposure_shift

    # --- Contrast (+10 to +15, use +12) ---
    contrast_factor = 1 + (0.12 * strength)
    img = (img - 127.5) * contrast_factor + 127.5

    # --- Highlights (-15): pull down bright areas ---
    lum = img.mean(axis=2, keepdims=True)
    highlight_mask = np.clip((lum - 180) / 75, 0, 1)  # ramps in for bright pixels
    img = img - (15 * strength) * highlight_mask

    # --- Shadows (+10): lift dark areas ---
    shadow_mask = np.clip((80 - lum) / 80, 0, 1)  # ramps in for dark pixels
    img = img + (10 * strength) * shadow_mask

    img = np.clip(img, 0, 255)

    # --- Saturation (-10 to -20, use -15) ---
    img_u8 = img.astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 - 0.15 * strength), 0, 255)
    img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img_u8.astype(np.float32)

    # --- Temperature: slight warm shift (+5 warmth reference) ---
    b, g, r = cv2.split(img)
    r = r + (5 * strength)
    b = b - (5 * strength)
    img = cv2.merge([b, g, r])
    img = np.clip(img, 0, 255)

    # --- Muted filter pass (soft "Moody"-style desaturated fade), ~35% strength ---
    filter_amount = 0.35 * strength
    faded = img * (1 - filter_amount * 0.3) + 20 * filter_amount
    img = np.clip(faded, 0, 255)

    img_u8 = img.astype(np.uint8)

    # --- Sharpen (+10): unsharp mask ---
    blur = cv2.GaussianBlur(img_u8, (0, 0), 3)
    sharpened = cv2.addWeighted(img_u8, 1 + 0.3 * strength, blur, -0.3 * strength, 0)
    img_u8 = sharpened

    # --- Film grain (15% intensity) ---
    grain_strength = 15 * strength
    noise = np.random.normal(0, grain_strength, img_u8.shape).astype(np.float32)
    grainy = img_u8.astype(np.float32) + noise
    img_u8 = np.clip(grainy, 0, 255).astype(np.uint8)

    # --- Vignette (10% strength, light dark border) ---
    h, w = img_u8.shape[:2]
    x = cv2.getGaussianKernel(w, w * 0.9)
    y = cv2.getGaussianKernel(h, h * 0.9)
    mask = (y @ x.T)
    mask = mask / mask.max()
    vignette_strength = 0.10 * strength
    mask = (1 - vignette_strength) + vignette_strength * mask
    result = img_u8.astype(np.float32) * mask[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return _encode(result)


def _blend_overlay(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Photoshop-style Overlay blend, both inputs float32 0-255."""
    b = base / 255.0
    t = top / 255.0
    result = np.where(
        b < 0.5,
        2 * b * t,
        1 - 2 * (1 - b) * (1 - t),
    )
    return np.clip(result * 255.0, 0, 255)


def _blend_soft_light(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Photoshop-style Soft Light blend, both inputs float32 0-255."""
    b = base / 255.0
    t = top / 255.0
    result = np.where(
        t < 0.5,
        2 * b * t + b * b * (1 - 2 * t),
        2 * b * (1 - t) + np.sqrt(np.clip(b, 0, 1)) * (2 * t - 1),
    )
    return np.clip(result * 255.0, 0, 255)


def _blend_screen(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Photoshop-style Screen blend, both inputs float32 0-255."""
    b = base / 255.0
    t = top / 255.0
    result = 1 - (1 - b) * (1 - t)
    return np.clip(result * 255.0, 0, 255)


def retro_90s(image_bytes: bytes, intensity: int = 100) -> bytes:
    """
    90s film-photo style: warm color grade (temperature/tint/saturation/
    contrast/highlights/shadows/fade), layered film grain (overlay/soft-light
    blend), a warm orange/red light leak (screen blend), dust & scratches
    texture (screen blend), and a light vignette. `intensity` (1-100) scales
    how strongly the whole stack is applied, with 100 matching the reference
    settings exactly.
    """
    intensity = max(1, min(100, int(intensity)))
    strength = intensity / 100.0

    img = _decode(image_bytes).astype(np.float32)
    h, w = img.shape[:2]

    # ---------------- Color grading ----------------

    # Temperature +8 (warm) and Tint +4 magenta
    b, g, r = cv2.split(img)
    r = r + (8 * strength)
    b = b - (8 * strength)
    g = g - (4 * strength) * 0.5  # slight green pull for magenta tint
    r = r + (4 * strength) * 0.5
    img = cv2.merge([b, g, r])
    img = np.clip(img, 0, 255)

    # Contrast -12
    contrast_factor = 1 - (0.12 * strength)
    img = (img - 127.5) * contrast_factor + 127.5
    img = np.clip(img, 0, 255)

    # Highlights -18 / Shadows +12
    lum = img.mean(axis=2, keepdims=True)
    highlight_mask = np.clip((lum - 170) / 85, 0, 1)
    img = img - (18 * strength) * highlight_mask
    shadow_mask = np.clip((85 - lum) / 85, 0, 1)
    img = img + (12 * strength) * shadow_mask
    img = np.clip(img, 0, 255)

    # Saturation -8
    img_u8 = img.astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 - 0.08 * strength), 0, 255)
    img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img_u8.astype(np.float32)

    # Fade / Matte +18: lift blacks toward gray for that faded film look
    fade_amount = 18 * strength
    img = img * (1 - fade_amount / 255) + fade_amount
    img = np.clip(img, 0, 255)

    # ---------------- Layered effects ----------------

    rng = np.random.default_rng()

    # Film Grain — overlay/soft-light blend, opacity 25-45% (use ~35% base)
    grain_gray = rng.normal(128, 35, (h, w)).astype(np.float32)
    grain_layer = np.repeat(grain_gray[:, :, None], 3, axis=2)
    grain_overlay = _blend_soft_light(img, grain_layer)
    grain_opacity = 0.35 * strength
    img = img * (1 - grain_opacity) + grain_overlay * grain_opacity

    # Light Leak (orange/red) — screen blend, opacity 15-35% (use ~25% base)
    leak = np.zeros((h, w, 3), dtype=np.float32)
    leak_x = int(w * 0.85)
    leak_y = int(h * 0.15)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - leak_x) ** 2 + (yy - leak_y) ** 2)
    radius = max(w, h) * 0.55
    leak_mask = np.clip(1 - dist / radius, 0, 1) ** 1.5
    leak[:, :, 0] = 20 * leak_mask   # B
    leak[:, :, 1] = 90 * leak_mask   # G
    leak[:, :, 2] = 255 * leak_mask  # R (orange/red leak)
    leak_screen = _blend_screen(img, leak)
    leak_opacity = 0.25 * strength
    img = img * (1 - leak_opacity) + leak_screen * leak_opacity

    # Dust & Scratches — screen blend, opacity 10-25% (use ~18% base)
    dust = np.zeros((h, w), dtype=np.float32)
    n_specks = int((h * w) / 4000)
    speck_y = rng.integers(0, h, n_specks)
    speck_x = rng.integers(0, w, n_specks)
    dust[speck_y, speck_x] = rng.uniform(120, 255, n_specks)
    n_scratches = max(1, int(3 * strength))
    for _ in range(n_scratches):
        x0 = rng.integers(0, w)
        length = rng.integers(int(h * 0.2), int(h * 0.6))
        y0 = rng.integers(0, max(1, h - length))
        thickness = 1
        cv2.line(dust, (x0, y0), (x0 + rng.integers(-5, 5), y0 + length), 180, thickness)
    dust = cv2.GaussianBlur(dust, (3, 3), 0)
    dust_layer = np.repeat(dust[:, :, None], 3, axis=2)
    dust_screen = _blend_screen(img, dust_layer)
    dust_opacity = 0.18 * strength
    img = img * (1 - dust_opacity) + dust_screen * dust_opacity

    img = np.clip(img, 0, 255)

    # Grain (separate coarser grain pass, 30-40 -> use ~35)
    coarse_grain_strength = 35 * strength * 0.35
    coarse_noise = rng.normal(0, coarse_grain_strength, img.shape).astype(np.float32)
    img = img + coarse_noise
    img = np.clip(img, 0, 255)

    # Vignette -12 (light dark border)
    x_k = cv2.getGaussianKernel(w, w * 0.85)
    y_k = cv2.getGaussianKernel(h, h * 0.85)
    vmask = (y_k @ x_k.T)
    vmask = vmask / vmask.max()
    vignette_strength = 0.12 * strength
    vmask = (1 - vignette_strength) + vignette_strength * vmask
    img = img * vmask[:, :, None]

    result = np.clip(img, 0, 255).astype(np.uint8)

    # Film Frame — normal blend, 100% opacity, optional white border
    border = max(4, int(min(h, w) * 0.03))
    result = cv2.copyMakeBorder(
        result, border, border, border, border,
        cv2.BORDER_CONSTANT, value=(245, 245, 245)
    )

    return _encode(result)


def cartoonify(image_bytes: bytes) -> bytes:
    """Bilateral-filter + edge-mask cartoon effect."""
    img = _decode(image_bytes)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )

    color = img
    for _ in range(5):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    div = 24
    color = (color // div) * div + div // 2

    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return _encode(cartoon)


def pencil_sketch(image_bytes: bytes, color: bool = False) -> bytes:
    """Pencil sketch effect. Set color=True for a lightly colored sketch."""
    img = _decode(image_bytes)
    gray_sketch, color_sketch = cv2.pencilSketch(
        img, sigma_s=60, sigma_r=0.07, shade_factor=0.05
    )
    result = color_sketch if color else gray_sketch
    return _encode(result)


def oil_painting(image_bytes: bytes) -> bytes:
    """Stylization filter — soft painterly look."""
    img = _decode(image_bytes)
    result = cv2.stylization(img, sigma_s=60, sigma_r=0.45)
    return _encode(result)


def sepia(image_bytes: bytes) -> bytes:
    """Classic warm vintage sepia tone."""
    img = _decode(image_bytes).astype(np.float32)
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ])
    # cv2 is BGR, kernel above is written for RGB order, so flip channels first
    rgb = img[:, :, ::-1]
    sepia_rgb = rgb @ kernel.T
    sepia_rgb = np.clip(sepia_rgb, 0, 255)
    result = sepia_rgb[:, :, ::-1].astype(np.uint8)
    return _encode(result)


def black_and_white(image_bytes: bytes) -> bytes:
    """High-contrast classic B&W with a slight contrast boost."""
    img = _decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=-10)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _encode(result)


def negative(image_bytes: bytes) -> bytes:
    """Inverted colors, film-negative look."""
    img = _decode(image_bytes)
    result = cv2.bitwise_not(img)
    return _encode(result)


def emboss(image_bytes: bytes) -> bytes:
    """Embossed/engraved metallic relief effect."""
    img = _decode(image_bytes)
    kernel = np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2],
    ])
    result = cv2.filter2D(img, -1, kernel) + 128
    result = np.clip(result, 0, 255).astype(np.uint8)
    return _encode(result)


def hdr_glow(image_bytes: bytes) -> bytes:
    """Detail-enhanced, punchy HDR-style look using OpenCV's built-in detailEnhance."""
    img = _decode(image_bytes)
    result = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return _encode(result)


def warm_vintage(image_bytes: bytes) -> bytes:
    """Faded warm-toned retro photo look (lifted shadows, warm color cast, vignette)."""
    img = _decode(image_bytes).astype(np.float32)

    # Warm color cast: boost red/green, pull down blue slightly
    b, g, r = cv2.split(img)
    r = np.clip(r * 1.12, 0, 255)
    g = np.clip(g * 1.04, 0, 255)
    b = np.clip(b * 0.88, 0, 255)
    warm = cv2.merge([b, g, r])

    # Lift shadows (faded film look)
    warm = warm * 0.9 + 25

    # Vignette
    h, w = warm.shape[:2]
    x = cv2.getGaussianKernel(w, w * 0.6)
    y = cv2.getGaussianKernel(h, h * 0.6)
    mask = (y @ x.T)
    mask = mask / mask.max()
    vignette = warm * mask[:, :, None]

    result = np.clip(vignette, 0, 255).astype(np.uint8)
    return _encode(result)


def cool_blue(image_bytes: bytes) -> bytes:
    """Cool blue/teal cinematic color grade."""
    img = _decode(image_bytes).astype(np.float32)
    b, g, r = cv2.split(img)
    b = np.clip(b * 1.15, 0, 255)
    g = np.clip(g * 1.02, 0, 255)
    r = np.clip(r * 0.9, 0, 255)
    result = cv2.merge([b, g, r]).astype(np.uint8)
    return _encode(result)


def pop_art(image_bytes: bytes) -> bytes:
    """Posterized, high-saturation pop-art poster look."""
    img = _decode(image_bytes)

    # Boost saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Posterize (reduce color levels for flat poster blocks)
    div = 48
    result = (boosted // div) * div + div // 2
    result = np.clip(result, 0, 255).astype(np.uint8)
    return _encode(result)


# =====================================================================
# "More styles" page — 20 additional, structurally distinct effects.
# These aren't just color-grade variations; each changes the underlying
# rendering technique (dithering, tiling, edge-detection, remapping,
# channel manipulation, etc).
# =====================================================================


def duotone(image_bytes: bytes) -> bytes:
    """Two-color luminance remap - shadows to deep indigo, highlights to hot coral."""
    img = _decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow = np.array([90, 20, 35])
    highlight = np.array([80, 150, 250])
    result = shadow[None, None, :] * (1 - gray[:, :, None]) + highlight[None, None, :] * gray[:, :, None]
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def halftone(image_bytes: bytes) -> bytes:
    """Newspaper-print halftone: image rendered as dot-size-by-brightness grid."""
    img = _decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = max(4, min(h, w) // 120)
    canvas = np.full((h, w), 255, dtype=np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            block = gray[y:y + cell, x:x + cell]
            if block.size == 0:
                continue
            mean_val = block.mean()
            radius = int((1 - mean_val / 255) * (cell / 2))
            if radius > 0:
                cy, cx = y + cell // 2, x + cell // 2
                cv2.circle(canvas, (cx, cy), radius, 0, -1)
    result = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    return _encode(result)


def glitch(image_bytes: bytes) -> bytes:
    """Digital glitch: RGB channel splitting/shifting plus torn scanline blocks."""
    img = _decode(image_bytes)
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)

    shift = max(2, w // 80)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    result = cv2.merge([b, g, r])

    rng = np.random.default_rng()
    n_tears = max(3, h // 60)
    for _ in range(n_tears):
        y0 = rng.integers(0, h)
        band_h = rng.integers(2, max(3, h // 40))
        y1 = min(h, y0 + band_h)
        tear_shift = rng.integers(-w // 15, w // 15)
        result[y0:y1] = np.roll(result[y0:y1], tear_shift, axis=1)

    return _encode(result)


def thermal(image_bytes: bytes) -> bytes:
    """False-color thermal/heatmap remap based on luminance."""
    img = _decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return _encode(result)


def cyberpunk_neon(image_bytes: bytes) -> bytes:
    """Teal/magenta cinematic split-tone with neon edge-glow overlay."""
    img = _decode(image_bytes).astype(np.float32)
    b, g, r = cv2.split(img)
    lum = (0.299 * r + 0.587 * g + 0.114 * b)
    shadow_mix = np.clip((150 - lum) / 150, 0, 1)
    highlight_mix = np.clip((lum - 100) / 155, 0, 1)
    r = r + 40 * highlight_mix - 10 * shadow_mix
    b = b + 45 * shadow_mix + 15 * highlight_mix
    g = g - 15 * shadow_mix
    graded = cv2.merge([b, g, r])
    graded = np.clip(graded, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(graded, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    glow = cv2.GaussianBlur(edges, (0, 0), 4)
    neon = np.zeros_like(graded, dtype=np.float32)
    neon[:, :, 0] = glow
    neon[:, :, 2] = glow * 0.7

    result = graded.astype(np.float32) + neon * 0.9
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def watercolor(image_bytes: bytes) -> bytes:
    """Soft diffused watercolor wash with bled, feathered edges."""
    img = _decode(image_bytes)
    smooth = cv2.edgePreservingFilter(img, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.5)
    smooth = cv2.medianBlur(smooth, 7)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    edges = cv2.GaussianBlur(edges, (0, 0), 2)
    edges_inv = 255 - edges
    edges_bgr = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    result = smooth.astype(np.float32) * (0.75 + 0.25 * edges_bgr)
    result = np.clip(result, 0, 255).astype(np.uint8)

    rng = np.random.default_rng()
    noise = rng.normal(0, 6, result.shape).astype(np.float32)
    result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return _encode(result)


def comic_book(image_bytes: bytes) -> bytes:
    """Bold black ink outlines over flat posterized color, like a comic panel."""
    img = _decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=7, C=5
    )
    edges = cv2.erode(edges, np.ones((2, 2), np.uint8))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    div = 40
    flat = (boosted // div) * div + div // 2
    flat = np.clip(flat, 0, 255).astype(np.uint8)

    result = cv2.bitwise_and(flat, flat, mask=edges)
    return _encode(result)


def xray(image_bytes: bytes) -> bytes:
    """Blue-tinted inverted-edge x-ray/skeletal scan look."""
    img = _decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    combined = cv2.addWeighted(inv, 0.7, edges, 0.6, 0)
    combined = cv2.equalizeHist(combined)
    result = np.zeros((*combined.shape, 3), dtype=np.float32)
    result[:, :, 0] = combined * 1.1
    result[:, :, 1] = combined * 0.6
    result[:, :, 2] = combined * 0.15
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def infrared(image_bytes: bytes) -> bytes:
    """False-color infrared simulation - greens push toward white/pink."""
    img = _decode(image_bytes).astype(np.float32)
    b, g, r = cv2.split(img)
    green_dominance = np.clip(g - np.maximum(r, b), 0, 255) / 255.0
    r_new = r + 120 * green_dominance
    g_new = g * (1 - 0.5 * green_dominance) + 60 * green_dominance
    b_new = b + 40 * green_dominance
    result = cv2.merge([b_new, g_new, r_new])
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def stained_glass(image_bytes: bytes) -> bytes:
    """Superpixel-style color blocks with dark leading lines, like stained glass."""
    img = _decode(image_bytes, max_dim=800)
    h, w = img.shape[:2]
    cell = max(10, min(h, w) // 40)

    small = cv2.resize(img, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_AREA)
    blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(blocky, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.resize(grad, (w, h), interpolation=cv2.INTER_NEAREST)
    leading = (grad > (grad.max() * 0.15)).astype(np.uint8) * 255
    leading = cv2.dilate(leading, np.ones((2, 2), np.uint8))
    leading_mask = (255 - leading).astype(np.float32) / 255.0

    result = boosted.astype(np.float32) * leading_mask[:, :, None]
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def mosaic_tile(image_bytes: bytes) -> bytes:
    """Chunky square-tile mosaic with grout lines between flat average-color cells."""
    img = _decode(image_bytes)
    h, w = img.shape[:2]
    cell = max(8, min(h, w) // 55)
    canvas = np.zeros_like(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            block = img[y:y + cell, x:x + cell]
            if block.size == 0:
                continue
            avg_color = block.reshape(-1, 3).mean(axis=0)
            canvas[y:y + cell, x:x + cell] = avg_color
    for y in range(0, h, cell):
        canvas[y:y + 1, :] = (30, 30, 30)
    for x in range(0, w, cell):
        canvas[:, x:x + 1] = (30, 30, 30)
    return _encode(canvas)


def double_exposure(image_bytes: bytes) -> bytes:
    """Self-blended offset double-exposure ghosting effect."""
    img = _decode(image_bytes).astype(np.float32)
    h, w = img.shape[:2]

    ghost = cv2.resize(img, (int(w * 1.15), int(h * 1.15)))
    gh, gw = ghost.shape[:2]
    off_y, off_x = (gh - h) // 2 + int(h * 0.08), (gw - w) // 2 - int(w * 0.06)
    off_y, off_x = max(0, off_y), max(0, off_x)
    ghost_crop = ghost[off_y:off_y + h, off_x:off_x + w]
    if ghost_crop.shape[:2] != (h, w):
        ghost_crop = cv2.resize(ghost_crop, (w, h))

    ghost_gray = cv2.cvtColor(ghost_crop.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ghost_tinted = np.stack([ghost_gray * 1.1, ghost_gray * 0.7, ghost_gray * 0.6], axis=-1)

    result = _blend_screen(img, ghost_tinted) * 0.6 + img * 0.4
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def pixel_art(image_bytes: bytes) -> bytes:
    """Heavy-downsample + palette-quantize + nearest-neighbor upscale for a retro pixel-art look."""
    img = _decode(image_bytes)
    h, w = img.shape[:2]
    small_w = 64
    small_h = max(1, int(h * (small_w / w)))
    tiny = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

    data = tiny.reshape(-1, 3).astype(np.float32)
    k = 16
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    quantized = centers[labels.flatten()].reshape(tiny.shape).astype(np.uint8)

    result = cv2.resize(quantized, (w, h), interpolation=cv2.INTER_NEAREST)
    return _encode(result)


def chalk_charcoal(image_bytes: bytes) -> bytes:
    """Dark textured charcoal/chalk sketch on gray paper."""
    img = _decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), 12)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    dark_sketch = 255 - cv2.multiply((255 - sketch).astype(np.float32), 1.3)
    dark_sketch = np.clip(dark_sketch, 0, 255).astype(np.uint8)

    paper = np.full_like(dark_sketch, 210)
    rng = np.random.default_rng()
    texture = rng.normal(0, 10, dark_sketch.shape).astype(np.float32)
    result_gray = np.clip(np.minimum(dark_sketch, paper).astype(np.float32) + texture, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
    return _encode(result)


def holographic(image_bytes: bytes) -> bytes:
    """Iridescent rainbow-sheen overlay that shifts hue diagonally across the image."""
    img = _decode(image_bytes).astype(np.float32)
    h, w = img.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    diag = (xx / w + yy / h) / 2.0
    hue = (diag * 179).astype(np.uint8)
    sat = np.full((h, w), 255, dtype=np.uint8)
    val = np.full((h, w), 255, dtype=np.uint8)
    rainbow_hsv = cv2.merge([hue, sat, val])
    rainbow_bgr = cv2.cvtColor(rainbow_hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    overlayed = _blend_soft_light(img, rainbow_bgr)
    result = img * 0.65 + overlayed * 0.35
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def crt_tv(image_bytes: bytes) -> bytes:
    """Old CRT television look: horizontal scanlines, RGB subpixel fringe, dark corners."""
    img = _decode(image_bytes).astype(np.float32)
    h, w = img.shape[:2]

    b, g, r = cv2.split(img)
    r = np.roll(r, 1, axis=1)
    b = np.roll(b, -1, axis=1)
    fringed = cv2.merge([b, g, r])

    scanline_mask = np.ones((h, 1), dtype=np.float32)
    scanline_mask[::2] = 0.72
    fringed = fringed * scanline_mask[:, :, None]

    x_k = cv2.getGaussianKernel(w, w * 0.7)
    y_k = cv2.getGaussianKernel(h, h * 0.7)
    vmask = (y_k @ x_k.T)
    vmask = vmask / vmask.max()
    vmask = 0.55 + 0.45 * vmask

    result = fringed * vmask[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv2.convertScaleAbs(result, alpha=1.08, beta=0)
    return _encode(result)


def frost_ice(image_bytes: bytes) -> bytes:
    """Cool crystalline frost/ice glaze - blue-white edge shimmer over a desaturated cold base."""
    img = _decode(image_bytes)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.55, 0, 255)
    cool_base = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    b, g, r = cv2.split(cool_base)
    b = np.clip(b * 1.2, 0, 255)
    r = np.clip(r * 0.9, 0, 255)
    cool_base = cv2.merge([b, g, r])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    shimmer = cv2.GaussianBlur(edges, (0, 0), 3).astype(np.float32)
    frost_overlay = np.zeros_like(cool_base)
    frost_overlay[:, :, 0] = shimmer * 1.1
    frost_overlay[:, :, 1] = shimmer * 1.0
    frost_overlay[:, :, 2] = shimmer * 0.95

    result = cool_base + frost_overlay * 0.8
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def solarize(image_bytes: bytes) -> bytes:
    """Partial tone inversion (Sabattier effect) - bright areas flip while shadows stay normal."""
    img = _decode(image_bytes)
    threshold = 128
    inverted = 255 - img
    mask = (img > threshold)
    result = np.where(mask, inverted, img).astype(np.uint8)
    return _encode(result)


def copper_etch(image_bytes: bytes) -> bytes:
    """Metallic engraved copper-tone relief, like an old printing plate."""
    img = _decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2],
    ])
    relief = cv2.filter2D(gray, -1, kernel).astype(np.float32) + 128
    relief = np.clip(relief, 0, 255)

    result = np.zeros((*relief.shape, 3), dtype=np.float32)
    result[:, :, 0] = relief * 0.35
    result[:, :, 1] = relief * 0.65
    result[:, :, 2] = relief * 1.05
    return _encode(np.clip(result, 0, 255).astype(np.uint8))


def galaxy(image_bytes: bytes) -> bytes:
    """Nebula/starfield color overlay blended into the image's shadow regions."""
    img = _decode(image_bytes).astype(np.float32)
    h, w = img.shape[:2]
    rng = np.random.default_rng()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n1 = np.sin(xx / 40.0) * np.cos(yy / 55.0)
    n2 = np.sin((xx + yy) / 70.0)
    nebula_pattern = (n1 * 0.6 + n2 * 0.4)
    nebula_pattern = (nebula_pattern - nebula_pattern.min()) / (np.ptp(nebula_pattern) + 1e-6)

    nebula = np.zeros((h, w, 3), dtype=np.float32)
    nebula[:, :, 0] = 120 + nebula_pattern * 100
    nebula[:, :, 1] = 20 + nebula_pattern * 60
    nebula[:, :, 2] = 90 + nebula_pattern * 120

    stars = np.zeros((h, w), dtype=np.float32)
    n_stars = (h * w) // 900
    sy = rng.integers(0, h, n_stars)
    sx = rng.integers(0, w, n_stars)
    stars[sy, sx] = rng.uniform(150, 255, n_stars)
    stars = cv2.GaussianBlur(stars, (0, 0), 0.6)
    star_layer = np.repeat(stars[:, :, None], 3, axis=2)

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    shadow_mask = np.clip((110 - gray) / 110, 0, 1)[:, :, None]

    blended = _blend_screen(img, nebula) * shadow_mask + img * (1 - shadow_mask)
    result = blended + star_layer * 0.5
    return _encode(np.clip(result, 0, 255).astype(np.uint8))
