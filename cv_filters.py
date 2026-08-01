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


def aesthetic_blur(image_bytes: bytes, intensity: int = 50, angle: float = 90.0) -> bytes:
    """
    Directional motion-blur "dreamy drift" effect — like a subject caught
    mid-movement in a field, streaked softly in one direction, rather than
    an even all-over Gaussian blur. Intensity 1-100 controls streak length
    and how much of the original sharp image blends back in.
    """
    intensity = max(1, min(100, int(intensity)))
    img = _decode(image_bytes)
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
