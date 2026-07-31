"""
Classic image-processing styles using OpenCV — zero API cost, zero rate limits,
run entirely on your own server. These don't produce true "anime" quality but give
a solid cartoon/sketch effect instantly and for free.
"""

import cv2
import numpy as np


def cartoonify(image_bytes: bytes) -> bytes:
    """Bilateral-filter + edge-mask cartoon effect."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Downscale for speed on large photos, upscale result back at the end
    h, w = img.shape[:2]
    max_dim = 1024
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Edge mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )

    # Smooth color regions (repeated bilateral filter approximates "cel shading")
    color = img
    for _ in range(5):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # Reduce color palette (posterize) for a flatter cartoon look
    div = 24
    color = (color // div) * div + div // 2

    cartoon = cv2.bitwise_and(color, color, mask=edges)

    success, encoded = cv2.imencode(".jpg", cartoon, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError("Failed to encode cartoonified image")
    return encoded.tobytes()


def pencil_sketch(image_bytes: bytes, color: bool = False) -> bytes:
    """Pencil sketch effect. Set color=True for a lightly colored sketch."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    gray_sketch, color_sketch = cv2.pencilSketch(
        img, sigma_s=60, sigma_r=0.07, shade_factor=0.05
    )

    result = color_sketch if color else gray_sketch

    success, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError("Failed to encode sketch image")
    return encoded.tobytes()


def oil_painting(image_bytes: bytes) -> bytes:
    """Stylization filter — gives a soft painterly look, still free/local."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    result = cv2.stylization(img, sigma_s=60, sigma_r=0.45)

    success, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError("Failed to encode stylized image")
    return encoded.tobytes()
