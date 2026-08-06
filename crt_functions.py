"""
crt_functions.py
----------------
Contains the helper functions used to modify the images and apply the CRT filter
"""

import numpy as np
import cv2

from constants import PRESETS

# --------------------------------------------------------------------------
# Individual effect functions. Each takes a float32 BGR image in [0,255]
# and returns a float32 BGR image in [0,255].
# --------------------------------------------------------------------------

def barrel_distort(img, strength=0.15):
    """Warp the image outward at the edges to simulate curved CRT glass."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    # Normalized coordinate grid centered at 0
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r2 = x ** 2 + y ** 2
    # Barrel distortion: pull sample points toward the center more at the edges
    factor = 1 + strength * r2
    map_x = ((x * factor) * 0.5 + 0.5) * (w - 1)
    map_y = ((y * factor) * 0.5 + 0.5) * (h - 1)
    distorted = cv2.remap(
        img, map_x.astype(np.float32), map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return distorted


def chromatic_aberration(img, shift=2):
    """Shift the red and blue channels slightly apart from green."""
    if shift <= 0:
        return img
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)

    m_r = np.float32([[1, 0, shift], [0, 1, 0]])
    m_b = np.float32([[1, 0, -shift], [0, 1, 0]])
    r = cv2.warpAffine(r, m_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(b, m_b, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return cv2.merge([b, g, r])


def scanlines(img, intensity=0.25, line_spacing=2):
    """Darken every other line (or every Nth line) to mimic the raster scan."""
    if intensity <= 0:
        return img
    h = img.shape[0]
    mask = np.ones((h, 1, 1), dtype=np.float32)
    mask[::line_spacing] = 1.0 - intensity
    return img * mask


def phosphor_mask(img, intensity=0.15):
    """Overlay a repeating RGB subpixel stripe pattern like a shadow mask."""
    if intensity <= 0:
        return img
    h, w = img.shape[:2]
    # One RGB triad tile, repeated across the width
    tile = np.array([
        [1.0, 1.0 - intensity, 1.0 - intensity],  # R stripe (B,G,R order)
        [1.0 - intensity, 1.0, 1.0 - intensity],  # G stripe
        [1.0 - intensity, 1.0 - intensity, 1.0],  # B stripe
    ], dtype=np.float32)
    reps = w // 3 + 1
    row = np.tile(tile, (reps, 1))[:w]
    mask = np.tile(row[np.newaxis, :, :], (h, 1, 1))
    return img * mask


def bloom(img, threshold=180, blur_size=15, intensity=0.35):
    """Add a soft glow around bright regions, like phosphor light bleed."""
    if intensity <= 0:
        return img
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.float32) / 255.0
    bright = img * mask[:, :, np.newaxis]
    glow = cv2.GaussianBlur(bright, (0, 0), sigmaX=blur_size)
    return np.clip(img + glow * intensity, 0, 255)


def vignette(img, strength=0.4):
    """Darken the corners of the frame."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(x ** 2 + y ** 2)
    mask = 1 - strength * np.clip(r - 0.3, 0, None) ** 2
    mask = np.clip(mask, 0, 1)
    return img * mask[:, :, np.newaxis]


def noise_and_flicker(img, noise_amount=6, flicker_amount=0.03, rng=None):
    """Add faint analog static and a slight random brightness flicker."""
    if rng is None:
        rng = np.random
    out = img
    if noise_amount > 0:
        static = rng.normal(0, noise_amount, img.shape).astype(np.float32)
        out = out + static
    if flicker_amount > 0:
        flicker = 1.0 + (rng.random() - 0.5) * 2 * flicker_amount
        out = out * flicker
    return np.clip(out, 0, 255)


def rounded_bezel(img, corner_radius_frac=0.06):
    """Mask the four corners black to suggest a curved tube bezel."""
    if corner_radius_frac <= 0:
        return img
    h, w = img.shape[:2]
    radius = int(min(h, w) * corner_radius_frac)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    # Draw four filled black quarter-circles into the corners of an all-white mask,
    # then AND everything together via a rounded-rectangle mask.
    rr_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(rr_mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(rr_mask, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in [(radius, radius), (w - radius, radius),
                   (radius, h - radius), (w - radius, h - radius)]:
        cv2.circle(rr_mask, (cx, cy), radius, 255, -1)
    mask = rr_mask.astype(np.float32) / 255.0
    return img * mask[:, :, np.newaxis]


def apply_crt_effect(frame_bgr, params, rng=None):
    """Run the full CRT pipeline on a single BGR uint8 frame."""
    img = frame_bgr.astype(np.float32)

    img = barrel_distort(img, params["barrel"])
    img = chromatic_aberration(img, params["aberration"])
    img = scanlines(img, params["scan_intensity"])
    img = phosphor_mask(img, params["phosphor_intensity"])
    img = bloom(img, intensity=params["bloom_intensity"])
    img = vignette(img, params["vignette_strength"])
    img = noise_and_flicker(img, params["noise_amount"], params["flicker_amount"], rng)
    img = rounded_bezel(img, params["bezel_radius"])

    return np.clip(img, 0, 255).astype(np.uint8)