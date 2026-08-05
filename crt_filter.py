#!/usr/bin/env python3
"""
crt_filter.py - Give an image or video that old tube-television look.

Effects applied (in order):
  1. Barrel distortion   - simulates the curved glass of a CRT screen
  2. Chromatic aberration - RGB channels shift slightly, like analog signal bleed
  3. Scanlines            - dark horizontal lines mimicking the electron beam raster
  4. Phosphor / shadow mask - RGB subpixel stripes like a real tube's phosphor dots
  5. Bloom / glow         - bright areas bleed light, like phosphor glow
  6. Vignette             - darkened corners, like screen falloff
  7. Noise + flicker      - faint analog static and brightness flicker
  8. Rounded corners      - optional dark rounded bezel mask

Works on:
  - Still images (jpg, png, bmp, ...)
  - Video files (mp4, avi, mov, ...) - processed frame by frame
  - Webcam / live capture (pass --input 0)

Usage:
    python3 crt_filter.py --input photo.jpg --output photo_crt.png
    python3 crt_filter.py --input clip.mp4  --output clip_crt.mp4
    python3 crt_filter.py --input 0 --output live_crt.mp4 --seconds 5

Tune the look with --intensity light|medium|heavy or override individual
parameters (see --help).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


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


# --------------------------------------------------------------------------
# Preset intensity levels: tuned bundles of the parameters above.
# --------------------------------------------------------------------------

PRESETS = {
    "light": dict(
        barrel=0.08, aberration=1, scan_intensity=0.15, phosphor_intensity=0.10,
        bloom_intensity=0.20, vignette_strength=0.25, noise_amount=3,
        flicker_amount=0.015, bezel_radius=0.03,
    ),
    "medium": dict(
        barrel=0.15, aberration=2, scan_intensity=0.25, phosphor_intensity=0.15,
        bloom_intensity=0.35, vignette_strength=0.40, noise_amount=6,
        flicker_amount=0.03, bezel_radius=0.06,
    ),
    "heavy": dict(
        barrel=0.25, aberration=4, scan_intensity=0.40, phosphor_intensity=0.25,
        bloom_intensity=0.55, vignette_strength=0.55, noise_amount=10,
        flicker_amount=0.05, bezel_radius=0.09,
    ),
}


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


# --------------------------------------------------------------------------
# I/O handling: figure out if input is an image, a video file, or a webcam.
# --------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image_path(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in IMAGE_EXTS


def process_image(input_path, output_path, params):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    rng = np.random.default_rng(0)
    result = apply_crt_effect(img, params, rng)
    cv2.imwrite(output_path, result)
    print(f"Saved CRT-filtered image to {output_path}")


def process_video(input_source, output_path, params, max_seconds=None):
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {input_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps <= 0 or fps != fps:  # guard against 0 or NaN from some webcams
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    rng = np.random.default_rng(0)
    start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = apply_crt_effect(frame, params, rng)
        writer.write(result)
        frame_count += 1

        if total_frames > 0:
            pct = 100 * frame_count / total_frames
            print(f"\rProcessing frame {frame_count}/{total_frames} ({pct:.1f}%)", end="")
        else:
            print(f"\rProcessing frame {frame_count}", end="")

        if max_seconds is not None and (time.time() - start) >= max_seconds:
            break

    print()
    cap.release()
    writer.release()
    print(f"Saved CRT-filtered video to {output_path} ({frame_count} frames)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Apply an old CRT tube-television filter to an image or video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                    help="Path to input image/video, or '0' for the default webcam.")
    p.add_argument("--output", required=True,
                    help="Path to save the filtered output.")
    p.add_argument("--intensity", choices=list(PRESETS.keys()), default="medium",
                    help="Overall strength preset for the effect.")
    p.add_argument("--seconds", type=float, default=None,
                    help="Limit video/webcam capture to N seconds (webcam mode).")

    # Optional fine-grained overrides
    p.add_argument("--barrel", type=float, default=None, help="Override barrel distortion strength.")
    p.add_argument("--aberration", type=float, default=None, help="Override chromatic aberration pixel shift.")
    p.add_argument("--scan-intensity", type=float, default=None, help="Override scanline darkness.")
    p.add_argument("--phosphor-intensity", type=float, default=None, help="Override phosphor mask strength.")
    p.add_argument("--bloom-intensity", type=float, default=None, help="Override glow/bloom strength.")
    p.add_argument("--vignette-strength", type=float, default=None, help="Override corner darkening.")
    p.add_argument("--noise-amount", type=float, default=None, help="Override static noise amount.")
    p.add_argument("--flicker-amount", type=float, default=None, help="Override brightness flicker amount.")
    p.add_argument("--bezel-radius", type=float, default=None, help="Override rounded bezel corner radius (0 disables).")

    return p


def resolve_params(args):
    params = dict(PRESETS[args.intensity])
    overrides = {
        "barrel": args.barrel,
        "aberration": args.aberration,
        "scan_intensity": args.scan_intensity,
        "phosphor_intensity": args.phosphor_intensity,
        "bloom_intensity": args.bloom_intensity,
        "vignette_strength": args.vignette_strength,
        "noise_amount": args.noise_amount,
        "flicker_amount": args.flicker_amount,
        "bezel_radius": args.bezel_radius,
    }
    for k, v in overrides.items():
        if v is not None:
            params[k] = v
    return params


def main():
    args = build_arg_parser().parse_args()
    params = resolve_params(args)

    input_arg = args.input
    # Webcam input is given as a digit, e.g. "0"
    if input_arg.isdigit():
        process_video(int(input_arg), args.output, params, max_seconds=args.seconds)
        return

    if not os.path.exists(input_arg):
        print(f"Error: input file not found: {input_arg}", file=sys.stderr)
        sys.exit(1)

    if is_image_path(input_arg):
        process_image(input_arg, args.output, params)
    else:
        process_video(input_arg, args.output, params, max_seconds=args.seconds)


if __name__ == "__main__":
    main()