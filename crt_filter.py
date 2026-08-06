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

from crt_functions import apply_crt_effect, cv2, np
from constants import PRESETS

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