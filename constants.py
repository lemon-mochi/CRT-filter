"""
constants.py
------------
Contains the constant values that will be used by other files
"""


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