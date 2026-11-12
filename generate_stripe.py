import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
params = json.loads((ROOT / "parameters.json").read_text(encoding="utf-8"))
x = np.arange(params["width"], dtype=np.float64)
intensity = params["mean_gray"] + params["modulation"] * np.cos(
    2 * np.pi * params["horizontal_periods"] * x / params["width"] + params["phase_radians"]
)
row = np.rint(np.clip(intensity, 0, 255)).astype(np.uint8)
image = np.tile(row, (params["height"], 1))
Image.fromarray(image).save(ROOT / "stripe.png")
print(f"Saved stripe.png: {params['width']}x{params['height']}, range {image.min()}–{image.max()}, mean {image.mean():.6f}")
