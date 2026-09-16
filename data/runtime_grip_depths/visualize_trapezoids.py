"""Overlay trapezoid ROIs from config on each runtime depth map."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# From config.py
GRIP_TOP_FINGER_TRAP = [(0.75, 0.348), (0.825, 0.188), (0.998, 0.138), (0.994, 0.31)]
GRIP_BLOCK_TRAP = [(0.797, 0.36), (0.997, 0.331), (0.997, 0.688), (0.806, 0.656)]
GRIP_BOTTOM_FINGER_TRAP = [(0.827, 0.825), (0.742, 0.665), (0.997, 0.7), (0.997, 0.871)]

data_dir = Path(__file__).parent

for folder in sorted(data_dir.iterdir()):
    depth_path = folder / 'depth.png'
    image_path = folder / 'image.jpg'
    if not depth_path.exists():
        continue

    depth = np.array(Image.open(depth_path).convert('L'))
    h, w = depth.shape
    color = Image.open(image_path).convert('RGB') if image_path.exists() else None

    for label, src in [('depth', depth), ('color', color)]:
        if src is None:
            continue
        if isinstance(src, np.ndarray):
            img = Image.fromarray(np.stack([src]*3, axis=-1))
        else:
            img = src.copy()

        draw = ImageDraw.Draw(img, 'RGBA')

        def to_pixels(pts):
            return [(int(x * w), int(y * h)) for x, y in pts]

        # Top finger (blue)
        pts = to_pixels(GRIP_TOP_FINGER_TRAP)
        draw.polygon(pts, fill=(0, 100, 255, 80), outline=(0, 100, 255, 200))
        # Block (green)
        pts = to_pixels(GRIP_BLOCK_TRAP)
        draw.polygon(pts, fill=(0, 255, 0, 60), outline=(0, 255, 0, 200))
        # Bottom finger (blue)
        pts = to_pixels(GRIP_BOTTOM_FINGER_TRAP)
        draw.polygon(pts, fill=(0, 100, 255, 80), outline=(0, 100, 255, 200))

        out_path = folder / f'trap_overlay_{label}.png'
        img.save(out_path)
        print(f"Saved: {out_path}")
