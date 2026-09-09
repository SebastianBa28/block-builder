"""Overlay current ROI regions on each runtime depth map for visual confirmation."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# Current config values
# Proposed new values (shifted right to match new camera position)
GRIP_ROI_X_START = 0.75
GRIP_ROI_X_END = 0.98
GRIP_ROI_Y_START = 0.12
GRIP_ROI_Y_END = 0.88
GRIP_FINGER_TOP_FRAC = 0.17
GRIP_FINGER_BOT_FRAC = 0.83

data_dir = Path(__file__).parent

for folder in sorted(data_dir.iterdir()):
    depth_path = folder / 'depth.png'
    image_path = folder / 'image.jpg'
    if not depth_path.exists():
        continue

    # Load depth as grayscale, convert to RGB for drawing
    depth = np.array(Image.open(depth_path).convert('L'))
    h, w = depth.shape

    # Also load the color image
    color = Image.open(image_path).convert('RGB') if image_path.exists() else None

    for label, src in [('depth', depth), ('color', color)]:
        if src is None:
            continue
        if isinstance(src, np.ndarray):
            img = Image.fromarray(np.stack([src]*3, axis=-1))
        else:
            img = src.copy()

        draw = ImageDraw.Draw(img, 'RGBA')

        # Outer ROI box
        x0 = int(w * GRIP_ROI_X_START)
        x1 = int(w * GRIP_ROI_X_END)
        y0 = int(h * GRIP_ROI_Y_START)
        y1 = int(h * GRIP_ROI_Y_END)

        roi_h = y1 - y0
        finger_top_end = y0 + int(roi_h * GRIP_FINGER_TOP_FRAC)
        finger_bot_start = y0 + int(roi_h * GRIP_FINGER_BOT_FRAC)

        # Fill finger regions (blue, semi-transparent)
        draw.rectangle([x0, y0, x1, finger_top_end], fill=(0, 100, 255, 80), outline=(0, 100, 255, 200), width=2)
        draw.rectangle([x0, finger_bot_start, x1, y1], fill=(0, 100, 255, 80), outline=(0, 100, 255, 200), width=2)

        # Fill block region (green, semi-transparent)
        draw.rectangle([x0, finger_top_end, x1, finger_bot_start], fill=(0, 255, 0, 60), outline=(0, 255, 0, 200), width=2)

        # Outer ROI outline (red)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)

        out_path = folder / f'roi_overlay_{label}.png'
        img.save(out_path)
        print(f"Saved: {out_path}")

print("\nROI config:")
print(f"  X: {GRIP_ROI_X_START:.2f} - {GRIP_ROI_X_END:.2f}")
print(f"  Y: {GRIP_ROI_Y_START:.2f} - {GRIP_ROI_Y_END:.2f}")
print(f"  Finger top frac: {GRIP_FINGER_TOP_FRAC:.2f}")
print(f"  Finger bot frac: {GRIP_FINGER_BOT_FRAC:.2f}")
print(f"\nImage size: {w}x{h}")
print(f"ROI pixels: x=[{x0},{x1}] y=[{y0},{y1}]  ({x1-x0}x{y1-y0})")
print(f"Finger top: y=[{y0},{finger_top_end}]  Block: y=[{finger_top_end},{finger_bot_start}]  Finger bot: y=[{finger_bot_start},{y1}]")
