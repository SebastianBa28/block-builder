"""Interactive tool to mark trapezoid ROIs on a grip depth image.

Click 4 corners for each region (top-left, top-right, bottom-right, bottom-left).
Regions to mark in order:
  1. Top finger
  2. Block
  3. Bottom finger

Press 'r' to reset the current region.
Press 'q' to quit early.

Prints normalized (fractional) coordinates at the end.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Pick first available folder by default, or pass folder name as arg
data_dir = Path(__file__).parent
if len(sys.argv) > 1:
    folder = data_dir / sys.argv[1]
else:
    folders = sorted([f for f in data_dir.iterdir() if (f / 'depth.png').exists()])
    folder = folders[0]

depth_path = folder / 'depth.png'
color_path = folder / 'image.jpg'

print(f'depth: {depth_path}')

# Load both, use color as base for clicking
color_img = cv2.imread(str(color_path))
depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
depth_rgb = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)

h, w = color_img.shape[:2]

REGION_NAMES = ['Top Finger', 'Block', 'Bottom Finger']
REGION_COLORS = [(255, 100, 0), (0, 255, 0), (255, 100, 0)]  # BGR
regions = {}  # name -> list of 4 (x,y) points

current_points = []
current_region_idx = 0


def draw_overlay(base_img):
    """Draw all completed regions and current points on the image."""
    img = base_img.copy()

    # Draw completed regions
    for name, pts in regions.items():
        idx = REGION_NAMES.index(name)
        color = REGION_COLORS[idx]
        poly = np.array(pts, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.polylines(img, [poly], True, color, 2)
        # Label
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(img, name, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw current points
    for i, pt in enumerate(current_points):
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
        cv2.putText(img, str(i+1), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if i > 0:
            cv2.line(img, current_points[i-1], pt, (0, 0, 255), 1)

    # Instructions
    if current_region_idx < len(REGION_NAMES):
        txt = f"Mark: {REGION_NAMES[current_region_idx]} ({len(current_points)}/4 corners) | 'r'=reset 'q'=quit"
    else:
        txt = "All done! Press any key to finish."
    cv2.putText(img, txt, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img


def on_mouse(event, x, y, flags, param):
    global current_points, current_region_idx

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_region_idx >= len(REGION_NAMES):
            return
        current_points.append((x, y))

        if len(current_points) == 4:
            regions[REGION_NAMES[current_region_idx]] = list(current_points)
            print(f"  {REGION_NAMES[current_region_idx]}: {current_points}")
            current_points = []
            current_region_idx += 1

        # Redraw on both windows
        cv2.imshow('Color', draw_overlay(color_img))
        cv2.imshow('Depth', draw_overlay(depth_rgb))


print(f"Image: {folder.name} ({w}x{h})")
print(f"Click 4 corners (TL, TR, BR, BL) for each region.")
print()

cv2.namedWindow('Color')
cv2.namedWindow('Depth')
cv2.setMouseCallback('Color', on_mouse)
cv2.setMouseCallback('Depth', on_mouse)

cv2.imshow('Color', draw_overlay(color_img))
cv2.imshow('Depth', draw_overlay(depth_rgb))

while True:
    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        current_points = []
        cv2.imshow('Color', draw_overlay(color_img))
        cv2.imshow('Depth', draw_overlay(depth_rgb))
    if current_region_idx >= len(REGION_NAMES) and len(current_points) == 0:
        # All regions marked, wait for final keypress
        cv2.imshow('Color', draw_overlay(color_img))
        cv2.imshow('Depth', draw_overlay(depth_rgb))
        cv2.waitKey(0)
        break

cv2.destroyAllWindows()

# Print results
print("\n=== Results (pixel coordinates) ===")
for name, pts in regions.items():
    print(f"{name}: {pts}")

print("\n=== Results (normalized 0-1) ===")
for name, pts in regions.items():
    norm = [(round(x/w, 3), round(y/h, 3)) for x, y in pts]
    print(f"{name}: {norm}")

print("\n=== Config-ready (paste into config.py) ===")
if len(regions) == 3:
    all_pts = []
    for name in REGION_NAMES:
        all_pts.extend(regions[name])
    xs = [p[0]/w for p in all_pts]
    ys = [p[1]/h for p in all_pts]
    print(f"# Trapezoid ROI corners (normalized)")
    for name in REGION_NAMES:
        pts = regions[name]
        norm = [(round(x/w, 3), round(y/h, 3)) for x, y in pts]
        label = name.upper().replace(' ', '_')
        print(f"GRIP_{label}_TRAP = {norm}  # TL, TR, BR, BL")
