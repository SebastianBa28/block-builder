"""
Offline test for GripAnalyzer thresholds using Depth Anything V2.
Runs on all *.jpg images in data/block_grips/ and prints scores + classifications.
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

# Add src to path so we can import legobuilder without ROS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "legobuilder"))

from legobuilder.vision.grip_analyzer import GripAnalyzer, GripQuality
from legobuilder.config import (
    GRIP_DIAGONAL_THRESHOLD, GRIP_HEIGHT_THRESHOLD,
    GRIP_ROI_X_START, GRIP_ROI_X_END,
    GRIP_ROI_Y_START, GRIP_ROI_Y_END,
    GRIP_FINGER_TOP_FRAC, GRIP_FINGER_BOT_FRAC,
)

# Expected label → acceptable GripQuality values
EXPECTED = {
    "good":         {GripQuality.GOOD},
    "diagonal":     {GripQuality.DIAGONAL},
    "low":          {GripQuality.LOW},
    "partial_high": {GripQuality.HIGH, GripQuality.DIAGONAL},
    "partial_low":  {GripQuality.LOW, GripQuality.DIAGONAL},
}


def main():
    from transformers import pipeline

    print("Loading Depth Anything V2 Base...")
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")
    print("Model loaded.\n")

    # Prepare output directory for depth maps
    depth_dir = SCRIPT_DIR / "depths"
    depth_dir.mkdir(exist_ok=True)

    analyzer = GripAnalyzer()
    images = sorted(SCRIPT_DIR.glob("*.jpg"))

    print(f"Found {len(images)} images")
    print(f"Current thresholds: DIAGONAL={GRIP_DIAGONAL_THRESHOLD}, HEIGHT={GRIP_HEIGHT_THRESHOLD}")
    print("=" * 90)
    print(f"{'Filename':<30} {'Expected':<14} {'Predicted':<10} {'Diag Score':>10} {'Hgt Score':>10} {'OK?':>5}")
    print("-" * 90)

    results = []
    grouped = defaultdict(list)

    for img_path in images:
        # Parse expected label from filename: {color}-{label}.jpg
        stem = img_path.stem  # e.g. "blue-diagonal"
        parts = stem.split("-", 1)
        if len(parts) != 2:
            print(f"  Skipping {img_path.name} (unexpected name format)")
            continue
        color, label = parts

        # Run depth estimation
        pil_img = Image.open(img_path).convert("RGB")
        depth_output = pipe(pil_img)
        depth_pil = depth_output["depth"]  # PIL Image

        # Convert to uint8 numpy
        depth_np = np.array(depth_pil)
        if depth_np.dtype != np.uint8:
            # Normalize to 0-255 uint8
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max > d_min:
                depth_np = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_np = np.zeros_like(depth_np, dtype=np.uint8)

        # Save depth map with ROI annotation overlay
        h_img, w_img = depth_np.shape
        depth_rgb = Image.fromarray(depth_np).convert("RGBA")
        overlay = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # ROI pixel coordinates
        rx0 = int(w_img * GRIP_ROI_X_START)
        rx1 = int(w_img * GRIP_ROI_X_END)
        ry0 = int(h_img * GRIP_ROI_Y_START)
        ry1 = int(h_img * GRIP_ROI_Y_END)
        roi_h = ry1 - ry0

        # Sub-region boundaries (in image coords)
        finger_top_end = ry0 + int(roi_h * GRIP_FINGER_TOP_FRAC)
        finger_bot_start = ry0 + int(roi_h * GRIP_FINGER_BOT_FRAC)

        alpha = 77  # ~0.3 * 255
        finger_color = (0, 120, 255, alpha)  # blue
        block_color = (0, 255, 0, alpha)     # green

        draw.rectangle([rx0, ry0, rx1, finger_top_end], fill=finger_color)
        draw.rectangle([rx0, finger_top_end, rx1, finger_bot_start], fill=block_color)
        draw.rectangle([rx0, finger_bot_start, rx1, ry1], fill=finger_color)

        annotated = Image.alpha_composite(depth_rgb, overlay)
        annotated.save(depth_dir / f"{stem}-depth.png")

        # Analyze
        result = analyzer.analyze(depth_np)
        acceptable = EXPECTED.get(label, set())
        ok = result.quality in acceptable

        row = {
            "filename": img_path.name,
            "color": color,
            "label": label,
            "predicted": result.quality.name,
            "diagonal_score": result.diagonal_score,
            "height_score": result.height_score,
            "ok": ok,
        }
        results.append(row)
        grouped[label].append(row)

        ok_str = "YES" if ok else "**NO**"
        print(f"{img_path.name:<30} {label:<14} {result.quality.name:<10} {result.diagonal_score:>10.2f} {result.height_score:>10.2f} {ok_str:>5}")

    # Summary
    n_correct = sum(1 for r in results if r["ok"])
    print("=" * 90)
    print(f"\nOverall: {n_correct}/{len(results)} correct\n")

    # Grouped summary
    print("=" * 90)
    print("GROUPED BY LABEL:")
    print("=" * 90)
    for label in ["good", "diagonal", "low", "partial_high", "partial_low"]:
        rows = grouped.get(label, [])
        if not rows:
            continue
        print(f"\n--- {label.upper()} ---")
        diag_scores = [r["diagonal_score"] for r in rows]
        hgt_scores = [r["height_score"] for r in rows]
        print(f"  Diagonal scores: {[f'{s:.2f}' for s in diag_scores]}")
        print(f"  Height scores:   {[f'{s:.2f}' for s in hgt_scores]}")
        print(f"  Predictions:     {[r['predicted'] for r in rows]}")
        all_ok = all(r["ok"] for r in rows)
        print(f"  All correct:     {'YES' if all_ok else 'NO'}")


if __name__ == "__main__":
    main()
