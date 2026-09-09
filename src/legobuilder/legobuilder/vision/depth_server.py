r"""Standalone FastAPI server for Depth Anything V2 monocular depth estimation.

Runs outside the ROS graph (no ROS dependencies) and serves depth
inference over HTTP.  The detector node sends camera frames to the
/depth endpoint and receives uint8 depth maps plus grip quality
analysis results.

Usage::

    .venv/bin/uvicorn legobuilder.vision.depth_server:app \
        --host 0.0.0.0 --port 8000
"""
import io
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response


_pipe = None
_grip_analyzer = None
SAVE_DIR = Path.home() / 'robotws' / 'data' / 'runtime_grip_depths'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Depth Anything model on startup, release on shutdown."""
    global _pipe, _grip_analyzer
    import torch
    from transformers import pipeline as hf_pipeline

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from legobuilder.vision.grip_analyzer import GripAnalyzer

    model_id = app.state.model_id
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading Depth Anything model: {model_id} (device={device})")
    _pipe = hf_pipeline("depth-estimation", model=model_id, device=device)
    _grip_analyzer = GripAnalyzer()
    print("Model loaded.")
    yield
    _pipe = None
    _grip_analyzer = None


app = FastAPI(lifespan=lifespan)
app.state.model_id = 'depth-anything/Depth-Anything-V2-Base-hf'


@app.get("/health")
def health():
    """Return server readiness status.

    Returns
    -------
    dict
        {"ready": bool} indicating whether the depth model is loaded.
    """
    return {"ready": _pipe is not None}


@app.post("/depth")
async def estimate_depth(
    file: UploadFile = File(...),
    request_id: str = Query(default=""),
):
    """Run depth estimation and grip analysis on an uploaded image.

    Crops the left background portion, runs Depth Anything V2 inference,
    normalises the result to uint8, performs grip quality analysis, and
    saves debug artefacts to disk.

    Arguments
    ---------
    file : UploadFile
        JPEG/PNG image from the end-effector camera.
    request_id : str
        Optional identifier for correlating request/response pairs.

    Returns
    -------
    Response
        Raw uint8 depth map bytes with X-Depth-Height and
        X-Depth-Width headers.
    """
    from PIL import Image

    from legobuilder.config import GRIP_IMAGE_CROP_X

    image_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    crop_x = int(pil_img.width * GRIP_IMAGE_CROP_X)
    pil_cropped = pil_img.crop((crop_x, 0, pil_img.width, pil_img.height))

    result = _pipe(pil_cropped)
    depth_np = np.array(result["depth"])

    # Normalize to uint8
    if depth_np.dtype != np.uint8:
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_np = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_np = np.zeros_like(depth_np, dtype=np.uint8)

    # Run grip analysis
    print("Running grip analysis...")
    analysis = _grip_analyzer.analyze(depth_np)

    # Save image, depth, and result
    print("Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"grip_check_{timestamp}"
    if request_id:
        folder_name += f"-{request_id}"
    save_dir = SAVE_DIR / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pil_img.save(save_dir / 'image_original.jpg')
    pil_cropped.save(save_dir / 'image.jpg')
    Image.fromarray(depth_np).save(save_dir / 'depth.png')
    with open(save_dir / 'result.json', 'w') as f:
        json.dump({
            'quality': analysis.quality.name,
            'quality_int': int(analysis.quality),
            'diagonal_score': analysis.diagonal_score,
            'height_score': analysis.height_score,
            'column_stds': analysis.column_stds,
            'n_high_columns': analysis.n_high_columns,
            'request_id': request_id,
            'timestamp': timestamp,
        }, f, indent=2)

    print(f"Saved: {save_dir} | {analysis.quality.name} "
          f"diag={analysis.diagonal_score:.2f} height={analysis.height_score:.2f}")

    dh, dw = depth_np.shape
    return Response(
        content=depth_np.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Depth-Height": str(dh),
            "X-Depth-Width": str(dw),
        },
    )


if __name__ == '__main__':
    import sys
    import uvicorn

    if len(sys.argv) > 1:
        app.state.model_id = sys.argv[1]

    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
