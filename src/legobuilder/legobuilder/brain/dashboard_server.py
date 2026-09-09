"""Standalone FastAPI dashboard server for monitoring the build workflow.

Runs outside the ROS graph (no ROS dependencies).  The brain node
pushes events via HTTP POST; browsers consume them via Server-Sent
Events (SSE).

Usage::

    .venv/bin/python -m legobuilder.brain.dashboard_server
    # or
    .venv/bin/uvicorn legobuilder.brain.dashboard_server:app \
        --host 0.0.0.0 --port 8001
"""

import asyncio
import json
import signal
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

_ASSEMBLIES_DIR = Path(__file__).resolve().parents[2] / "assemblies"


_clients: list[asyncio.Queue] = []
_recent_events: deque = deque(maxlen=200)
_shutdown_event: asyncio.Event = asyncio.Event()
_latest_structure: dict | None = None


def _notify_shutdown():
    """Signal all SSE generators to exit."""
    _shutdown_event.set()
    for q in _clients:
        try:
            q.put_nowait(None)
        except asyncio.QueueFull:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down server state."""
    global _latest_structure
    _clients.clear()
    _recent_events.clear()
    _shutdown_event.clear()
    _latest_structure = None

    # Install signal handlers so SSE connections drop before uvicorn
    # tries to drain them (avoids the shutdown deadlock).
    loop = asyncio.get_running_loop()
    original_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        original_handlers[sig] = loop._default_executor  # save ref
        try:
            original = signal.getsignal(sig)
            original_handlers[sig] = original

            def _handler(signum, frame, _orig=original):
                _notify_shutdown()
                # Re-raise to uvicorn so it proceeds with shutdown
                if callable(_orig) and _orig not in (signal.SIG_DFL, signal.SIG_IGN):
                    _orig(signum, frame)
                else:
                    raise KeyboardInterrupt

            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass  # not main thread, skip

    print("Dashboard server ready")
    yield
    _notify_shutdown()
    _clients.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": True}


@app.post("/events")
async def receive_event(request: Request):
    """Receive an event from the brain node and fan out to SSE clients."""
    global _latest_structure
    event = await request.json()
    if event.get('type') == 'structure':
        _latest_structure = event
    elif event.get('type') == 'structure_update' and _latest_structure is not None:
        _latest_structure['data']['placed_grid'] = event['data']['placed_grid']
    _recent_events.append(event)
    for q in _clients:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass
    return {"ok": True}


@app.get("/snapshot")
def snapshot():
    """Return recent events for late-joining browsers."""
    events = list(_recent_events)
    if _latest_structure is not None:
        events = [e for e in events if e.get('type') != 'structure']
        events.insert(0, _latest_structure)
    return JSONResponse(content=events)


@app.get("/events/stream")
async def event_stream(request: Request):
    """SSE endpoint: streams events to connected browsers."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _clients.append(q)

    async def generate():
        try:
            while not _shutdown_event.is_set():
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15)
                    if event is None:
                        break  # Shutdown sentinel
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in _clients:
                _clients.remove(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/assemblies")
def list_assemblies():
    """Return list of available assembly JSON filenames."""
    files = sorted(p.name for p in _ASSEMBLIES_DIR.glob("*.json"))
    return JSONResponse(content=files)


@app.get("/assemblies/{filename}")
def get_assembly(filename: str):
    """Return the contents of an assembly JSON file."""
    path = (_ASSEMBLIES_DIR / filename).resolve()
    if not path.is_relative_to(_ASSEMBLIES_DIR.resolve()) or not path.exists():
        return JSONResponse(content={"error": "not found"}, status_code=404)
    data = json.loads(path.read_text())
    return JSONResponse(content=data)


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the dashboard HTML page."""
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Block Builder Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #F5F0E8; color: #3D3929;
    padding: 16px; line-height: 1.5;
  }
  h1 { font-size: 1.2rem; margin-bottom: 12px; }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #D9D0C1; padding-bottom: 8px; margin-bottom: 16px;
  }
  .status-dot {
    width: 10px; height: 10px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
  }
  .online { background: #3fb950; }
  .offline { background: #f85149; }
  .unknown { background: #8b949e; }
  .grid {
    display: grid;
    grid-template-columns: 2fr 2fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
  }
  .card {
    background: #FFFFFF; border: 1px solid #D9D0C1;
    border-radius: 6px; padding: 12px;
  }
  .card h2 {
    font-size: 0.85rem; color: #8C8272;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 8px;
  }
  .card.full { grid-column: 1 / -1; }
  .card.top-row { grid-column: auto; }
  .phase-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-weight: bold; font-size: 0.85rem;
    background: #1f6feb; color: #fff;
  }
  .phase-idle { background: #30363d; color: #8b949e; }
  .phase-grasp { background: #da3633; }
  .phase-approach { background: #d29922; color: #000; }
  .phase-place { background: #3fb950; color: #000; }
  .phase-scan { background: #a371f7; }
  .phase-grip_check { background: #f778ba; }
  .phase-grip_connection_detection { background: #f0883e; color: #000; }
  .phase-clearing_drop { background: #8b949e; }
  .phase-connection_drop { background: #8b949e; }
  .kv { margin: 4px 0; }
  .kv .label { color: #8C8272; font-size: 0.8rem; }
  .kv .value { color: #3D3929; }
  .health-row {
    display: flex; gap: 24px; flex-wrap: wrap;
  }
  .health-item { display: flex; align-items: center; gap: 6px; }
  #event-log {
    max-height: 300px; overflow-y: auto;
    font-size: 0.78rem; line-height: 1.6;
  }
  .log-entry { border-bottom: 1px solid #E5DDD0; padding: 2px 0; }
  .log-time { color: #8C8272; }
  .log-source { color: #2E7D32; }
  .log-type { color: #7B1FA2; }
  #three-canvas {
    width: 100%; height: 350px;
    border-radius: 4px;
    cursor: grab;
    background: #F5F0E8;
  }
  #three-canvas:active { cursor: grabbing; }
  .structure-stats {
    margin-top: 8px; font-size: 0.8rem; color: #8C8272;
  }
  #unreachable-canvas {
    width: 100%; height: 350px;
    border-radius: 4px;
    cursor: grab;
    background: #F5F0E8;
  }
  #unreachable-canvas:active { cursor: grabbing; }
  #unreachable-list {
    font-size: 0.8rem;
    line-height: 1.8;
  }
  .unreachable-entry {
    border-bottom: 1px solid #E5DDD0;
    padding: 4px 0;
  }
  .reason-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
    vertical-align: middle;
  }
  .toggle-wrap {
    display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: #8C8272;
  }
  .toggle-switch {
    position: relative; width: 36px; height: 20px; cursor: pointer;
  }
  .toggle-switch input { display: none; }
  .toggle-slider {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: #D9D0C1; border-radius: 10px; transition: 0.2s;
  }
  .toggle-slider::before {
    content: ''; position: absolute; width: 16px; height: 16px;
    left: 2px; bottom: 2px; background: #fff; border-radius: 50%; transition: 0.2s;
  }
  .toggle-switch input:checked + .toggle-slider { background: #1f6feb; }
  .toggle-switch input:checked + .toggle-slider::before { transform: translateX(16px); }
</style>
</head>
<body>
<div class="header">
  <h1>Block Builder Dashboard</h1>
  <div id="conn-status">
    <span class="status-dot unknown"></span> connecting...
  </div>
</div>

<div class="grid">
  <div class="card top-row">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <h2 style="margin-bottom:0;">Block Structure</h2>
      <div style="display:flex;align-items:center;gap:8px;">
        <select id="assembly-select" style="font-family:inherit;font-size:0.8rem;padding:4px 8px;border:1px solid #D9D0C1;border-radius:4px;background:#fff;">
          <option value="">-- Select Assembly --</option>
        </select>
        <button id="set-structure-btn" style="font-family:inherit;font-size:0.8rem;padding:4px 12px;border:1px solid #D9D0C1;border-radius:4px;background:#1f6feb;color:#fff;cursor:pointer;">Set Structure</button>
      </div>
      <div class="toggle-wrap">
        <span>Show Full</span>
        <label class="toggle-switch">
          <input type="checkbox" id="toggle-full-structure">
          <span class="toggle-slider"></span>
        </label>
      </div>
    </div>
    <canvas id="three-canvas"></canvas>
    <div class="structure-stats">
      <span id="struct-info">Waiting for structure data...</span>
    </div>
  </div>

  <div id="unreachable-card" class="card">
    <h2>Unreachable Blocks</h2>
    <canvas id="unreachable-canvas"></canvas>
  </div>

  <div id="unreachable-panel" class="card">
    <h2>Block Details</h2>
    <div id="unreachable-list">
      <div style="color:#8C8272;font-size:0.8rem;">No unreachable blocks</div>
    </div>
  </div>

  <div class="card full">
    <h2>Build Pipeline</h2>
    <div class="kv">
      <span class="label">Phase: </span>
      <span id="pipeline-phase" class="phase-badge phase-idle">IDLE</span>
    </div>
    <div class="kv">
      <span class="label">Target Cell: </span>
      <span id="pipeline-target" class="value">--</span>
    </div>
    <div class="kv">
      <span class="label">Block Type: </span>
      <span id="pipeline-block" class="value">--</span>
    </div>
    <div class="kv">
      <span class="label">Detail: </span>
      <span id="pipeline-detail" class="value">--</span>
    </div>
  </div>

  <div class="card">
    <h2>Gestures</h2>
    <div class="kv">
      <span class="label">State: </span>
      <span id="gesture-state" class="phase-badge phase-idle">INACTIVE</span>
    </div>
    <div class="kv">
      <span class="label">Detail: </span>
      <span id="gesture-detail" class="value">--</span>
    </div>
  </div>

  <div class="card">
    <h2>Structure Scanner</h2>
    <div class="kv">
      <span class="label">State: </span>
      <span id="scanner-state" class="phase-badge phase-idle">IDLE</span>
    </div>
    <div class="kv">
      <span class="label">Last Scan: </span>
      <span id="scanner-result" class="value">--</span>
    </div>
  </div>

  <div class="card">
    <h2>Calibrator</h2>
    <div class="kv">
      <span class="label">State: </span>
      <span id="calibrator-state" class="phase-badge phase-idle">IDLE</span>
    </div>
  </div>

  <div class="card full">
    <h2>Node Health</h2>
    <div class="health-row">
      <div class="health-item">
        <span id="health-manipulator" class="status-dot unknown"></span>
        Manipulator
      </div>
      <div class="health-item">
        <span id="health-detector" class="status-dot unknown"></span>
        Detector
      </div>
      <div class="health-item">
        <span id="health-ik_solver" class="status-dot unknown"></span>
        IK Solver
      </div>
    </div>
  </div>

  <div class="card full">
    <h2>Event Log</h2>
    <div id="event-log"></div>
  </div>
</div>

<script>
const MAX_LOG = 100;
const BLOCK_COLORS = {
  1: 0xffff00,  // YELLOW
  2: 0x0000ff,  // BLUE
  3: 0x00ff00,  // GREEN
  4: 0xff0000,  // RED
};
const UNREACHABLE_DOT_CSS = {
  'ungrippable': '#e53935',
  'overhang':    '#fb8c00',
  'both':        '#8e24aa',
};

// ── Three.js scene setup ──────────────────────────────────────────
const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setClearColor(0xF5F0E8);
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 2, 0.1, 100);
camera.position.set(8, 6, 8);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);

// Inline OrbitControls (minimal)
let orbiting = false, orbitStart = {x:0, y:0};
let spherical = {theta: -Math.PI, phi: Math.PI/4, r: 12};

function updateCamera() {
  const st = Math.sin(spherical.phi);
  camera.position.set(
    spherical.r * st * Math.sin(spherical.theta) + orbitCenter.x,
    spherical.r * Math.cos(spherical.phi) + orbitCenter.y,
    spherical.r * st * Math.cos(spherical.theta) + orbitCenter.z
  );
  camera.lookAt(orbitCenter);
}
let orbitCenter = new THREE.Vector3(0, 0, 0);

canvas.addEventListener('mousedown', e => { orbiting = true; orbitStart = {x: e.clientX, y: e.clientY}; });
canvas.addEventListener('mousemove', e => {
  if (!orbiting) return;
  spherical.theta -= (e.clientX - orbitStart.x) * 0.008;
  spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1,
    spherical.phi + (e.clientY - orbitStart.y) * 0.008));
  orbitStart = {x: e.clientX, y: e.clientY};
  updateCamera();
});
canvas.addEventListener('mouseup', () => { orbiting = false; });
canvas.addEventListener('mouseleave', () => { orbiting = false; });
canvas.addEventListener('wheel', e => {
  spherical.r = Math.max(3, Math.min(30, spherical.r + e.deltaY * 0.01));
  updateCamera();
  e.preventDefault();
}, {passive: false});
// Touch support
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    orbiting = true;
    orbitStart = {x: e.touches[0].clientX, y: e.touches[0].clientY};
  }
});
canvas.addEventListener('touchmove', e => {
  if (!orbiting || e.touches.length !== 1) return;
  const t = e.touches[0];
  spherical.theta -= (t.clientX - orbitStart.x) * 0.008;
  spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1,
    spherical.phi + (t.clientY - orbitStart.y) * 0.008));
  orbitStart = {x: t.clientX, y: t.clientY};
  updateCamera();
  e.preventDefault();
}, {passive: false});
canvas.addEventListener('touchend', () => { orbiting = false; });

// ── Structure state ───────────────────────────────────────────────
let blockGrid = null;
let placedGrid = null;
const targetGroup = new THREE.Group();
const placedGroup = new THREE.Group();
const flashGroup = new THREE.Group();
scene.add(targetGroup);
scene.add(placedGroup);
scene.add(flashGroup);
let currentTarget = null; // {row, col, layer}
let showFullStructure = false;

const boxGeo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
const edgeGeo = new THREE.EdgesGeometry(boxGeo);
const MAX_GRID_ROWS = 6;
const MAX_GRID_COLS = 6;

function makeTextSprite(text) {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#8C8272';
  ctx.font = 'bold 48px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 32);
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.5, 0.5, 1);
  return sprite;
}

function addRefGrid(targetScene, rows, cols) {
  // Remove old grid + labels
  const oldGrid = targetScene.getObjectByName('ref-grid');
  if (oldGrid) targetScene.remove(oldGrid);
  const oldLabels = targetScene.getObjectByName('ref-labels');
  if (oldLabels) targetScene.remove(oldLabels);

  const gridY = 0.5;
  const pts = [];
  // Vertical lines (along Z) at each col boundary (negated X axis)
  for (let c = 0; c <= cols; c++) {
    const x = -c + 0.5;
    pts.push(x, gridY, -0.5,  x, gridY, rows - 0.5);
  }
  // Horizontal lines (along X) at each row boundary
  for (let r = -0.5; r <= rows - 0.5; r += 1) {
    pts.push(0.5, gridY, r,  -(cols - 0.5), gridY, r);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  const grid = new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ color: 0xBBB0A0 }));
  grid.name = 'ref-grid';
  targetScene.add(grid);

  // Labels outside the grid like axis ticks
  const labels = new THREE.Group();
  labels.name = 'ref-labels';
  // Col labels along negated X, outside at near Z edge
  for (let c = 0; c < cols; c++) {
    const s = makeTextSprite(String(c));
    s.position.set(-c, gridY, -1.0);
    labels.add(s);
  }
  // Row labels along Z, outside at positive X edge (right side = col 0 side)
  for (let r = 0; r < rows; r++) {
    const s = makeTextSprite(String(r));
    s.position.set(1.0, gridY, r);
    labels.add(s);
  }
  targetScene.add(labels);
}

function buildScene(block_grid, placed_grid) {
  blockGrid = block_grid;
  placedGrid = placed_grid;

  // Clear existing
  while (targetGroup.children.length) targetGroup.remove(targetGroup.children[0]);
  while (placedGroup.children.length) placedGroup.remove(placedGroup.children[0]);

  const rows = block_grid.length;
  const cols = block_grid[0].length;
  const layers = block_grid[0][0].length;

  // Center the structure
  orbitCenter.set(-cols / 2 + 0.5, layers / 2, rows / 2);

  let totalBlocks = 0;
  let placedCount = 0;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      for (let l = 0; l < layers; l++) {
        const val = block_grid[r][c][l];
        if (val === 0) continue;
        totalBlocks++;

        const pVal = placed_grid[r][c][l];

        if (showFullStructure) {
          // Show Full mode: all target blocks as solid cubes
          if (pVal !== 0) placedCount++;
          const color = BLOCK_COLORS[val] || 0xcccccc;
          const mat = new THREE.MeshLambertMaterial({
            color: color, transparent: true, opacity: 0.9
          });
          const mesh = new THREE.Mesh(boxGeo, mat);
          mesh.position.set(-c, l, r);
          placedGroup.add(mesh);
          const line = new THREE.LineSegments(edgeGeo,
            new THREE.LineBasicMaterial({ color: 0x999080 }));
          line.position.copy(mesh.position);
          placedGroup.add(line);
        } else if (pVal !== 0) {
          // Placed: solid cube — show actual placed color
          placedCount++;
          const pColor = BLOCK_COLORS[pVal] || 0xcccccc;
          const mat = new THREE.MeshLambertMaterial({
            color: pColor, transparent: true, opacity: 0.9
          });
          const mesh = new THREE.Mesh(boxGeo, mat);
          mesh.position.set(-c, l, r);
          placedGroup.add(mesh);

          // Dark edge outline
          const line = new THREE.LineSegments(edgeGeo,
            new THREE.LineBasicMaterial({ color: 0x999080 }));
          line.position.copy(mesh.position);
          placedGroup.add(line);
        } else {
          // Target (unplaced): wireframe — show desired color
          const tColor = BLOCK_COLORS[val] || 0xcccccc;
          const line = new THREE.LineSegments(edgeGeo,
            new THREE.LineBasicMaterial({ color: tColor, transparent: true, opacity: 0.55 }));
          line.position.set(-c, l, r);
          targetGroup.add(line);
        }
      }
    }
  }

  addRefGrid(scene, MAX_GRID_ROWS, MAX_GRID_COLS);
  updateFlash();
  updateCamera();
  document.getElementById('struct-info').textContent =
    placedCount + ' / ' + totalBlocks + ' blocks placed (' +
    rows + 'x' + cols + 'x' + layers + ' grid)';
}

function updatePlaced(placed_grid) {
  if (!blockGrid) return;
  buildScene(blockGrid, placed_grid);
}

function updateFlash() {
  while (flashGroup.children.length) flashGroup.remove(flashGroup.children[0]);
  if (showFullStructure) return;
  if (!currentTarget || !blockGrid) return;
  const {row, col, layer} = currentTarget;
  if (row >= blockGrid.length || col >= blockGrid[0].length || layer >= blockGrid[0][0].length) return;
  const val = blockGrid[row][col][layer];
  if (val === 0) return;
  // Don't flash if already placed
  if (placedGrid && placedGrid[row][col][layer] !== 0) return;
  const color = BLOCK_COLORS[val] || 0xcccccc;
  const mat = new THREE.MeshLambertMaterial({ color: color, transparent: true, opacity: 0.85 });
  const mesh = new THREE.Mesh(boxGeo, mat);
  mesh.position.set(-col, layer, row);
  flashGroup.add(mesh);
  const line = new THREE.LineSegments(edgeGeo,
    new THREE.LineBasicMaterial({ color: 0x999080 }));
  line.position.copy(mesh.position);
  flashGroup.add(line);
}

// Resize handler
function resizeRenderer() {
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  // Also resize unreachable canvas if visible
  const uRect = uCanvas.getBoundingClientRect();
  if (uRect.width > 0 && uRect.height > 0) {
    uRenderer.setSize(uRect.width, uRect.height, false);
    uCamera.aspect = uRect.width / uRect.height;
    uCamera.updateProjectionMatrix();
  }
}
window.addEventListener('resize', resizeRenderer);

// Render loop
function animate() {
  requestAnimationFrame(animate);
  if (flashGroup.children.length > 0) {
    const visible = Math.sin(Date.now() * 0.006) > 0;
    flashGroup.children.forEach(c => { c.visible = visible; });
  }
  renderer.render(scene, camera);
  uRenderer.render(uScene, uCamera);
}
// ── Unreachable blocks 3D view ──────────────────────────────────
const uCanvas = document.getElementById('unreachable-canvas');
const uRenderer = new THREE.WebGLRenderer({ canvas: uCanvas, antialias: true, alpha: false });
uRenderer.setClearColor(0xF5F0E8);
uRenderer.setPixelRatio(window.devicePixelRatio);

const uScene = new THREE.Scene();
const uCamera = new THREE.PerspectiveCamera(45, 2, 0.1, 100);
uCamera.position.set(8, 6, 8);
uCamera.lookAt(0, 0, 0);

uScene.add(new THREE.AmbientLight(0xffffff, 0.6));
const uDirLight = new THREE.DirectionalLight(0xffffff, 0.8);
uDirLight.position.set(5, 10, 7);
uScene.add(uDirLight);

const uGroup = new THREE.Group();
uScene.add(uGroup);

let uOrbiting = false, uOrbitStart = {x:0, y:0};
let uSpherical = {theta: -Math.PI, phi: Math.PI/4, r: 12};
let uOrbitCenter = new THREE.Vector3(0, 0, 0);

function uUpdateCamera() {
    const st = Math.sin(uSpherical.phi);
    uCamera.position.set(
        uSpherical.r * st * Math.sin(uSpherical.theta) + uOrbitCenter.x,
        uSpherical.r * Math.cos(uSpherical.phi) + uOrbitCenter.y,
        uSpherical.r * st * Math.cos(uSpherical.theta) + uOrbitCenter.z
    );
    uCamera.lookAt(uOrbitCenter);
}

uCanvas.addEventListener('mousedown', e => { uOrbiting = true; uOrbitStart = {x: e.clientX, y: e.clientY}; });
uCanvas.addEventListener('mousemove', e => {
    if (!uOrbiting) return;
    uSpherical.theta -= (e.clientX - uOrbitStart.x) * 0.008;
    uSpherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1,
        uSpherical.phi + (e.clientY - uOrbitStart.y) * 0.008));
    uOrbitStart = {x: e.clientX, y: e.clientY};
    uUpdateCamera();
});
uCanvas.addEventListener('mouseup', () => { uOrbiting = false; });
uCanvas.addEventListener('mouseleave', () => { uOrbiting = false; });
uCanvas.addEventListener('wheel', e => {
    uSpherical.r = Math.max(3, Math.min(30, uSpherical.r + e.deltaY * 0.01));
    uUpdateCamera();
    e.preventDefault();
}, {passive: false});
uCanvas.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
        uOrbiting = true;
        uOrbitStart = {x: e.touches[0].clientX, y: e.touches[0].clientY};
    }
});
uCanvas.addEventListener('touchmove', e => {
    if (!uOrbiting || e.touches.length !== 1) return;
    const t = e.touches[0];
    uSpherical.theta -= (t.clientX - uOrbitStart.x) * 0.008;
    uSpherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1,
        uSpherical.phi + (t.clientY - uOrbitStart.y) * 0.008));
    uOrbitStart = {x: t.clientX, y: t.clientY};
    uUpdateCamera();
    e.preventDefault();
}, {passive: false});
uCanvas.addEventListener('touchend', () => { uOrbiting = false; });

// ── Start rendering (after both canvases are set up) ──
resizeRenderer();
updateCamera();
uUpdateCamera();
animate();

function updateUnreachableView(blocks) {
    // Clear existing
    while (uGroup.children.length) uGroup.remove(uGroup.children[0]);

    if (blocks.length === 0) return;

    // Center orbit on structure orbit center if available, otherwise compute from blocks
    if (orbitCenter.x !== 0 || orbitCenter.y !== 0 || orbitCenter.z !== 0) {
        uOrbitCenter.copy(orbitCenter);
    } else {
        let sx = 0, sy = 0, sz = 0;
        blocks.forEach(b => { sx += -b.grid_cell.col; sy += b.grid_cell.layer; sz += b.grid_cell.row; });
        uOrbitCenter.set(sx / blocks.length, sy / blocks.length, sz / blocks.length);
    }
    uUpdateCamera();

    blocks.forEach(b => {
        const color = BLOCK_COLORS[b.block_type] || 0xcccccc;
        const mat = new THREE.MeshLambertMaterial({ color: color, transparent: true, opacity: 0.9 });
        const mesh = new THREE.Mesh(boxGeo, mat);
        mesh.position.set(-b.grid_cell.col, b.grid_cell.layer, b.grid_cell.row);
        uGroup.add(mesh);

        const line = new THREE.LineSegments(edgeGeo,
            new THREE.LineBasicMaterial({ color: 0x999080 }));
        line.position.copy(mesh.position);
        uGroup.add(line);
    });

    // Add reference grid using structure dimensions if available
    if (blockGrid) {
        addRefGrid(uScene, MAX_GRID_ROWS, MAX_GRID_COLS);
    }
}

function updateUnreachablePanel(blocks) {
    const list = document.getElementById('unreachable-list');
    list.innerHTML = '';
    blocks.forEach(b => {
        const dotColor = UNREACHABLE_DOT_CSS[b.reason] || '#888';
        const entry = document.createElement('div');
        entry.className = 'unreachable-entry';
        entry.innerHTML =
            '<span class="reason-dot" style="background:' + dotColor + '"></span>' +
            '(' + b.grid_cell.row + ', ' + b.grid_cell.col + ', ' + b.grid_cell.layer + ') ' +
            b.reason + ' — ' + b.nod_attempts + ' nods';
        list.appendChild(entry);
    });
}

function showUnreachable(hasBlocks) {
    if (!hasBlocks) {
        document.getElementById('unreachable-list').innerHTML =
            '<div style="color:#8C8272;font-size:0.8rem;">No unreachable misplaced blocks</div>';
    }
}

// ── Event handling ────────────────────────────────────────────────
function fmt(t) {
  return typeof t === 'number' ? t.toFixed(1) + 's' : '--';
}

function handleEvent(ev) {
  // Structure events
  if (ev.type === 'structure') {
    buildScene(ev.data.block_grid, ev.data.placed_grid);
    return;
  }
  if (ev.type === 'structure_update') {
    updatePlaced(ev.data.placed_grid);
    return;
  }

  if (ev.type === 'unreachable_blocks') {
    const blocks = ev.data.blocks;
    showUnreachable(blocks.length > 0);
    updateUnreachableView(blocks);
    updateUnreachablePanel(blocks);
    return;
  }

  // State change events
  if (ev.type === 'state_change') {
    if (ev.source === 'pipeline') {
      const phase = ev.data.phase || 'idle';
      const el = document.getElementById('pipeline-phase');
      el.textContent = phase.toUpperCase();
      el.className = 'phase-badge phase-' + (phase || 'idle');
      document.getElementById('pipeline-target').textContent =
        ev.data.target_cell || '--';
      document.getElementById('pipeline-block').textContent =
        ev.data.block_type || '--';
      document.getElementById('pipeline-detail').textContent =
        ev.data.detail || '--';
      // Track current target for flashing
      if (phase === 'idle') {
        currentTarget = null;
      } else if (ev.data.target_cell && ev.data.target_cell !== '--') {
        const m = ev.data.target_cell.match(/\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/);
        if (m) currentTarget = {row:+m[1], col:+m[2], layer:+m[3]};
      }
      updateFlash();
    } else if (ev.source === 'scanner') {
      const el = document.getElementById('scanner-state');
      el.textContent = (ev.data.state || 'idle').toUpperCase();
      el.className = 'phase-badge phase-' + (ev.data.state === 'idle' ? 'idle' : 'scan');
    } else if (ev.source === 'calibrator') {
      const el = document.getElementById('calibrator-state');
      el.textContent = (ev.data.state || 'idle').toUpperCase();
      el.className = 'phase-badge phase-' + (ev.data.state === 'idle' ? 'idle' : 'scan');
    } else if (ev.source === 'gestures') {
      const el = document.getElementById('gesture-state');
      const s = ev.data.state || 'inactive';
      el.textContent = s.toUpperCase();
      el.className = 'phase-badge phase-' + (s === 'inactive' ? 'idle' : 'scan');
      document.getElementById('gesture-detail').textContent = ev.data.detail || '--';
    }
  } else if (ev.type === 'health') {
    for (const [node, online] of Object.entries(ev.data)) {
      const dot = document.getElementById('health-' + node);
      if (dot) dot.className = 'status-dot ' + (online ? 'online' : 'offline');
    }
  } else if (ev.type === 'scan_result') {
    document.getElementById('scanner-result').textContent =
      (ev.data.placed_count || 0) + ' placed, ' +
      (ev.data.unassigned_count || 0) + ' unassigned, target=' +
      (ev.data.target_cell || '--');
  } else if (ev.type === 'detection') {
    document.getElementById('pipeline-detail').textContent =
      'grip: ' + (ev.data.grip_quality || '?') +
      ' (diag=' + (ev.data.diagonal_score || 0).toFixed(2) +
      ', h=' + (ev.data.height_score || 0).toFixed(1) + ')';
  }

  // Add to event log (skip structure/structure_update — too verbose)
  if (ev.type === 'structure' || ev.type === 'structure_update') return;
  const log = document.getElementById('event-log');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML =
    '<span class="log-time">' + fmt(ev.timestamp) + '</span> ' +
    '<span class="log-source">[' + (ev.source || '?') + ']</span> ' +
    '<span class="log-type">' + (ev.type || '?') + '</span> ' +
    JSON.stringify(ev.data || {});
  log.insertBefore(entry, log.firstChild);
  while (log.children.length > MAX_LOG) log.removeChild(log.lastChild);
}

// Hydrate from snapshot
fetch('/snapshot')
  .then(r => r.json())
  .then(events => events.forEach(handleEvent))
  .catch(() => {});

// SSE connection
function connect() {
  const es = new EventSource('/events/stream');
  const statusEl = document.getElementById('conn-status');

  es.onopen = () => {
    statusEl.innerHTML = '<span class="status-dot online"></span> connected';
  };
  es.onmessage = (msg) => {
    try { handleEvent(JSON.parse(msg.data)); } catch(e) {}
  };
  es.onerror = () => {
    statusEl.innerHTML = '<span class="status-dot offline"></span> disconnected';
    es.close();
    setTimeout(connect, 2000);
  };
}
connect();

document.getElementById('toggle-full-structure').addEventListener('change', function() {
  showFullStructure = this.checked;
  if (blockGrid && placedGrid) buildScene(blockGrid, placedGrid);
});

// ── Assembly selector ────────────────────────────────────────────
const COLOR_TO_INT = { yellow: 1, blue: 2, green: 3, red: 4, orange: 5 };

function assemblyToBlockGrid(data) {
  const gs = data.metadata.gridSize;
  const rows = gs.width, cols = gs.length, layers = gs.height;
  // Create 3D grid [row][col][layer] filled with 0
  const grid = [];
  for (let r = 0; r < rows; r++) {
    grid[r] = [];
    for (let c = 0; c < cols; c++) {
      grid[r][c] = new Array(layers).fill(0);
    }
  }
  for (const block of data.blocks) {
    const col = block.position.x;
    const row = block.position.y;
    const layer = block.position.z - 1; // z=1 is ground, map to index 0
    if (row >= 0 && row < rows && col >= 0 && col < cols && layer >= 0 && layer < layers) {
      grid[row][col][layer] = COLOR_TO_INT[block.color] || 1;
    }
  }
  return grid;
}

// Populate dropdown
fetch('/assemblies')
  .then(r => r.json())
  .then(files => {
    const sel = document.getElementById('assembly-select');
    files.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f.replace('.json', '');
      sel.appendChild(opt);
    });
  })
  .catch(() => {});

// On selection, load and preview
document.getElementById('assembly-select').addEventListener('change', function() {
  const filename = this.value;
  if (!filename) return;
  fetch('/assemblies/' + encodeURIComponent(filename))
    .then(r => r.json())
    .then(data => {
      if (data.error) return;
      const grid = assemblyToBlockGrid(data);
      const emptyGrid = JSON.parse(JSON.stringify(grid)); // all zeros for placed
      for (let r = 0; r < emptyGrid.length; r++)
        for (let c = 0; c < emptyGrid[r].length; c++)
          emptyGrid[r][c] = new Array(emptyGrid[r][c].length).fill(0);
      showFullStructure = true;
      document.getElementById('toggle-full-structure').checked = true;
      buildScene(grid, emptyGrid);
    })
    .catch(() => {});
});

// Set Structure button — client-side only for now
document.getElementById('set-structure-btn').addEventListener('click', function() {
  const sel = document.getElementById('assembly-select');
  console.log('Set Structure clicked:', sel.value || '(none selected)');
});
</script>
</body>
</html>

"""


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
