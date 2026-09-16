import os
from isaacsim import SimulationApp

# 1. Start the Simulator
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.cloner import Cloner
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import SingleXFormPrim

# 2. Settings
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CUBES = 15
CUBE_SIZE = 0.034 # 1.34 in = 0.034 cm
SEARCH_AREA_BOUNDS = [(0.2, 0.2), (0.6, 0.5)]

# --- PATH CONFIGURATION (MATCHING YOUR SCREENSHOT) ---
ROBOT_PRIM_PATH = "/World/fivedof"
USD_PATH = os.path.join(_SCRIPT_DIR, "fivedof", "fivedof.usd")
EE_PRIM_PATH = "/World/fivedof/tip" # <--- derived from your screenshot

# 3. Initialize World
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# --- Setup Robot ---
# Load the USD
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM_PATH)

# Wrap the robot (for joint control)
robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="fivedof_robot"))

# Create a specific wrapper just for the tip (for sensing)
ee_prim = SingleXFormPrim(prim_path=EE_PRIM_PATH)

# --- Setup Cubes using Cloner ---
template_path = "/World/Cube_Template"
DynamicCuboid(prim_path=template_path, name="cube_template", size=CUBE_SIZE, color=np.array([1, 0, 0]))

cloner = Cloner()
cloner.define_base_env("/World/Envs")
target_paths = cloner.generate_paths("/World/Cube", NUM_CUBES) 
cloner.clone(source_prim_path=template_path, prim_paths=target_paths)

# Wrap cubes
cubes = []
for path in target_paths:
    cubes.append(world.scene.add(DynamicCuboid(prim_path=path, name=path.split("/")[-1])))

# --- Helper: Random Position Generator ---
def get_random_positions(n, bounds, min_dist):
    positions = []
    max_attempts = 100
    for _ in range(n):
        for _ in range(max_attempts):
            x = np.random.uniform(bounds[0][0], bounds[1][0])
            y = np.random.uniform(bounds[0][1], bounds[1][1])
            z = CUBE_SIZE / 2.0 
            candidate = np.array([x, y, z])
            if all(np.linalg.norm(candidate - p) >= min_dist for p in positions):
                positions.append(candidate)
                break
    return np.array(positions)

# --- Helper: Reset Logic ---
def reset_environment():
    # Randomize Cubes
    valid_positions = get_random_positions(len(cubes), SEARCH_AREA_BOUNDS, min_dist=CUBE_SIZE * 1.5)
    for i, cube in enumerate(cubes):
        if i < len(valid_positions):
            cube.set_world_pose(position=valid_positions[i])
            cube.set_visibility(True)
        else:
            cube.set_visibility(False)

# --- Helper: Noisy Perception Model ---
def get_noisy_observation(robot_ee_pos, target_pos):
    true_dist = np.linalg.norm(robot_ee_pos - target_pos)
    base_noise = 0.0001 
    distance_factor = 0.02 
    current_std_dev = base_noise + (distance_factor * true_dist)
    noise = np.random.normal(0, current_std_dev, size=3)
    return target_pos + noise

# 4. Run Simulation
world.reset()
reset_environment()

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing():
        # --- 1. Get Robot End Effector Position ---
        # We use the XFormPrim wrapper we made earlier. 
        # This works even if 'tip' has no mass/physics.
        robot_ee_pos, _ = ee_prim.get_world_pose()
        
        # --- 2. Get Target Cube (e.g., the first one) ---
        target_cube_pos, _ = cubes[0].get_world_pose()
        
        # --- 3. Calculate Noisy Target ---
        noisy_target = get_noisy_observation(robot_ee_pos, target_cube_pos)
        
        # Optional: Print to verify it's working
        # print(f"EE: {robot_ee_pos} | Target: {target_cube_pos}")

simulation_app.close()