# Uncalled Functions Catalog

Functions with zero call sites across the legobuilder codebase.
Generated during Phase 2 refactoring for manual review.

**These functions are NOT deleted** -- they may be useful for future development,
interactive debugging, or external consumers. Review each and decide whether to
keep, delete, or mark as part of the public API.

**Exclusions applied:** ROS callbacks, entry points, dunder methods, matplotlib
handlers, `if __name__ == '__main__'` blocks, visualize/plotting utilities,
FastAPI endpoints, private methods called within their own class.


## kinematics/transform_utils.py

These are pure math utility functions. Many exist as a complete API surface
(e.g. every interpolation variant, every ROS message converter) even though
the current robot code only uses a subset.

| Function | Description | Likely reason uncalled |
|----------|-------------|----------------------|
| `nx()` | Unit vector along x-axis | Only `ny()` is used in the codebase |
| `nz()` | Unit vector along z-axis | Only `ny()` is used in the codebase |
| `vzero()` | Zero 3-vector | `pzero()` is used instead (identical output) |
| `vxyz(x, y, z)` | 3D vector from components | `pxyz()` is used instead (identical) |
| `pmid(p0, p1)` | Midpoint between two positions | No midpoint computation needed currently |
| `Rmid(R0, R1)` | Rotation midpoint | No rotation midpoint needed currently |
| `pinter(p0, p1, s)` | Linear position interpolation | Trajectory utils used instead |
| `vinter(p0, p1, sdot)` | Velocity for linear interpolation | Trajectory utils used instead |
| `winter(R0, R1, sdot)` | Angular velocity for rotation interpolation | No rotation interpolation in current pipeline |
| `eR(Rd, R)` | Rotation error vector | No orientation IK in current solver |
| `quateye()` | Identity quaternion | No quaternion math needed directly |
| `quat_from_xyzw(x, y, z, w)` | Quaternion from components | No quaternion construction needed directly |
| `T_from_Pose(pose)` | Transform from geometry_msgs/Pose | No inbound Pose messages currently |
| `T_from_Transform(transform)` | Transform from geometry_msgs/Transform | No inbound Transform messages currently |
| `Pose_from_T(T)` | geometry_msgs/Pose from transform | Pose_from_Rp is used directly instead |
| `Transform_from_T(T)` | geometry_msgs/Transform from transform | Transform_from_Rp is used directly instead |
| `Twist_from_vw(v, w)` | geometry_msgs/Twist from velocities | No Twist publishing in current nodes |


## kinematics/trajectory_utils.py

| Function | Description | Likely reason uncalled |
|----------|-------------|----------------------|
| `interpolate(t, T, p0, pf)` | Linear interpolation over duration T | Quintic spline (`goto5`) is used for all trajectories |
| `goto(t, T, p0, pf)` | Cubic spline with zero endpoint velocity | Quintic spline (`goto5`) is used instead |
| `spline(t, T, p0, pf, v0, vf)` | Cubic spline with specified velocities | Quintic spline (`spline5`) is used instead |
| `solve_line_ellipsoid_interception(...)` | Line-ellipsoid intersection for ball catching | Ball-catching feature not implemented |


## kinematics/block_manipulator.py

| Function | Description | Likely reason uncalled |
|----------|-------------|----------------------|
| `ManipulatorState.open_gripper()` | Set gripper joint to open position | Gripper opened via direct q manipulation in manipulator.py, not through ManipulatorState |


## vision/detection.py

| Function | Description | Likely reason uncalled |
|----------|-------------|----------------------|
| `get_ewma_weights(T, half_life)` | Compute EWMA weights for temporal filtering | Defined but never integrated into the detection pipeline |


## vision/world_map.py

| Function | Description | Likely reason uncalled |
|----------|-------------|----------------------|
| `WorldMap.get_point_cloud(...)` | Build Open3D PointCloud from voxel data | Replaced by `to_pointcloud2_msg` for ROS publishing |
| `WorldMap.get_numpy_points(...)` | Nx3 array of voxel centres | Data accessed through `query_region` instead |
| `WorldMap.get_numpy_points_colors(...)` | Voxel centres and colours as arrays | Data accessed through `query_region` instead |


## vision/voxel_store.py

| Function | Description | Likely reason uncalled |
|----------|-------------|----------------------|
| `VoxelStore.memory_bytes` | Estimate memory usage in bytes | Diagnostic property, never queried |
| `VoxelStore.get_voxel(key)` | Per-voxel state lookup | Bulk operations (`get_all_data`) used instead |
| `VoxelStore.voxel_center(key)` | World position of a voxel centre | Not needed by current consumers |
