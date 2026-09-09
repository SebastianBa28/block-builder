"""
Thin composition shell over VoxelStore for 3D world mapping.

WorldMap delegates ALL probabilistic operations (log-odds updates, decay,
frustum miss, confidence filtering, pruning) to the underlying VoxelStore.
It adds:

- Range filtering in camera frame before delegation to VoxelStore.
- Open3D PointCloud construction from confidence-filtered voxel data.
- PointCloud2 message building with packed RGB (lazy ROS imports).
- Compressed .npz persistence (save/load round-trips full state + config).
- Publish throttle via WorldMapConfig.publish_every_n_scans.

The class is node-agnostic: no ROS dependencies at module level.  The
to_pointcloud2_msg() method lazily imports sensor_msgs types.
"""

import numpy as np
import open3d as o3d
from typing import Optional

from legobuilder.config import WorldMapConfig
from legobuilder.vision.voxel_store import VoxelStore, HeightResult


class WorldMap:
    """Probabilistic 3D world map backed by a sparse VoxelStore.

    Maintains a VoxelStore internally via composition.  All storage,
    Bayesian updates, decay, frustum miss, and pruning are handled by
    the store.  WorldMap adds range filtering, Open3D / PointCloud2
    construction, persistence, and convenience properties.

    Attributes
    ----------
    config : WorldMapConfig
        Map configuration (voxel resolution, Bayesian parameters,
        workspace bounds, Kalman settings).
    logger
        Optional ROS-compatible logger (currently disabled).
    """

    def __init__(
        self,
        config: Optional[WorldMapConfig] = None,
        logger=None,
    ) -> None:
        self.config = config or WorldMapConfig()
        self._store = VoxelStore(self.config, logger)
        # self.logger = logger
        self.logger = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        """Number of scans integrated so far."""
        return self._store.scan_index

    @property
    def has_data(self) -> bool:
        """Whether at least one scan has been integrated."""
        return self._store.scan_index > 0

    # ── Integration ────────────────────────────────────────────────────

    def integrate(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        T_world_camera: np.ndarray,
    ) -> None:
        """Integrate a point cloud frame into the map.

        Applies range filtering in camera frame before delegating to
        VoxelStore for the full probabilistic pipeline.

        Arguments
        ---------
        points : np.ndarray
            (N, 3) float array of XYZ in camera frame.
        colors : np.ndarray
            (N, 3) float or uint8 array of RGB per point.
        T_world_camera : np.ndarray
            (4, 4) camera-to-world homogeneous transform.
        """
        # Range filter in camera frame (Z = depth).
        z = points[:, 2]
        mask = (z >= self.config.range_min) & (z <= self.config.range_max)

        if self.logger:
            self.logger.info(
                f"WorldMap.integrate: {len(points)} pts in, "
                f"z range [{z.min():.3f}, {z.max():.3f}], "
                f"range filter [{self.config.range_min}, {self.config.range_max}] "
                f"→ {mask.sum()} pts survive range filter"
            )

        points = points[mask]
        colors = colors[mask]

        if len(points) == 0:
            # Still run decay / frustum miss / pruning via empty integrate.
            self._store.integrate(
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float32),
                T_world_camera,
            )
            return

        n_new = self._store.integrate(points, colors, T_world_camera)

        if self.logger:
            self.logger.info(
                f"WorldMap: scan {self._store.scan_index}: "
                f"{len(points)} pts → {n_new} new voxels, "
                f"{self._store.voxel_count} total voxels"
            )

    def integrate_world_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Integrate points already in world frame (no camera transform).

        Bypasses camera-frame range filtering since overhead-derived
        points are already in world coordinates.  VoxelStore still
        applies workspace clipping.

        Arguments
        ---------
        points : np.ndarray
            (N, 3) float array of XYZ in world frame.
        colors : np.ndarray
            (N, 3) float or uint8 array of RGB per point.
        """
        if len(points) == 0:
            return

        n_new = self._store.integrate(
            points, colors, np.eye(4, dtype=np.float64))

        if self.logger:
            self.logger.debug(
                f"WorldMap.integrate_world_points: "
                f"{len(points)} pts → {n_new} new voxels, "
                f"{self._store.voxel_count} total"
            )

    # ── Internal Helpers ────────────────────────────────────────────────

    def _get_filtered_data(
        self,
        confidence_min: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Return confidence-filtered voxel data from the store.

        Arguments
        ---------
        confidence_min : float, optional
            Log-odds threshold.  None uses
            WorldMapConfig.confidence_threshold.

        Returns
        -------
        dict[str, np.ndarray]
            Dict with keys 'keys', 'centers', 'log_odds',
            'obs_count', 'last_scan', 'colors'.  Arrays are
            masked to only include voxels above the confidence threshold.
        """
        data = self._store.get_all_data()

        if len(data['log_odds']) == 0:
            return data

        thresh = (
            confidence_min
            if confidence_min is not None
            else self.config.confidence_threshold
        )
        mask = data['log_odds'] > thresh

        return {k: v[mask] for k, v in data.items()}

    # ── Point Cloud Extraction ────────────────────────────────────────

    def get_point_cloud(
        self,
        confidence_min: Optional[float] = None,
    ) -> o3d.geometry.PointCloud:
        """Build an Open3D PointCloud from confidence-filtered voxel data.

        Arguments
        ---------
        confidence_min : float, optional
            Log-odds threshold.  None uses config value.

        Returns
        -------
        o3d.geometry.PointCloud
            Open3D PointCloud with voxel centres as points and stored RGB
            colours.  Empty PointCloud if no data passes the filter.
        """
        data = self._get_filtered_data(confidence_min)
        pcd = o3d.geometry.PointCloud()

        positions = data['positions']
        if len(positions) == 0:
            return pcd

        pcd.points = o3d.utility.Vector3dVector(positions)
        colors = data['colors']
        if len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(
                colors.astype(np.float64)
            )

        return pcd

    def get_numpy_points(
        self,
        confidence_min: Optional[float] = None,
    ) -> np.ndarray:
        """Return Nx3 float64 array of confidence-filtered voxel centres.

        Arguments
        ---------
        confidence_min : float, optional
            Log-odds threshold.  None uses config value.

        Returns
        -------
        np.ndarray
            (N, 3) float64 array, or (0, 3) if no data.
        """
        data = self._get_filtered_data(confidence_min)
        positions = data['positions']
        if len(positions) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return positions

    def get_numpy_points_colors(
        self,
        confidence_min: Optional[float] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return confidence-filtered points and colours as numpy arrays.

        Arguments
        ---------
        confidence_min : float, optional
            Log-odds threshold.  None uses config value.

        Returns
        -------
        tuple[np.ndarray, np.ndarray or None]
            Tuple (points, colors) where *points* is (N, 3) float64
            and *colors* is (N, 3) float32 or None if no data.
        """
        data = self._get_filtered_data(confidence_min)
        positions = data['positions']
        if len(positions) == 0:
            return np.empty((0, 3), dtype=np.float64), None
        colors = data['colors']
        if len(colors) == 0:
            return positions, None
        return positions, colors

    # ── Spatial Queries ────────────────────────────────────────────────

    def query_height(
        self,
        x: float,
        y: float,
        radius: float = 0.02,
        obs_min: int = 2,
    ) -> Optional[HeightResult]:
        """Query maximum height at (x, y) via VoxelStore.

        Arguments
        ---------
        x : float
            X position in world frame.
        y : float
            Y position in world frame.
        radius : float
            XY search radius (metres).
        obs_min : int
            Minimum observation count (inclusive).

        Returns
        -------
        HeightResult or None
            HeightResult or None.
        """
        return self._store.query_height(x, y, radius, obs_min=obs_min)

    def query_region(
        self,
        xy_center: np.ndarray,
        radius: float,
        z_min: float = -float('inf'),
        z_max: float = float('inf'),
        confidence_min: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Query voxels in a cylindrical region via VoxelStore.

        Arguments
        ---------
        xy_center : np.ndarray
            Length-2 array [x, y] in world frame.
        radius : float
            XY search radius (metres).
        z_min : float
            Minimum Z (inclusive); meters.
        z_max : float
            Maximum Z (inclusive); meters.
        confidence_min : float, optional
            Log-odds threshold.

        Returns
        -------
        dict[str, np.ndarray]
            Dict of numpy arrays (centres, colours, log_odds, obs_count, keys),
            representing the voxels.
        """
        return self._store.query_region(
            xy_center, radius, z_min=z_min, z_max=z_max,
            confidence_min=confidence_min,
        )

    def clear_region(self, center: np.ndarray, radius: float) -> int:
        """Remove all voxels within a sphere via VoxelStore.

        Arguments
        ---------
        center : np.ndarray
            Length-3 array [x, y, z] in world frame.
        radius : float
            Spherical clearing radius (metres).

        Returns
        -------
        int
            Number of voxels removed.
        """
        return self._store.clear_region(center, radius)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the map and reinitialise the VoxelStore."""
        self._store.reset()
        if self.logger:
            self.logger.info("WorldMap: reset")

    # ── Publishing Helpers ──────────────────────────────────────────────

    def should_publish(self) -> bool:
        """Check whether to publish a PointCloud2 this scan.

        Returns True when scan_index % publish_every_n_scans == 0.
        """
        return self._store.scan_index % self.config.publish_every_n_scans == 0

    def to_pointcloud2_msg(
        self,
        stamp,
        frame_id: str = 'world',
        confidence_min: Optional[float] = None,
    ):
        """Build a sensor_msgs/PointCloud2 from confidence-filtered data.

        Uses lazy ROS imports so that WorldMap can be instantiated and
        tested without a ROS environment.

        Arguments
        ---------
        stamp
            ROS Time message for the header stamp.
        frame_id : str
            TF frame for the header (default 'world').
        confidence_min : float, optional
            Log-odds threshold.  None uses config value.

        Returns
        -------
        PointCloud2 or None
            PointCloud2 message, or None if no data passes filter.
        """
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header

        data = self._get_filtered_data(confidence_min)
        positions = data['positions']
        if len(positions) == 0:
            # Debug: show raw store state before filtering
            raw = self._store.get_all_data()
            n_raw = len(raw['log_odds'])
            lo_stats = ""
            if n_raw > 0:
                lo_stats = (f", log_odds range [{raw['log_odds'].min():.3f}, "
                            f"{raw['log_odds'].max():.3f}]")
            if self.logger:
                self.logger.info(
                    f"WorldMap: no points to publish "
                    f"(confidence_min={confidence_min}, "
                    f"thresh={self.config.confidence_threshold}, "
                    f"raw_voxels={n_raw}{lo_stats}), skipping publish"
                )
            return None

        colors = data['colors']
        has_color = len(colors) > 0

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(positions)

        if has_color:
            msg.fields = [
                PointField(name='x', offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12,
                           datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 16  # 3 * float32 + 1 packed rgb float32

            # Pack [0,1] colours into bit-packed uint32 viewed as float32.
            rgb_clipped = np.clip(colors, 0.0, 1.0)
            rgb_uint8 = (rgb_clipped * 255).astype(np.uint8)
            rgb_packed = (
                rgb_uint8[:, 0].astype(np.uint32) << 16
                | rgb_uint8[:, 1].astype(np.uint32) << 8
                | rgb_uint8[:, 2].astype(np.uint32)
            )
            rgb_float = rgb_packed.view(np.float32)

            xyz_f32 = positions.astype(np.float32)
            packed = np.column_stack((xyz_f32, rgb_float))
            msg.data = packed.tobytes()
        else:
            msg.fields = [
                PointField(name='x', offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,
                           datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 12  # 3 * float32
            msg.data = positions.astype(np.float32).tobytes()

        msg.is_bigendian = False
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        return msg

    def get_table_height(self) -> float:
        return self._store.table_height

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist all voxel state and config to a compressed .npz file.

        Arguments
        ---------
        path : str
            File path (should end in .npz).
        """
        data = self._store.get_all_data()

        save_dict: dict[str, np.ndarray] = {
            'keys': data['keys'],
            'log_odds': data['log_odds'],
            'obs_count': data['obs_count'],
            'last_scan': data['last_scan'],
            'colors': data['colors'],
            'scan_index': np.array(self._store.scan_index),
        }

        # Persist all config fields with config_ prefix.
        cfg = self.config
        save_dict['config_voxel_resolution'] = np.array(cfg.voxel_resolution)
        save_dict['config_range_min'] = np.array(cfg.range_min)
        save_dict['config_range_max'] = np.array(cfg.range_max)
        save_dict['config_log_odds_prior'] = np.array(cfg.log_odds_prior)
        save_dict['config_log_odds_hit'] = np.array(cfg.log_odds_hit)
        save_dict['config_log_odds_miss'] = np.array(cfg.log_odds_miss)
        save_dict['config_log_odds_max'] = np.array(cfg.log_odds_max)
        save_dict['config_log_odds_min'] = np.array(cfg.log_odds_min)
        save_dict['config_obs_count_max'] = np.array(cfg.obs_count_max)
        save_dict['config_decay_after_scans'] = np.array(cfg.decay_after_scans)
        save_dict['config_decay_factor'] = np.array(cfg.decay_factor)
        save_dict['config_prune_interval'] = np.array(cfg.prune_interval)
        save_dict['config_confidence_threshold'] = np.array(
            cfg.confidence_threshold
        )
        save_dict['config_camera_hfov'] = np.array(cfg.camera_hfov)
        save_dict['config_camera_vfov'] = np.array(cfg.camera_vfov)
        save_dict['config_publish_every_n_scans'] = np.array(
            cfg.publish_every_n_scans
        )
        save_dict['config_workspace_base_xy'] = cfg.workspace_base_xy
        save_dict['config_workspace_inner_radius'] = np.array(cfg.workspace_inner_radius)
        save_dict['config_workspace_outer_radius'] = np.array(cfg.workspace_outer_radius)
        save_dict['config_workspace_z_min'] = np.array(cfg.workspace_z_min)
        save_dict['config_workspace_z_max'] = np.array(cfg.workspace_z_max)

        # Kalman state
        save_dict['config_kalman_enabled'] = np.array(cfg.kalman_enabled)
        save_dict['config_kalman_process_noise'] = np.array(cfg.kalman_process_noise)
        save_dict['config_kalman_measurement_noise'] = np.array(cfg.kalman_measurement_noise)

        if cfg.kalman_enabled:
            kf_data = self._store.get_kalman_data()
            if kf_data is not None:
                save_dict['kf_mean'] = kf_data['kf_mean']
                save_dict['kf_var'] = kf_data['kf_var']
                save_dict['kf_n'] = kf_data['kf_n']

        np.savez_compressed(path, **save_dict)

        if self.logger:
            self.logger.info(
                f"WorldMap: saved {self._store.voxel_count} voxels to {path}"
            )

    @staticmethod
    def load(path: str) -> 'WorldMap':
        """Load a WorldMap from a compressed .npz file.

        Static factory that reconstructs the WorldMapConfig and VoxelStore
        state from the persisted arrays.

        Arguments
        ---------
        path : str
            File path to load (should end in .npz).

        Returns
        -------
        WorldMap
            A new WorldMap with restored config and voxel state.
        """
        data = np.load(path, allow_pickle=False)

        # Reconstruct config from config_* keys.
        config = WorldMapConfig(
            voxel_resolution=float(data['config_voxel_resolution']),
            range_min=float(data['config_range_min']),
            range_max=float(data['config_range_max']),
            log_odds_prior=float(data['config_log_odds_prior']),
            log_odds_hit=float(data['config_log_odds_hit']),
            log_odds_miss=float(data['config_log_odds_miss']),
            log_odds_max=float(data['config_log_odds_max']),
            log_odds_min=float(data['config_log_odds_min']),
            obs_count_max=int(data['config_obs_count_max']),
            decay_after_scans=int(data['config_decay_after_scans']),
            decay_factor=float(data['config_decay_factor']),
            prune_interval=int(data['config_prune_interval']),
            confidence_threshold=float(data['config_confidence_threshold']),
            camera_hfov=float(data['config_camera_hfov']),
            camera_vfov=float(data['config_camera_vfov']),
            publish_every_n_scans=int(data['config_publish_every_n_scans']),
            workspace_base_xy=data['config_workspace_base_xy'].copy(),
            workspace_inner_radius=float(data['config_workspace_inner_radius']),
            workspace_outer_radius=float(data['config_workspace_outer_radius']),
            workspace_z_min=float(data['config_workspace_z_min']),
            workspace_z_max=float(data['config_workspace_z_max']),
            kalman_enabled=bool(data.get('config_kalman_enabled', np.array(False))),
            kalman_process_noise=float(data.get('config_kalman_process_noise', np.array(1e-8))),
            kalman_measurement_noise=float(data.get('config_kalman_measurement_noise', np.array(9e-6))),
        )

        wm = WorldMap(config)

        # Build Kalman kwargs for _restore_from_arrays (optional).
        kf_kwargs = {}
        if config.kalman_enabled and 'kf_mean' in data:
            kf_kwargs = {
                'kf_mean': data['kf_mean'],
                'kf_var': data['kf_var'],
                'kf_n': data['kf_n'],
            }

        wm._store._restore_from_arrays(
            keys=data['keys'],
            log_odds=data['log_odds'],
            obs_count=data['obs_count'],
            last_scan=data['last_scan'],
            colors=data['colors'],
            scan_index=int(data['scan_index']),
            **kf_kwargs,
        )

        return wm
