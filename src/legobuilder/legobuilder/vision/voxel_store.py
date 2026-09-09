"""
Sparse dict-based voxel grid for probabilistic world mapping.

Stores per-voxel occupancy (log-odds), observation count, last-seen scan index,
RGB color, and optional Kalman refinement state in a single structured numpy
array (VOXEL_DTYPE) keyed by integer (ix, iy, iz) tuples via a companion dict.
No Open3D dependency -- pure numpy/dict data structure.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from legobuilder.config import WorldMapConfig, BLOCK_SIZE

VOXEL_DTYPE = np.dtype([
    ('key',       np.int32,   (3,)),
    ('log_odds',  np.float32),
    ('obs_count', np.int32),
    ('last_scan', np.int32),
    ('color',     np.float32, (3,)),
    ('kf_mean',   np.float64, (3,)),
    ('kf_var',    np.float64, (3,)),
    ('kf_n',      np.int32),
])


@dataclass(frozen=True)
class HeightResult:
    """Result from query_height().

    Attributes
    ----------
    max_z : float
        Maximum Z (height) among passing voxels, in world frame.
    voxel_count : int
        Number of voxels that passed the confidence and
        observation-count filters.
    mean_log_odds : float
        Mean log-odds of passing voxels, indicating overall
        confidence in the height measurement.
    """

    max_z: float
    voxel_count: int
    mean_log_odds: float


class VoxelStore:
    """Sparse voxel grid with Bayesian occupancy, decay, and frustum miss.

    Storage is keyed by integer (ix, iy, iz) tuples derived from
    world-frame coordinates.  A single structured numpy array
    (self._data, dtype=VOXEL_DTYPE) holds all per-voxel state: log-odds
    occupancy, observation count, last-seen scan index, RGB color, and
    optional Kalman position refinement fields.  A companion dict
    (self._voxels) provides O(1) key-to-index lookup.

    Each integrate() call runs the full probabilistic pipeline:

    1. Decay -- multiplicative confidence decay on stale positive log-odds
    2. Hit update -- log-odds addition with clamping to [L_MIN, L_MAX]
       and observation count capped at obs_count_max
    3. Frustum miss -- penalty for confident voxels in the camera's field
       of view but not observed this scan
    4. Pruning -- periodic garbage collection of dead voxels (near L_MIN)

    The grid doubles its backing array on demand so that typical integrate()
    calls involve only bulk numpy operations plus a dict-membership check over
    the (usually 1-10 K) unique voxels per frame.
    """

    # ── Construction / Reset ─────────────────────────────────────────────

    def __init__(self, config: Optional[WorldMapConfig] = None, logger=None) -> None:
        """Initialise the voxel store.

        Arguments
        ---------
        config : WorldMapConfig, optional
            World-map configuration.  Falls back to default
            WorldMapConfig() when *None*.
        """
        self._config = config or WorldMapConfig()
        self.logger = logger

        # Workspace AABB derived from cylindrical workspace bounds.
        # Used for voxel index computation (origin), not filtering.
        cfg = self._config
        bx = cfg.workspace_base_xy
        r = cfg.workspace_outer_radius
        self._bounds_min = np.array([
            bx[0] - r, bx[1] - r, cfg.workspace_z_min
        ])
        self._bounds_max = np.array([
            bx[0] + r, bx[1] + r, cfg.workspace_z_max
        ])

        # Pre-allocate structured array.
        self._capacity: int = 1024
        self._size: int = 0

        self._voxels: dict[tuple[int, int, int], int] = {}
        self._data = np.zeros(self._capacity, dtype=VOXEL_DTYPE)
        self._scan_index: int = 0
        self.table_height: float = 0 # Most recent detection of table height

    def reset(self) -> None:
        """Clear all voxels and re-initialise storage to initial capacity."""
        self._voxels.clear()
        self._size = 0
        self._scan_index = 0
        self._capacity = 1024
        self._data = np.zeros(self._capacity, dtype=VOXEL_DTYPE)

    # ── State Restoration ────────────────────────────────────────────────

    def _restore_from_arrays(
        self,
        keys: np.ndarray,
        log_odds: np.ndarray,
        obs_count: np.ndarray,
        last_scan: np.ndarray,
        colors: np.ndarray,
        scan_index: int,
        kf_mean: Optional[np.ndarray] = None,
        kf_var: Optional[np.ndarray] = None,
        kf_n: Optional[np.ndarray] = None,
    ) -> None:
        """Bulk-restore internal state from saved arrays.

        Called by :meth:`WorldMap.load()` to reconstruct a VoxelStore from
        a persisted .npz file.  The caller is responsible for ensuring
        array shapes and dtypes are consistent.

        Arguments
        ---------
        keys : np.ndarray
            (N, 3) int32 voxel index keys.
        log_odds : np.ndarray
            (N,) float32 log-odds occupancy values.
        obs_count : np.ndarray
            (N,) int32 observation counts.
        last_scan : np.ndarray
            (N,) int32 last-seen scan indices.
        colors : np.ndarray
            (N, 3) float32 RGB colours in [0, 1].
        scan_index : int
            Global scan counter to restore.
        kf_mean : np.ndarray, optional
            (N, 3) float64 Kalman position estimates.
        kf_var : np.ndarray, optional
            (N, 3) float64 Kalman position variances.
        kf_n : np.ndarray, optional
            (N,) int32 Kalman observation counts.
        """
        n = len(keys)

        # Grow capacity to hold all restored voxels.
        while self._capacity < n:
            self._grow(n)

        # Copy arrays into structured array fields.
        self._data['key'][:n] = keys
        self._data['log_odds'][:n] = log_odds
        self._data['obs_count'][:n] = obs_count
        self._data['last_scan'][:n] = last_scan
        self._data['color'][:n] = colors

        # Rebuild the voxel lookup dict.
        self._voxels.clear()
        for i in range(n):
            self._voxels[tuple(keys[i])] = i

        self._size = n
        self._scan_index = scan_index

        # Restore Kalman state if available and enabled.
        if self._config.kalman_enabled and kf_mean is not None:
            self._data['kf_mean'][:n] = kf_mean
            self._data['kf_var'][:n] = kf_var
            self._data['kf_n'][:n] = kf_n

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def scan_index(self) -> int:
        """Current global scan counter."""
        return self._scan_index

    @property
    def voxel_count(self) -> int:
        """Number of active (occupied) voxels."""
        return self._size

    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes.

        Accounts for the structured array (VOXEL_DTYPE.itemsize = 88 bytes
        per voxel, always includes Kalman fields) plus approximate CPython
        dict overhead (~100 bytes per entry for dict with tuple keys).
        """
        per_voxel_array = VOXEL_DTYPE.itemsize  # 88 bytes
        per_voxel_dict = 100  # approximate CPython overhead
        return self._size * (per_voxel_array + per_voxel_dict)

    # ── Dunder Helpers ────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of active voxels."""
        return self._size

    def __contains__(self, key: tuple[int, int, int]) -> bool:
        """Check whether *key* (ix, iy, iz) exists in the store."""
        return key in self._voxels

    # ── Single-voxel Access ──────────────────────────────────────────────

    def get_voxel(self, key: tuple[int, int, int]) -> Optional[dict]:
        """Return per-voxel state for *key*, or *None* if absent.

        Arguments
        ---------
        key : tuple[int, int, int]
            Integer voxel index (ix, iy, iz).

        Returns
        -------
        dict or None
            Dict with 'log_odds', 'obs_count', 'last_scan',
            'color' entries, or *None*.
        """
        idx = self._voxels.get(key)
        if idx is None:
            return None
        result = {
            "log_odds": float(self._data['log_odds'][idx]),
            "obs_count": int(self._data['obs_count'][idx]),
            "last_scan": int(self._data['last_scan'][idx]),
            "color": self._data['color'][idx].copy(),
        }
        if self._config.kalman_enabled:
            result["kf_mean"] = self._data['kf_mean'][idx].copy()
            result["kf_var"] = self._data['kf_var'][idx].copy()
            result["kf_n"] = int(self._data['kf_n'][idx])
        return result

    # ── Bulk Data Access ──────────────────────────────────────────────────

    def get_all_data(self) -> dict[str, np.ndarray]:
        """Return all per-voxel data as sliced numpy arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Dict with keys 'keys' (Nx3 int32), 'centers' (Nx3 float64),
            'log_odds' (N float32), 'obs_count' (N int32),
            'last_scan' (N int32), 'colors' (Nx3 float32).
        """
        n = self._size
        if n == 0:
            return {
                "keys": np.empty((0, 3), dtype=np.int32),
                "centers": np.empty((0, 3), dtype=np.float64),
                "positions": np.empty((0, 3), dtype=np.float64),
                "log_odds": np.empty(0, dtype=np.float32),
                "obs_count": np.empty(0, dtype=np.int32),
                "last_scan": np.empty(0, dtype=np.int32),
                "colors": np.empty((0, 3), dtype=np.float32),
            }

        keys = self._data['key'][:n].copy()
        res = self._config.voxel_resolution
        centers = (keys.astype(np.float64) + 0.5) * res + self._bounds_min

        # Positions: refined (Kalman) where available, centroids otherwise.
        if self._config.kalman_enabled:
            positions = centers.copy()
            refined_mask = self._data['kf_n'][:n] >= 2
            if np.any(refined_mask):
                positions[refined_mask] = (
                    self._data['kf_mean'][:n][refined_mask]
                )
        else:
            positions = centers

        return {
            "keys": keys,
            "centers": centers,
            "positions": positions,
            "log_odds": self._data['log_odds'][:n].copy(),
            "obs_count": self._data['obs_count'][:n].copy(),
            "last_scan": self._data['last_scan'][:n].copy(),
            "colors": self._data['color'][:n].copy(),
        }

    def get_kalman_data(self) -> Optional[dict[str, np.ndarray]]:
        """Return Kalman state arrays for persistence, or None if disabled."""
        if not self._config.kalman_enabled:
            return None
        n = self._size
        return {
            'kf_mean': self._data['kf_mean'][:n].copy(),
            'kf_var': self._data['kf_var'][:n].copy(),
            'kf_n': self._data['kf_n'][:n].copy(),
        }

    def voxel_center(self, key: tuple[int, int, int]) -> np.ndarray:
        """Return the 3-D world position of *key*'s voxel centre.

        Arguments
        ---------
        key : tuple[int, int, int]
            Integer voxel index (ix, iy, iz).

        Returns
        -------
        np.ndarray
            Length-3 float64 array [x, y, z].
        """
        res = self._config.voxel_resolution
        return (np.array(key, dtype=np.float64) + 0.5) * res + self._bounds_min

    # ── Queries and Clearing ─────────────────────────────────────────────

    def _empty_query_result(self) -> dict[str, np.ndarray]:
        """Return empty dict-of-arrays matching query_region format."""
        return {
            'centers': np.empty((0, 3), dtype=np.float64),
            'colors': np.empty((0, 3), dtype=np.float32),
            'log_odds': np.empty(0, dtype=np.float32),
            'obs_count': np.empty(0, dtype=np.int32),
            'keys': np.empty((0, 3), dtype=np.int32),
        }

    def query_height(
        self,
        x: float,
        y: float,
        radius: float,
        obs_min: int = 2,
        confidence_min: Optional[float] = None,
    ) -> Optional[HeightResult]:
        """Query maximum height in a cylindrical column around (x, y).

        Only voxels that pass both the confidence threshold and the
        observation-count gate are considered.  This ensures that noisy
        single-frame detections do not produce spurious height answers.

        Arguments
        ---------
        x : float
            Query X position in world frame.
        y : float
            Query Y position in world frame.
        radius : float
            XY search radius (metres).
        obs_min : int
            Minimum observation count (inclusive).  Default 2
            ensures single-observation voxels are excluded.
        confidence_min : float, optional
            Log-odds threshold.  None uses
            WorldMapConfig.confidence_threshold.

        Returns
        -------
        HeightResult or None
            A HeightResult with max_z, voxel_count, and
            mean_log_odds; or None if no voxels pass the filters.
        """
        n = self._size
        if n == 0:
            return None

        conf_thresh = (
            confidence_min
            if confidence_min is not None
            else self._config.confidence_threshold
        )

        res = self._config.voxel_resolution
        centers = (
            self._data['key'][:n].astype(np.float64) + 0.5
        ) * res + self._bounds_min

        # Cylindrical (XY-only) distance.
        dxy = np.linalg.norm(centers[:, :2] - np.array([x, y]), axis=1)

        mask = (
            (dxy < radius)
            & (self._data['log_odds'][:n] > conf_thresh)
            & (self._data['obs_count'][:n] >= obs_min)
        )

        if not np.any(mask):
            return None

        return HeightResult(
            max_z=float(np.max(centers[mask, 2])),
            voxel_count=int(np.sum(mask)),
            mean_log_odds=float(
                np.mean(self._data['log_odds'][:n][mask])
            ),
        )

    def query_region(
        self,
        xy_center: np.ndarray,
        radius: float,
        z_min: float,
        z_max: float,
        confidence_min: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Query voxels in a cylindrical region.

        Returns all voxel state (centers, colors, log-odds, obs-count, keys)
        for voxels within the XY radius and Z range that pass the confidence
        gate.  Does not apply observation-count gating.

        Arguments
        ---------
        xy_center : np.ndarray
            Length-2 array [x, y] in world frame.
        radius : float
            XY search radius (metres).
        z_min : float
            Minimum Z (inclusive); meters;
        z_max : float
            Maximum Z (inclusive); meters;
        confidence_min : float, optional
            Log-odds threshold.  None uses
            WorldMapConfig.confidence_threshold.

        Returns
        -------
        dict[str, np.ndarray]
            Dict of numpy arrays with keys 'centers' (Nx3 float64),
            'colors' (Nx3 float32), 'log_odds' (N float32),
            'obs_count' (N int32), 'keys' (Nx3 int32).  Empty
            arrays (N=0) when no voxels pass the filters.
        """
        n = self._size
        if n == 0:
            return self._empty_query_result()

        conf_thresh = (
            confidence_min
            if confidence_min is not None
            else self._config.confidence_threshold
        )

        res = self._config.voxel_resolution
        centers = (
            self._data['key'][:n].astype(np.float64) + 0.5
        ) * res + self._bounds_min

        # Cylindrical (XY distance + Z range).
        dxy = np.linalg.norm(centers[:, :2] - xy_center[:2], axis=1)
        mask = (
            (dxy < radius)
            & (centers[:, 2] >= z_min)
            & (centers[:, 2] <= z_max)
            & (self._data['log_odds'][:n] > conf_thresh)
        )

        if not np.any(mask):
            return self._empty_query_result()

        return {
            'centers': centers[mask].copy(),
            'colors': self._data['color'][:n][mask].copy(),
            'log_odds': self._data['log_odds'][:n][mask].copy(),
            'obs_count': self._data['obs_count'][:n][mask].copy(),
            'keys': self._data['key'][:n][mask].copy(),
        }

    def clear_region(self, center: np.ndarray, radius: float) -> int:
        """Remove all voxels within a sphere.

        Used by the brain to instantly suppress ghost blocks after a pick.
        Voxels are deleted (not reset to L_MIN), and the backing
        array is compacted via :meth:`_compact`.

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
        n = self._size
        if n == 0:
            return 0

        res = self._config.voxel_resolution
        centers = (
            self._data['key'][:n].astype(np.float64) + 0.5
        ) * res + self._bounds_min

        # Spherical (3D Euclidean) distance.
        dists = np.linalg.norm(centers - center, axis=1)
        clear_mask = dists <= radius

        n_clear = int(np.sum(clear_mask))
        if n_clear == 0:
            return 0

        keep_mask = ~clear_mask
        return self._compact(keep_mask)

    # ── Integration ──────────────────────────────────────────────────────

    def integrate(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        T_world_camera: np.ndarray,
    ) -> int:
        """Integrate a point cloud frame into the voxel store.

        Full probabilistic pipeline:

        1. Increment scan index
        2. Decay stale positive log-odds (multiplicative)
        3. Transform points from camera to world frame
        4. Normalise colours, clip to workspace bounds
        5. Voxelise and deduplicate
        6. Hit update with clamping to [L_MIN, L_MAX] and obs_count cap
        7. Insert new voxels
        8. Frustum miss penalty for confident in-view unobserved voxels
        9. Prune dead voxels periodically

        Arguments
        ---------
        points : np.ndarray
            (N, 3) float array of XYZ in camera frame.
        colors : np.ndarray
            (N, 3) float or uint8 array of RGB per point.
        T_world_camera : np.ndarray
            (4, 4) homogeneous camera-to-world transform.

        Returns
        -------
        int
            Number of new voxels created by this call.
        """
        self._scan_index += 1

        # ── 1. Decay (before any hit processing) ─────────────────────
        self._apply_decay()

        # ── Early exit on empty input ─────────────────────────────────
        if len(points) == 0:
            # Still apply frustum miss and pruning even with no new points
            observed_mask = np.zeros(self._size, dtype=bool)
            self._apply_frustum_miss(T_world_camera, observed_mask)
            self._maybe_prune()
            return 0

        # ── 2. Transform to world frame ──────────────────────────────
        R = T_world_camera[:3, :3]
        t = T_world_camera[:3, 3]
        points_world = (R @ points.T).T + t

        # ── 3. Normalise colours to float32 [0, 1] ──────────────────
        colors = np.asarray(colors, dtype=np.float32)
        if colors.size > 0 and colors.max() > 1.0:
            colors = colors / 255.0

        # ── 4. Clip to cylindrical workspace bounds ──────────────────
        cfg = self._config
        dxy = points_world[:, :2] - cfg.workspace_base_xy
        r_xy = np.sqrt(dxy[:, 0]**2 + dxy[:, 1]**2)
        in_bounds = (
            (r_xy >= cfg.workspace_inner_radius)
            & (r_xy <= cfg.workspace_outer_radius)
            & (points_world[:, 0] > 0.0)   # matches is_point_reachable: x > 0
            & (points_world[:, 1] > 0.0)   # matches is_point_reachable: y > 0
            & (points_world[:, 2] >= cfg.workspace_z_min)
            & (points_world[:, 2] <= cfg.workspace_z_max)
        )

        points_world = points_world[in_bounds]
        colors = colors[in_bounds]

        if len(points_world) == 0:
            observed_mask = np.zeros(self._size, dtype=bool)
            self._apply_frustum_miss(T_world_camera, observed_mask)
            self._maybe_prune()
            return 0
        
        # ── 5. Calibrate z to table height ───────────────────────────
        table_height = self._determine_table_height(points_world)
        self.table_height = table_height
        # self.logger.info(f"table_height : {table_height}")
        points_world[:, 2] -= table_height

        # ── 6. Voxelise ──────────────────────────────────────────────
        voxel_keys = self._world_to_voxel(points_world)

        # ── 7. Deduplicate (last-write-wins for colour) ──────────────
        unique_keys, inverse = np.unique(
            voxel_keys, axis=0, return_inverse=True
        )
        n_unique = len(unique_keys)
        last_colors = np.zeros((n_unique, 3), dtype=np.float32)
        last_colors[inverse] = colors  # last-write-wins

        # ── 7b. Per-voxel averaged measurements (Kalman) ─────────────
        if self._config.kalman_enabled:
            mean_positions = np.zeros((n_unique, 3), dtype=np.float64)
            point_counts = np.zeros(n_unique, dtype=np.int32)
            np.add.at(mean_positions, inverse, points_world.astype(np.float64))
            np.add.at(point_counts, inverse, 1)
            mean_positions /= point_counts[:, np.newaxis]

        # ── 8. Separate existing vs new voxels ────────────────────────
        existing_indices = []  # structured-array indices for existing voxels
        existing_unique = []   # which unique-key row maps to existing
        new_unique = []        # which unique-key row is new

        for i in range(n_unique):
            key = (int(unique_keys[i, 0]),
                   int(unique_keys[i, 1]),
                   int(unique_keys[i, 2]))
            idx = self._voxels.get(key)
            if idx is not None:
                existing_indices.append(idx)
                existing_unique.append(i)
            else:
                new_unique.append(i)

        # ── 9. Hit update with clamping (existing voxels) ─────────────
        if existing_indices:
            ei = np.array(existing_indices, dtype=np.intp)
            eu = np.array(existing_unique, dtype=np.intp)

            # Log-odds hit + clamp (fancy-index returns copy, must assign back)
            vals = self._data['log_odds'][ei] + self._config.log_odds_hit
            np.clip(vals, self._config.log_odds_min,
                    self._config.log_odds_max, out=vals)
            self._data['log_odds'][ei] = vals

            # Obs count + cap
            counts = self._data['obs_count'][ei] + 1
            np.minimum(counts, self._config.obs_count_max, out=counts)
            self._data['obs_count'][ei] = counts

            self._data['last_scan'][ei] = self._scan_index
            self._data['color'][ei] = last_colors[eu]

            # Kalman predict+update for existing voxels.
            if self._config.kalman_enabled:
                Q = self._config.kalman_process_noise
                R = self._config.kalman_measurement_noise
                z_k = mean_positions[eu]   # per-voxel averaged measurement

                # Split: voxels with kf_n==0 (decay-reset) need re-init,
                # others get normal Kalman update.
                kf_n_vals = self._data['kf_n'][ei]
                reset_mask = kf_n_vals == 0

                if np.any(~reset_mask):
                    # Normal Kalman predict+update for non-reset voxels.
                    upd_ei = ei[~reset_mask]
                    upd_z = z_k[~reset_mask]

                    P_pred = self._data['kf_var'][upd_ei] + Q
                    K = P_pred / (P_pred + R)
                    innovation = upd_z - self._data['kf_mean'][upd_ei]

                    new_mean = (
                        self._data['kf_mean'][upd_ei] + K * innovation
                    )
                    new_var = (1.0 - K) * P_pred
                    new_n = self._data['kf_n'][upd_ei] + 1

                    self._data['kf_mean'][upd_ei] = new_mean
                    self._data['kf_var'][upd_ei] = new_var
                    self._data['kf_n'][upd_ei] = new_n

                if np.any(reset_mask):
                    # Re-initialize decay-reset voxels.
                    rst_ei = ei[reset_mask]
                    rst_eu = np.array(
                        existing_unique, dtype=np.intp
                    )[reset_mask]
                    rst_z = z_k[reset_mask]
                    rst_n_pts = point_counts[rst_eu]
                    var_init = np.maximum(
                        R / rst_n_pts[:, np.newaxis], R * 0.5
                    )
                    self._data['kf_mean'][rst_ei] = rst_z
                    self._data['kf_var'][rst_ei] = var_init
                    self._data['kf_n'][rst_ei] = 1

        # ── 10. Insert new voxels ─────────────────────────────────────
        old_size = self._size
        n_new = len(new_unique)
        if n_new > 0:
            required = self._size + n_new
            if required > self._capacity:
                self._grow(required)

            start = self._size
            end = start + n_new
            nu = np.array(new_unique, dtype=np.intp)

            self._data['log_odds'][start:end] = self._config.log_odds_prior
            self._data['obs_count'][start:end] = 1
            self._data['last_scan'][start:end] = self._scan_index
            self._data['color'][start:end] = last_colors[nu]
            self._data['key'][start:end] = unique_keys[nu]

            for j, ui in enumerate(new_unique):
                key = (int(unique_keys[ui, 0]),
                       int(unique_keys[ui, 1]),
                       int(unique_keys[ui, 2]))
                self._voxels[key] = start + j

            # Kalman initialization for new voxels.
            if self._config.kalman_enabled:
                R = self._config.kalman_measurement_noise
                z_k = mean_positions[nu]
                n_pts = point_counts[nu]

                # Initial variance: R/n_pts floored at R*0.5
                var_init = np.maximum(
                    R / n_pts[:, np.newaxis], R * 0.5
                )

                self._data['kf_mean'][start:end] = z_k
                self._data['kf_var'][start:end] = var_init
                self._data['kf_n'][start:end] = 1

            self._size = end

        # ── 11. Build observed_mask ──────────────────────────────────
        observed_mask = np.zeros(self._size, dtype=bool)
        if existing_indices:
            observed_mask[np.array(existing_indices, dtype=np.intp)] = True
        if n_new > 0:
            observed_mask[old_size:self._size] = True

        # ── 12. Frustum miss ─────────────────────────────────────────
        self._apply_frustum_miss(T_world_camera, observed_mask)

        # ── 13. Periodic pruning ─────────────────────────────────────
        self._maybe_prune()

        # ── 14. Z-remainder filtering and memory eviction ─────────
        self._apply_z_remainder_filter()
        self._evict_if_needed()

        return n_new

    # ── Probabilistic Helpers ─────────────────────────────────────────

    def _apply_decay(self) -> None:
        """Apply multiplicative decay to stale positive-confidence voxels.

        Prevents ghost voxels from persisting indefinitely when an object
        moves or is removed.  Only positive log-odds are decayed -- negative
        log-odds (known-free space) are left untouched to avoid drifting
        them toward zero.  When a voxel's log-odds decays below the
        confidence threshold, its Kalman state is reset so the next
        observation reinitialises the position estimate from scratch.
        """
        n = self._size
        if n == 0:
            return
        d = self._data[:n]
        staleness = self._scan_index - d['last_scan']
        stale_mask = staleness > self._config.decay_after_scans
        # Only decay positive log-odds
        decay_mask = stale_mask & (d['log_odds'] > 0.0)
        if np.any(decay_mask):
            d['log_odds'][decay_mask] *= self._config.decay_factor

            # Reset Kalman state for voxels that decayed below confidence
            # threshold. Only applies to voxels actually decayed this call.
            if self._config.kalman_enabled:
                decayed_below = (
                    decay_mask
                    & (d['log_odds'] <= self._config.confidence_threshold)
                    & (d['kf_n'] > 0)
                )
                if np.any(decayed_below):
                    d['kf_mean'][decayed_below] = 0.0
                    d['kf_var'][decayed_below] = 0.0
                    d['kf_n'][decayed_below] = 0

    def _apply_frustum_miss(
        self,
        T_world_camera: np.ndarray,
        observed_mask: np.ndarray,
    ) -> None:
        """Apply miss penalty to confident voxels visible but not observed.

        If a confident voxel falls within the camera's field-of-view
        frustum but was not hit by any point this scan, it receives a
        negative log-odds update.  This accelerates removal of voxels
        representing objects that have moved out of view or been picked.

        Arguments
        ---------
        T_world_camera : np.ndarray
            (4, 4) camera-to-world homogeneous transform.
        observed_mask : np.ndarray
            Boolean array of size self._size.  True for voxels
            that were hit or newly inserted this scan.
        """
        n = self._size
        if n == 0:
            return

        # Reconstruct voxel centres in world frame from structured array.
        res = self._config.voxel_resolution
        centers_world = (
            self._data['key'][:n].astype(np.float64) + 0.5
        ) * res + self._bounds_min

        # Transform to camera frame.
        T_camera_world = np.linalg.inv(T_world_camera)
        R_cw = T_camera_world[:3, :3]
        t_cw = T_camera_world[:3, 3]
        centers_cam = (R_cw @ centers_world.T).T + t_cw

        # Frustum bounding-box check (all vectorised).
        z = centers_cam[:, 2]
        x = centers_cam[:, 0]
        y = centers_cam[:, 1]

        in_depth = (
            (z >= self._config.range_min)
            & (z <= self._config.range_max)
        )
        in_horiz = np.abs(x) < z * np.tan(self._config.camera_hfov / 2.0)
        in_vert = np.abs(y) < z * np.tan(self._config.camera_vfov / 2.0)
        in_frustum = in_depth & in_horiz & in_vert

        # Miss mask: in frustum, not observed, confident enough.
        confident = (
            self._data['log_odds'][:n] > self._config.confidence_threshold
        )
        miss_mask = in_frustum & ~observed_mask & confident

        if np.any(miss_mask):
            # Boolean mask on slice: += works via __setitem__, but np.clip
            # with out= on a boolean-indexed result writes to a copy.
            # Use explicit read-modify-write instead.
            vals = (
                self._data['log_odds'][:n][miss_mask]
                + self._config.log_odds_miss
            )
            np.clip(vals, self._config.log_odds_min,
                    self._config.log_odds_max, out=vals)
            self._data['log_odds'][:n][miss_mask] = vals

    def _maybe_prune(self) -> int:
        """Prune dead voxels if at prune interval.

        Returns
        -------
        int
            Number of voxels pruned.
        """
        if self._scan_index % self._config.prune_interval != 0:
            return 0

        n = self._size
        if n == 0:
            return 0

        # Keep voxels above pruning threshold (L_MIN + epsilon).
        keep_mask = (
            self._data['log_odds'][:n]
            >= (self._config.log_odds_min + 0.01)
        )
        return self._compact(keep_mask)

    def _apply_z_remainder_filter(self) -> int:
        """Remove voxels whose z-index is not near a block boundary.

        Computes the distance from each voxel's integer z-index to the
        nearest block-height multiple in voxel units.  Voxels further
        than z_remainder_threshold are discarded.  Operates on the raw
        voxel grid key (integer iz), not Kalman-refined positions, to
        avoid oscillation from floating-point jitter.

        Returns
        -------
        int
            Number of voxels removed.
        """
        if self._config.z_remainder_threshold == float('inf'):
            return 0
        n = self._size
        if n == 0:
            return 0

        block_height_voxels = (
            BLOCK_SIZE / self._config.voxel_resolution
        )
        iz = self._data['key'][:n, 2].astype(np.float64)
        remainder = iz % block_height_voxels
        dist_to_boundary = np.minimum(
            remainder, block_height_voxels - remainder
        )
        keep_mask = dist_to_boundary <= self._config.z_remainder_threshold
        removed = n - int(np.sum(keep_mask))
        if removed > 0:
            self._compact(keep_mask)
        return removed

    def _evict_if_needed(self) -> int:
        """Evict excess voxels when store exceeds max_voxels limit.

        Eviction priority (highest score = most evictable):
        1. Furthest from workspace centre (50% weight)
        2. Lowest confidence / log-odds (30% weight)
        3. Oldest unseen / stalest (20% weight)

        Returns
        -------
        int
            Number of voxels evicted.
        """
        if self._size <= self._config.max_voxels:
            return 0

        n = self._size
        d = self._data[:n]
        res = self._config.voxel_resolution

        # Distance from workspace centre.
        centers = (
            d['key'].astype(np.float64) + 0.5
        ) * res + self._bounds_min
        ws_center = self._config.workspace_base_xy
        dxy = np.linalg.norm(
            centers[:, :2] - ws_center, axis=1
        )

        # Normalised scores (higher = more evictable).
        max_dxy = dxy.max() if dxy.max() > 0 else 1.0
        dist_score = dxy / max_dxy

        lo_range = (
            self._config.log_odds_max - self._config.log_odds_min
        )
        conf_score = 1.0 - (
            d['log_odds'] - self._config.log_odds_min
        ) / (lo_range if lo_range > 0 else 1.0)

        max_scan = max(self._scan_index, 1)
        age_score = (
            1.0 - d['last_scan'].astype(np.float64) / max_scan
        )

        eviction_score = (
            0.5 * dist_score + 0.3 * conf_score + 0.2 * age_score
        )

        n_keep = self._config.max_voxels
        keep_indices = np.argpartition(
            eviction_score, n_keep
        )[:n_keep]
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep_indices] = True
        removed = n - n_keep
        self._compact(keep_mask)
        return removed

    def _compact(self, keep_mask: np.ndarray) -> int:
        """Remove voxels where *keep_mask* is False and defragment.

        Compacts the structured array (self._data), zeros freed slots,
        and rebuilds the _voxels dict with updated indices.  This keeps
        the backing array contiguous so vectorised operations remain
        efficient after pruning or region clearing.

        Arguments
        ---------
        keep_mask : np.ndarray
            Boolean array of length self._size.  True for
            voxels to keep.

        Returns
        -------
        int
            Number of voxels removed.
        """
        n = self._size
        n_keep = int(np.sum(keep_mask))
        n_removed = n - n_keep

        if n_removed == 0:
            return 0

        # Compact structured array via fancy indexing.
        keep_indices = np.where(keep_mask)[0]
        self._data[:n_keep] = self._data[keep_indices]

        # Zero out freed slots (defensive).
        self._data[n_keep:n] = np.zeros(1, dtype=VOXEL_DTYPE)

        # Rebuild dict: old indices are invalid after compaction.
        old_idx_to_key = {v: k for k, v in self._voxels.items()}
        self._voxels.clear()
        for new_idx, old_idx in enumerate(keep_indices):
            key = old_idx_to_key[old_idx]
            self._voxels[key] = new_idx

        self._size = n_keep
        return n_removed

    # ── Private Helpers ───────────────────────────────────────────────────

    def _world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """Convert Nx3 world-frame points to Nx3 integer voxel indices.

        Arguments
        ---------
        points : np.ndarray
            (N, 3) float array in world frame (already clipped).

        Returns
        -------
        np.ndarray
            (N, 3) int32 array of voxel indices.
        """
        return np.floor(
            (points - self._bounds_min) / self._config.voxel_resolution
        ).astype(np.int32)

    def _grow(self, min_capacity: int) -> None:
        """Double backing array until it can hold *min_capacity* voxels."""
        new_cap = self._capacity
        while new_cap < min_capacity:
            new_cap *= 2

        new_data = np.zeros(new_cap, dtype=VOXEL_DTYPE)
        new_data[:self._size] = self._data[:self._size]
        self._data = new_data
        self._capacity = new_cap

    def _determine_table_height(self, points_world: np.ndarray) -> float:
        """Estimate table surface Z from world-frame points near z=0."""
        z_vals = points_world[:, 2]
        table_mask = z_vals < BLOCK_SIZE / 2
        table_z = z_vals[table_mask]
        if len(table_z) == 0:
            return 0.0
        table_z_sorted = np.sort(table_z)
        q1 = int(len(table_z_sorted) * 0.25)
        q3 = int(len(table_z_sorted) * 0.75)
        if q1 == q3:
            return float(np.median(table_z_sorted))
        return float(np.mean(table_z_sorted[q1:q3]))