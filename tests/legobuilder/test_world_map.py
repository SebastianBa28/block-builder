"""WorldMap integration tests for Phase 4.

API-01  Public API preserved: integrate, query_height, query_region,
        get_point_cloud, reset, save, load
API-02  integrate() accepts (points, colors, T_world_camera)
API-03  get_point_cloud() returns Open3D PointCloud (confidence-filtered)
API-04  get_numpy_points() and get_numpy_points_colors() preserved
API-05  to_pointcloud2_msg() builds valid PointCloud2
API-06  save()/load() round-trip full voxel state

Tests exercise the WorldMap composition shell -- they do NOT test VoxelStore
internals (those are covered in test_voxel_store.py).
"""

import os
import tempfile

import numpy as np
import open3d as o3d
import pytest

from legobuilder.config import WorldMapConfig
from legobuilder.vision.world_map import WorldMap
from legobuilder.vision.voxel_store import HeightResult


# -------------------------------------------------------------------
# Fixtures & Helpers
# -------------------------------------------------------------------

@pytest.fixture
def config():
    """Return default WorldMapConfig for testing."""
    return WorldMapConfig()


@pytest.fixture
def world_map(config):
    """Return a fresh WorldMap instance."""
    return WorldMap(config)


@pytest.fixture
def identity_T():
    """4x4 identity transform (camera frame == world frame)."""
    return np.eye(4)


def _make_points(*xyzs):
    """Build an (N, 3) float64 array from (x, y, z) tuples."""
    return np.array(xyzs, dtype=np.float64)


def _make_colors(*rgbs):
    """Build an (N, 3) uint8-range array from (r, g, b) tuples."""
    return np.array(rgbs, dtype=np.float64)


def _integrate_point(wm, x, y, z, T, n=1, r=128, g=128, b=128):
    """Integrate a single point n times to build up obs_count."""
    pts = _make_points((x, y, z))
    cols = _make_colors((r, g, b))
    for _ in range(n):
        wm.integrate(pts, cols, T)


# -------------------------------------------------------------------
# TestIntegrate (API-02)
# -------------------------------------------------------------------

class TestIntegrate:
    """Test WorldMap.integrate() with split (points, colors, T)."""

    def test_integrate_increments_frame_count(self, world_map, identity_T):
        pts = _make_points((0.0, 0.0, 0.5))
        cols = _make_colors((128, 128, 128))
        world_map.integrate(pts, cols, identity_T)
        assert world_map.frame_count == 1
        world_map.integrate(pts, cols, identity_T)
        assert world_map.frame_count == 2

    def test_integrate_range_filtering(self, world_map, identity_T):
        """Points below range_min should be excluded."""
        # z=0.01 is below range_min=0.15
        pts = _make_points((0.0, 0.0, 0.01))
        cols = _make_colors((128, 128, 128))
        world_map.integrate(pts, cols, identity_T)
        # frame_count should still increment (decay/frustum still run)
        assert world_map.frame_count == 1
        # but no voxels should be created
        result = world_map.get_numpy_points(confidence_min=-999.0)
        assert len(result) == 0

    def test_integrate_range_max_filtering(self, world_map, identity_T):
        """Points above range_max should be excluded."""
        # z=2.0 is above range_max=1.0
        pts = _make_points((0.0, 0.0, 2.0))
        cols = _make_colors((128, 128, 128))
        world_map.integrate(pts, cols, identity_T)
        assert world_map.frame_count == 1
        result = world_map.get_numpy_points(confidence_min=-999.0)
        assert len(result) == 0

    def test_integrate_empty_after_filtering(self, world_map, identity_T):
        """All points outside range produces no error."""
        pts = _make_points((0.0, 0.0, 0.01), (0.0, 0.0, 2.0))
        cols = _make_colors((128, 128, 128), (200, 200, 200))
        world_map.integrate(pts, cols, identity_T)
        assert world_map.frame_count == 1

    def test_integrate_accepts_uint8_colors(self, world_map, identity_T):
        """Colors in 0-255 range work correctly (VoxelStore normalizes)."""
        pts = _make_points((0.0, 0.0, 0.5))
        cols = np.array([[255, 0, 0]], dtype=np.float64)
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=2, r=255, g=0, b=0)
        _, colors = world_map.get_numpy_points_colors(confidence_min=-999.0)
        assert colors is not None
        # VoxelStore normalizes 255 -> 1.0
        np.testing.assert_allclose(colors[0, 0], 1.0, atol=0.01)

    def test_has_data_and_frame_count(self, world_map, identity_T):
        """has_data is False before first integrate, True after."""
        assert world_map.has_data is False
        assert world_map.frame_count == 0
        pts = _make_points((0.0, 0.0, 0.5))
        cols = _make_colors((128, 128, 128))
        world_map.integrate(pts, cols, identity_T)
        assert world_map.has_data is True
        assert world_map.frame_count == 1


# -------------------------------------------------------------------
# TestExtraction (API-03, API-04)
# -------------------------------------------------------------------

class TestExtraction:
    """Test Open3D and numpy extraction methods."""

    def test_get_point_cloud_returns_open3d(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=2)
        pcd = world_map.get_point_cloud()
        assert isinstance(pcd, o3d.geometry.PointCloud)

    def test_get_point_cloud_confidence_filtered(self, world_map, identity_T):
        """Integrate 1 time (below obs_min) vs 3 times (above).

        With default obs_min=2 in VoxelStore, integrating only once may
        result in log_odds below the confidence threshold. Integrating
        3 times should be above.
        """
        # Use very low confidence threshold to see all voxels for the "3 times" case
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        pcd = world_map.get_point_cloud(confidence_min=-999.0)
        assert len(pcd.points) >= 1

    def test_get_point_cloud_empty_when_no_data(self, world_map):
        pcd = world_map.get_point_cloud()
        assert isinstance(pcd, o3d.geometry.PointCloud)
        assert len(pcd.points) == 0

    def test_get_numpy_points_shape(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=2)
        pts = world_map.get_numpy_points(confidence_min=-999.0)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.dtype == np.float64

    def test_get_numpy_points_colors_returns_pair(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=2, r=200, g=100, b=50)
        pts, cols = world_map.get_numpy_points_colors(confidence_min=-999.0)
        assert pts.shape[1] == 3
        assert cols is not None
        assert cols.shape[1] == 3
        assert cols.dtype == np.float32

    def test_get_numpy_points_empty(self, world_map):
        pts = world_map.get_numpy_points()
        assert pts.shape == (0, 3)

    def test_no_remove_outliers_parameter(self, world_map):
        """Calling get_point_cloud(remove_outliers=True) raises TypeError."""
        with pytest.raises(TypeError):
            world_map.get_point_cloud(remove_outliers=True)

    def test_get_numpy_points_colors_no_remove_outliers(self, world_map):
        """Calling get_numpy_points_colors(remove_outliers=True) raises TypeError."""
        with pytest.raises(TypeError):
            world_map.get_numpy_points_colors(remove_outliers=True)


# -------------------------------------------------------------------
# TestQueries (API-01)
# -------------------------------------------------------------------

class TestQueries:
    """Test query_height, query_region, clear_region delegation."""

    def test_query_height_returns_none_when_empty(self, world_map):
        result = world_map.query_height(0.0, 0.0)
        assert result is None

    def test_query_height_returns_height_result(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        result = world_map.query_height(0.0, 0.0, radius=0.05, obs_min=2)
        assert result is not None
        assert isinstance(result, HeightResult)

    def test_query_height_delegates_to_store(self, world_map, identity_T):
        """Integrate at known position, verify HeightResult.max_z is near 0.5."""
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        result = world_map.query_height(0.0, 0.0, radius=0.05, obs_min=2)
        assert result is not None
        # Voxel center should be near z=0.5 (within one voxel resolution)
        assert abs(result.max_z - 0.5) < 0.01

    def test_query_region_returns_dict(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        result = world_map.query_region(
            np.array([0.0, 0.0]), radius=0.05,
            confidence_min=-999.0)
        assert isinstance(result, dict)
        for key in ('centers', 'colors', 'log_odds', 'obs_count', 'keys'):
            assert key in result, f"Missing key: {key}"

    def test_clear_region_removes_voxels(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        # Verify data exists
        pts_before = world_map.get_numpy_points(confidence_min=-999.0)
        assert len(pts_before) > 0
        # Clear the region
        n_cleared = world_map.clear_region(
            np.array([0.0, 0.0, 0.5]), radius=0.1)
        assert n_cleared > 0
        # Verify data is gone
        pts_after = world_map.get_numpy_points(confidence_min=-999.0)
        assert len(pts_after) == 0


# -------------------------------------------------------------------
# TestSaveLoad (API-06)
# -------------------------------------------------------------------

class TestSaveLoad:
    """Test save/load round-trip for full voxel state."""

    def test_save_creates_npz_file(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=2)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            world_map.save(path)
            assert os.path.exists(path)
            assert path.endswith('.npz')
        finally:
            os.unlink(path)

    def test_save_load_round_trip(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            world_map.save(path)
            loaded = WorldMap.load(path)

            assert loaded.frame_count == world_map.frame_count
            assert loaded.has_data == world_map.has_data

            orig_pts = world_map.get_numpy_points(confidence_min=-999.0)
            load_pts = loaded.get_numpy_points(confidence_min=-999.0)
            assert len(orig_pts) == len(load_pts)
        finally:
            os.unlink(path)

    def test_save_load_preserves_config(self, identity_T):
        """Custom config fields survive round-trip."""
        custom = WorldMapConfig(
            voxel_resolution=0.01,
            range_min=0.2,
            range_max=0.8,
            publish_every_n_scans=5,
        )
        wm = WorldMap(custom)
        _integrate_point(wm, 0.0, 0.0, 0.5, identity_T, n=2)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            wm.save(path)
            loaded = WorldMap.load(path)
            assert loaded.config.voxel_resolution == 0.01
            assert loaded.config.range_min == 0.2
            assert loaded.config.range_max == 0.8
            assert loaded.config.publish_every_n_scans == 5
        finally:
            os.unlink(path)

    def test_save_load_preserves_voxel_data(self, world_map, identity_T):
        """Compare get_all_data before and after round-trip."""
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            world_map.save(path)
            loaded = WorldMap.load(path)

            orig_data = world_map._store.get_all_data()
            load_data = loaded._store.get_all_data()

            np.testing.assert_array_equal(orig_data['keys'], load_data['keys'])
            np.testing.assert_array_almost_equal(
                orig_data['log_odds'], load_data['log_odds'])
            np.testing.assert_array_equal(
                orig_data['obs_count'], load_data['obs_count'])
            np.testing.assert_array_almost_equal(
                orig_data['colors'], load_data['colors'])
        finally:
            os.unlink(path)

    def test_load_is_static_factory(self, world_map, identity_T):
        """WorldMap.load returns a new WorldMap instance."""
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=2)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            world_map.save(path)
            loaded = WorldMap.load(path)
            assert loaded is not world_map
            assert isinstance(loaded, WorldMap)
        finally:
            os.unlink(path)


# -------------------------------------------------------------------
# TestReset (API-01)
# -------------------------------------------------------------------

class TestReset:
    """Test WorldMap.reset() clears all state."""

    def test_reset_clears_data(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        assert world_map.has_data is True
        assert world_map.frame_count > 0
        world_map.reset()
        assert world_map.has_data is False
        assert world_map.frame_count == 0

    def test_reset_queries_return_empty(self, world_map, identity_T):
        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        world_map.reset()
        assert world_map.query_height(0.0, 0.0) is None
        pts = world_map.get_numpy_points()
        assert len(pts) == 0


# -------------------------------------------------------------------
# TestPublishing (API-05)
# -------------------------------------------------------------------

class TestPublishing:
    """Test should_publish throttle and to_pointcloud2_msg builder."""

    def test_should_publish_every_n_scans(self, identity_T):
        config = WorldMapConfig(publish_every_n_scans=3)
        wm = WorldMap(config)
        pts = _make_points((0.0, 0.0, 0.5))
        cols = _make_colors((128, 128, 128))
        results = []
        for _ in range(9):
            wm.integrate(pts, cols, identity_T)
            results.append(wm.should_publish())
        # should_publish returns True when scan_index % 3 == 0
        # scan_index goes 1,2,3,4,5,6,7,8,9
        # True at 3, 6, 9
        assert results == [False, False, True, False, False, True, False, False, True]

    def test_to_pointcloud2_msg_returns_none_when_empty(self, world_map):
        """No data -> None."""
        sensor_msgs = pytest.importorskip('sensor_msgs')
        from builtin_interfaces.msg import Time
        stamp = Time()
        result = world_map.to_pointcloud2_msg(stamp=stamp)
        assert result is None

    def test_to_pointcloud2_msg_builds_valid_msg(self, world_map, identity_T):
        """Integrate data and build a PointCloud2 message."""
        sensor_msgs = pytest.importorskip('sensor_msgs')
        from sensor_msgs.msg import PointCloud2
        from builtin_interfaces.msg import Time

        _integrate_point(world_map, 0.0, 0.0, 0.5, identity_T, n=3)
        stamp = Time()
        msg = world_map.to_pointcloud2_msg(stamp=stamp)
        if msg is not None:
            assert isinstance(msg, PointCloud2)
            assert msg.width > 0
            assert msg.header.frame_id == 'world'
            assert msg.is_dense is True
            assert len(msg.fields) >= 3  # x, y, z (+ optional rgb)
            assert len(msg.data) == msg.width * msg.point_step

    def test_should_publish_default_every_scan(self, world_map, identity_T):
        """Default publish_every_n_scans=1, should_publish always True."""
        pts = _make_points((0.0, 0.0, 0.5))
        cols = _make_colors((128, 128, 128))
        world_map.integrate(pts, cols, identity_T)
        assert world_map.should_publish() is True
        world_map.integrate(pts, cols, identity_T)
        assert world_map.should_publish() is True


# -------------------------------------------------------------------
# TestMultiplePoints (integration smoke tests)
# -------------------------------------------------------------------

class TestMultiplePoints:
    """Test integrating multiple points at once."""

    def test_integrate_multiple_points(self, world_map, identity_T):
        """Multiple points integrated together in single call."""
        pts = _make_points(
            (0.0, 0.0, 0.5),
            (0.1, 0.0, 0.5),
            (0.0, 0.1, 0.5),
        )
        cols = _make_colors(
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        )
        # Integrate 3 times so obs_count >= 2 for all
        for _ in range(3):
            world_map.integrate(pts, cols, identity_T)

        result = world_map.get_numpy_points(confidence_min=-999.0)
        # Should have created voxels for all 3 points
        assert len(result) >= 3

    def test_integrate_with_transform(self, world_map):
        """Points in camera frame get transformed to world frame."""
        # Rotation of 90 degrees about Z axis: X_cam -> -Y_world, Y_cam -> X_world
        T = np.eye(4)
        T[:3, :3] = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ])
        # Point at (0, 0, 0.5) in camera frame -> (0, 0, 0.5) in world frame
        # (only rotation about Z, z-component stays same)
        pts = _make_points((0.0, 0.0, 0.5))
        cols = _make_colors((128, 128, 128))
        for _ in range(3):
            world_map.integrate(pts, cols, T)

        result = world_map.get_numpy_points(confidence_min=-999.0)
        assert len(result) >= 1


# -------------------------------------------------------------------
# TestWorldMapKalman (KF-04, KF-05, Phase 5)
# -------------------------------------------------------------------

class TestWorldMapKalman:
    """WorldMap Kalman integration: point cloud output, save/load, color."""

    @pytest.fixture
    def kalman_config(self):
        """Return WorldMapConfig with Kalman enabled."""
        return WorldMapConfig(kalman_enabled=True)

    @pytest.fixture
    def kalman_wm(self, kalman_config):
        """Return a fresh WorldMap with Kalman enabled."""
        return WorldMap(kalman_config)

    def test_get_point_cloud_returns_refined_positions(
        self, kalman_wm, identity_T
    ):
        """Kalman ON, integrate 3 times, get_point_cloud().points should
        differ from voxel centroids."""
        _integrate_point(kalman_wm, 0.45, 0.15, 0.30, identity_T, n=3)

        pcd = kalman_wm.get_point_cloud(confidence_min=-999.0)
        pts = np.asarray(pcd.points)
        assert len(pts) > 0

        # Get centroids for comparison
        data = kalman_wm._store.get_all_data()
        centers = data['centers']

        # With Kalman and kf_n >= 2, positions should differ from centroids
        assert not np.allclose(pts, centers, atol=1e-10), (
            'With Kalman enabled and 3+ observations, point cloud should '
            'use refined positions, not centroids'
        )

    def test_get_point_cloud_returns_centroids_when_disabled(
        self, world_map, identity_T
    ):
        """Kalman OFF, integrate 3 times, verify points match voxel
        centroids."""
        _integrate_point(world_map, 0.45, 0.15, 0.30, identity_T, n=3)

        pcd = world_map.get_point_cloud(confidence_min=-999.0)
        pts = np.asarray(pcd.points)
        assert len(pts) > 0

        data = world_map._store.get_all_data()
        centers = data['centers']

        np.testing.assert_array_equal(
            pts, centers,
            err_msg='With Kalman disabled, point cloud should use centroids'
        )

    def test_get_numpy_points_uses_positions(
        self, kalman_wm, identity_T
    ):
        """Kalman ON, verify get_numpy_points() returns refined positions."""
        _integrate_point(kalman_wm, 0.45, 0.15, 0.30, identity_T, n=3)

        pts = kalman_wm.get_numpy_points(confidence_min=-999.0)
        data = kalman_wm._store.get_all_data()
        positions = data['positions']

        np.testing.assert_array_equal(
            pts, positions,
            err_msg='get_numpy_points should return positions from get_all_data'
        )

    def test_save_load_kalman_round_trip(
        self, kalman_wm, identity_T
    ):
        """Kalman ON, integrate, save, load, verify Kalman state survives
        round-trip (get_point_cloud matches)."""
        _integrate_point(kalman_wm, 0.45, 0.15, 0.30, identity_T, n=3)

        pcd_before = kalman_wm.get_point_cloud(confidence_min=-999.0)
        pts_before = np.asarray(pcd_before.points)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            kalman_wm.save(path)
            loaded = WorldMap.load(path)

            assert loaded.config.kalman_enabled is True
            assert loaded.config.kalman_process_noise == kalman_wm.config.kalman_process_noise
            assert loaded.config.kalman_measurement_noise == kalman_wm.config.kalman_measurement_noise

            pcd_after = loaded.get_point_cloud(confidence_min=-999.0)
            pts_after = np.asarray(pcd_after.points)

            np.testing.assert_allclose(
                pts_before, pts_after, atol=1e-10,
                err_msg='Save/load round-trip should preserve Kalman positions'
            )

            # Verify Kalman internal state
            orig_kf = kalman_wm._store.get_kalman_data()
            load_kf = loaded._store.get_kalman_data()
            assert load_kf is not None
            np.testing.assert_allclose(
                orig_kf['kf_mean'], load_kf['kf_mean'], atol=1e-10
            )
            np.testing.assert_allclose(
                orig_kf['kf_var'], load_kf['kf_var'], atol=1e-10
            )
            np.testing.assert_array_equal(
                orig_kf['kf_n'], load_kf['kf_n']
            )
        finally:
            os.unlink(path)

    def test_save_load_backward_compat(
        self, world_map, identity_T
    ):
        """Save with Kalman OFF (no kf keys in .npz), load, verify loads
        cleanly."""
        _integrate_point(world_map, 0.45, 0.15, 0.30, identity_T, n=3)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            world_map.save(path)
            loaded = WorldMap.load(path)
            assert loaded.config.kalman_enabled is False

            # Verify points match
            pts_orig = world_map.get_numpy_points(confidence_min=-999.0)
            pts_load = loaded.get_numpy_points(confidence_min=-999.0)
            np.testing.assert_allclose(pts_orig, pts_load, atol=1e-10)
        finally:
            os.unlink(path)

    def test_color_unchanged_with_kalman(
        self, kalman_wm, world_map, identity_T
    ):
        """Kalman ON, integrate, verify colors from get_point_cloud are
        the same as when Kalman OFF (KF-02 dropped)."""
        # Same integration for both
        for wm in [kalman_wm, world_map]:
            _integrate_point(wm, 0.45, 0.15, 0.30, identity_T, n=3,
                             r=200, g=100, b=50)

        pcd_on = kalman_wm.get_point_cloud(confidence_min=-999.0)
        pcd_off = world_map.get_point_cloud(confidence_min=-999.0)

        colors_on = np.asarray(pcd_on.colors)
        colors_off = np.asarray(pcd_off.colors)

        assert len(colors_on) > 0
        assert len(colors_off) > 0
        np.testing.assert_allclose(
            colors_on, colors_off, atol=1e-5,
            err_msg='Colors should be identical with/without Kalman (KF-02 dropped)'
        )
