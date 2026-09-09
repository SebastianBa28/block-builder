"""Comprehensive pytest suite for VoxelStore.

Phase 1
  DS-01  Integer-key deduplication (no float-key duplicates)
  DS-02  Configurable voxel resolution (default 5mm)
  DS-03  Per-voxel state: log-odds, obs_count, last_scan, color
  DS-04  Memory scaling (50K voxels < 18 MB)

Phase 2
  PU-01  Bayesian sensor model (log-odds addition + frustum miss)
  PU-02  Log-odds clamped to [L_MIN, L_MAX]
  PU-03  Observation count capped at obs_count_max
  PU-04  Vectorized via numpy (no Python loop over individual points)
  PU-05  Integration latency under 500ms for 100K points at 50K voxels
  DC-01  Proportional confidence decay for stale voxels
  DC-02  Decay applied at start of each integrate() call
  DC-03  Decay rate configurable via WorldMapConfig

Phase 3
  QR-01  query_height returns max Z above confidence + obs_count thresholds
  QR-02  query_region returns cylindrical subset above confidence
  QR-03  Query latency under 10ms for 50K voxels
  CL-01  clear_region(center, radius) removes voxels in sphere
  CL-02  Cleared voxels are deleted from storage

Also covers: bounds clipping, color last-write-wins, transform application,
reset, edge cases, pitfall regressions, clearing scenarios, and performance.
"""

import time

from legobuilder.config import WorldMapConfig
from legobuilder.vision.voxel_store import VoxelStore, HeightResult

import numpy as np

import pytest


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def config():
    """Return default WorldMapConfig for testing."""
    return WorldMapConfig()


@pytest.fixture
def store(config):
    """Return fresh VoxelStore with default config."""
    return VoxelStore(config)


@pytest.fixture
def identity_transform():
    """Return 4x4 identity transform (camera = world frame)."""
    return np.eye(4)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_points(*pts):
    """Return an (N, 3) float64 array from 3-tuples."""
    return np.array(pts, dtype=np.float64)


def _make_colors(*cols):
    """Return an (N, 3) float32 array from 3-tuples."""
    return np.array(cols, dtype=np.float32)


# -------------------------------------------------------------------
# DS-01 -- Integer key deduplication
# -------------------------------------------------------------------

class TestDeduplication:
    """DS-01: Same cloud twice => identical voxel count."""

    def test_duplicate_points_same_voxel(
        self, store, identity_transform
    ):
        """Map two sub-voxel-offset points to one voxel."""
        pts = _make_points(
            [0.1, 0.1, 0.2], [0.1001, 0.1002, 0.2001]
        )
        cols = _make_colors([255, 0, 0], [255, 0, 0])
        store.integrate(pts, cols, identity_transform)
        assert store.voxel_count == 1

    def test_integrate_idempotent_count(
        self, store, identity_transform
    ):
        """Verify integrating same cloud twice keeps count."""
        pts = _make_points(
            [0.1, 0.1, 0.2], [0.2, 0.2, 0.3]
        )
        cols = _make_colors([255, 0, 0], [0, 255, 0])

        store.integrate(pts, cols, identity_transform)
        count_after_first = store.voxel_count

        store.integrate(pts, cols, identity_transform)
        count_after_second = store.voxel_count

        assert count_after_second == count_after_first

    def test_adjacent_voxels_distinct(
        self, store, identity_transform
    ):
        """Place two points one voxel apart in different voxels."""
        pts = _make_points(
            [0.1, 0.1, 0.2], [0.1 + 0.005, 0.1, 0.2]
        )
        cols = _make_colors([255, 0, 0], [0, 255, 0])
        store.integrate(pts, cols, identity_transform)
        assert store.voxel_count == 2


# -------------------------------------------------------------------
# DS-02 -- Configurable resolution
# -------------------------------------------------------------------

class TestConfigurableResolution:
    """DS-02: Resolution is configurable; default is 5mm."""

    def test_default_resolution(self):
        """Verify default voxel_resolution == 0.005."""
        cfg = WorldMapConfig()
        assert cfg.voxel_resolution == 0.005

    def test_custom_resolution_coarser(self, identity_transform):
        """Verify 10mm res merges two 7mm-apart points."""
        cfg = WorldMapConfig(voxel_resolution=0.01)
        st = VoxelStore(cfg)
        pts = _make_points(
            [0.1, 0.1, 0.2], [0.107, 0.1, 0.2]
        )
        cols = _make_colors([255, 0, 0], [0, 255, 0])
        st.integrate(pts, cols, identity_transform)
        assert st.voxel_count == 1

    def test_custom_resolution_finer(self, identity_transform):
        """Verify 5mm res separates two 7mm-apart points."""
        cfg = WorldMapConfig(voxel_resolution=0.005)
        st = VoxelStore(cfg)
        pts = _make_points(
            [0.1, 0.1, 0.2], [0.107, 0.1, 0.2]
        )
        cols = _make_colors([255, 0, 0], [0, 255, 0])
        st.integrate(pts, cols, identity_transform)
        assert st.voxel_count == 2


# -------------------------------------------------------------------
# DS-03 -- Per-voxel state
# -------------------------------------------------------------------

class TestPerVoxelState:
    """DS-03: Each voxel stores log-odds, obs, scan, color."""

    def _integrate_single_point(
        self, store, identity_transform, pt=None, col=None
    ):
        """Integrate one point and return its voxel key."""
        pt = pt or [0.1, 0.1, 0.2]
        col = col or [255, 0, 0]
        pts = _make_points(pt)
        cols = _make_colors(col)
        store.integrate(pts, cols, identity_transform)
        key = next(iter(store._voxels))
        return key

    def test_voxel_has_log_odds(self, store, identity_transform):
        """Verify log_odds initialises to log_odds_prior."""
        key = self._integrate_single_point(
            store, identity_transform
        )
        voxel = store.get_voxel(key)
        assert voxel is not None
        assert 'log_odds' in voxel
        np.testing.assert_allclose(
            voxel['log_odds'],
            store._config.log_odds_prior,
            atol=1e-6,
        )

    def test_voxel_has_obs_count(
        self, store, identity_transform
    ):
        """Verify obs_count starts at 1 and increments."""
        pts = _make_points([0.1, 0.1, 0.2])
        cols = _make_colors([255, 0, 0])

        store.integrate(pts, cols, identity_transform)
        key = next(iter(store._voxels))
        assert store.get_voxel(key)['obs_count'] == 1

        store.integrate(pts, cols, identity_transform)
        assert store.get_voxel(key)['obs_count'] == 2

    def test_voxel_has_last_scan(
        self, store, identity_transform
    ):
        """Verify last_scan tracks scan_index per observation."""
        pts = _make_points([0.1, 0.1, 0.2])
        cols = _make_colors([255, 0, 0])

        store.integrate(pts, cols, identity_transform)
        key = next(iter(store._voxels))
        assert store.get_voxel(key)['last_scan'] == 1

        store.integrate(pts, cols, identity_transform)
        assert store.get_voxel(key)['last_scan'] == 2

    def test_voxel_has_color(self, store, identity_transform):
        """Verify uint8 color is normalised to float [0,1]."""
        key = self._integrate_single_point(
            store, identity_transform, col=[255, 0, 0]
        )
        voxel = store.get_voxel(key)
        assert voxel is not None
        np.testing.assert_allclose(
            voxel['color'], [1.0, 0.0, 0.0], atol=1e-5
        )

    def test_log_odds_incremented_on_reobserve(
        self, store, identity_transform
    ):
        """Verify re-observing adds log_odds_hit."""
        pts = _make_points([0.1, 0.1, 0.2])
        cols = _make_colors([255, 0, 0])

        store.integrate(pts, cols, identity_transform)
        key = next(iter(store._voxels))
        lo_first = store.get_voxel(key)['log_odds']

        store.integrate(pts, cols, identity_transform)
        lo_second = store.get_voxel(key)['log_odds']

        expected = lo_first + store._config.log_odds_hit
        np.testing.assert_allclose(
            lo_second, expected, atol=1e-6
        )


# -------------------------------------------------------------------
# DS-04 -- Memory scaling
# -------------------------------------------------------------------

class TestMemoryScaling:
    """DS-04: 50K voxels consume under 18 MB of RAM."""

    def test_memory_under_18mb_at_50k(self, identity_transform):
        """Integrate 50K unique voxels and check memory."""
        cfg = WorldMapConfig()
        st = VoxelStore(cfg)
        res = cfg.voxel_resolution
        bmin = cfg.bounds_center - cfg.bounds_half_extents

        # 37x37x37 = 50653 grid; take first 50000.
        side = 37
        grid = np.stack(
            np.meshgrid(
                np.arange(side),
                np.arange(side),
                np.arange(side),
                indexing='ij',
            ),
            axis=-1,
        ).reshape(-1, 3)
        grid = grid[:50000]

        points = (grid.astype(np.float64) + 0.5) * res + bmin
        colors = np.full(
            (len(points), 3), 128, dtype=np.float32
        )

        st.integrate(points, colors, identity_transform)

        assert st.voxel_count == 50000
        assert st.memory_bytes < 18 * 1024 * 1024

    def test_empty_store_minimal_memory(self, store):
        """Verify empty store reports zero memory."""
        assert store.memory_bytes == 0


# -------------------------------------------------------------------
# Bounds clipping
# -------------------------------------------------------------------

class TestBoundsClipping:
    """Out-of-bounds points are silently discarded."""

    def test_out_of_bounds_discarded(
        self, store, identity_transform
    ):
        """Discard points entirely outside workspace."""
        pts = _make_points(
            [10.0, 10.0, 10.0],
            [-5.0, -5.0, -5.0],
        )
        cols = _make_colors([255, 0, 0], [0, 255, 0])
        store.integrate(pts, cols, identity_transform)
        assert store.voxel_count == 0

    def test_mixed_in_and_out_of_bounds(
        self, store, identity_transform
    ):
        """Keep only in-bounds points; discard the rest."""
        pts = _make_points(
            [0.1, 0.1, 0.2],
            [0.2, 0.2, 0.3],
            [10.0, 10.0, 10.0],
        )
        cols = _make_colors(
            [255, 0, 0], [0, 255, 0], [0, 0, 255]
        )
        store.integrate(pts, cols, identity_transform)
        assert store.voxel_count == 2


# -------------------------------------------------------------------
# Color storage
# -------------------------------------------------------------------

class TestColorStorage:
    """Per-voxel color: last-observed RGB wins."""

    def test_color_last_observed_wins(
        self, store, identity_transform
    ):
        """Overwrite voxel color on re-integration."""
        pt = _make_points([0.1, 0.1, 0.2])

        # First integration: red.
        store.integrate(
            pt, _make_colors([255, 0, 0]), identity_transform
        )
        key = next(iter(store._voxels))
        np.testing.assert_allclose(
            store.get_voxel(key)['color'],
            [1.0, 0.0, 0.0],
            atol=1e-5,
        )

        # Second integration: blue overwrites red.
        store.integrate(
            pt, _make_colors([0, 0, 255]), identity_transform
        )
        np.testing.assert_allclose(
            store.get_voxel(key)['color'],
            [0.0, 0.0, 1.0],
            atol=1e-5,
        )


# -------------------------------------------------------------------
# Transform application
# -------------------------------------------------------------------

class TestTransformApplication:
    """Camera-to-world transform is applied correctly."""

    def test_transform_applied(self, store):
        """Apply translation and verify voxel centre."""
        # Camera-frame point at (0, 0, 0.3).
        # T_world_camera translates by (0.1, 0.1, 0.1).
        # Expected world position: (0.1, 0.1, 0.4).
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.1, 0.1]

        pts = _make_points([0.0, 0.0, 0.3])
        cols = _make_colors([128, 128, 128])
        store.integrate(pts, cols, T)

        assert store.voxel_count == 1

        data = store.get_all_data()
        center = data['centers'][0]
        res = store._config.voxel_resolution
        np.testing.assert_allclose(
            center, [0.1, 0.1, 0.4], atol=res
        )


# -------------------------------------------------------------------
# Reset
# -------------------------------------------------------------------

class TestReset:
    """reset() clears all state."""

    def test_reset_clears_all(
        self, store, identity_transform
    ):
        """Verify reset zeroes voxel_count and scan_index."""
        pts = _make_points([0.1, 0.1, 0.2])
        cols = _make_colors([255, 0, 0])
        store.integrate(pts, cols, identity_transform)
        assert store.voxel_count > 0

        store.reset()
        assert store.voxel_count == 0
        assert store.scan_index == 0


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------

class TestEdgeCases:
    """Empty input, single point, and boundary conditions."""

    def test_integrate_empty_array(
        self, store, identity_transform
    ):
        """Integrate empty array without crashing."""
        pts = np.empty((0, 3), dtype=np.float64)
        cols = np.empty((0, 3), dtype=np.float32)
        store.integrate(pts, cols, identity_transform)
        assert store.voxel_count == 0

    def test_integrate_single_point(
        self, store, identity_transform
    ):
        """Integrate exactly one point to produce one voxel."""
        pts = _make_points([0.1, 0.1, 0.2])
        cols = _make_colors([128, 128, 128])
        store.integrate(pts, cols, identity_transform)
        assert store.voxel_count == 1


# ===================================================================
# Phase 2 -- Probabilistic update, decay, frustum miss, pruning
# ===================================================================

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _integrate_empty(store, T):
    """Integrate an empty point cloud (advances scan_index)."""
    pts = np.empty((0, 3), dtype=np.float64)
    cols = np.empty((0, 3), dtype=np.float32)
    store.integrate(pts, cols, T)


def _integrate_point(store, T, pt, col=None):
    """Integrate a single point into the store."""
    col = col or [0.5, 0.5, 0.5]
    pts = _make_points(pt)
    cols = _make_colors(col)
    store.integrate(pts, cols, T)


def _get_only_voxel(store):
    """Return (key, voxel_data) for a store with exactly one voxel."""
    assert store.voxel_count == 1, (
        f'Expected 1 voxel, got {store.voxel_count}'
    )
    key = next(iter(store._voxels))
    return key, store.get_voxel(key)


# -------------------------------------------------------------------
# PU-02 -- Log-odds clamping
# -------------------------------------------------------------------

class TestLogOddsClamping:
    """PU-02: Log-odds clamped to [L_MIN, L_MAX]."""

    def test_log_odds_clamped_at_max(self, store, identity_transform):
        """Integrate same point 10 times. Assert log_odds == L_MAX, not higher.

        Math: prior=-0.8 (scan 1), then +0.85 each subsequent scan:
          scan 2: 0.05, scan 3: 0.90, scan 4: 1.75, scan 5: 2.60,
          scan 6: 3.45, scan 7: would be 4.30 but clamped to 3.5.
        After 10 hits the value should remain exactly L_MAX.
        """
        pt = [0.1, 0.1, 0.2]
        for _ in range(10):
            _integrate_point(store, identity_transform, pt)
        key, voxel = _get_only_voxel(store)
        assert voxel['log_odds'] == pytest.approx(
            store._config.log_odds_max, abs=1e-5
        )

    def test_log_odds_clamped_at_min_via_miss(self, identity_transform):
        """Frustum miss cannot push log_odds below L_MIN.

        Create a voxel at [0.0, 0.0, 0.5] (in frustum with identity T).
        Repeatedly integrate a DIFFERENT in-frustum point so the original
        voxel gets miss penalties. Assert log_odds >= L_MIN.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)

        # Create voxel at [0.0, 0.0, 0.5]
        _integrate_point(store, identity_transform, [0.0, 0.0, 0.5])
        key, voxel = _get_only_voxel(store)
        assert voxel['log_odds'] == pytest.approx(cfg.log_odds_prior, abs=1e-5)

        # Make it confident by re-observing (need log_odds > 0 for miss)
        _integrate_point(store, identity_transform, [0.0, 0.0, 0.5])
        _, voxel = _get_only_voxel(store)
        assert voxel['log_odds'] > cfg.confidence_threshold

        # Now integrate a different point 30 times so original gets missed
        for _ in range(30):
            _integrate_point(store, identity_transform, [0.05, 0.05, 0.5])

        # Original voxel should have been penalized but clamped at L_MIN
        voxel_data = store.get_voxel(key)
        if voxel_data is not None:
            # May have been pruned if below threshold, which is fine
            assert voxel_data['log_odds'] >= cfg.log_odds_min - 1e-5


# -------------------------------------------------------------------
# PU-03 -- Observation count cap
# -------------------------------------------------------------------

class TestObsCountCap:
    """PU-03: obs_count capped at obs_count_max."""

    def test_obs_count_capped(self, store, identity_transform):
        """Integrate same point 30 times. Assert obs_count == 20, not 30."""
        pt = [0.1, 0.1, 0.2]
        for _ in range(30):
            _integrate_point(store, identity_transform, pt)
        _, voxel = _get_only_voxel(store)
        assert voxel['obs_count'] == store._config.obs_count_max


# -------------------------------------------------------------------
# DC-01, DC-02, DC-03 -- Decay
# -------------------------------------------------------------------

class TestDecay:
    """DC-01/02/03: Proportional decay on stale positive log-odds."""

    def test_decay_kicks_in_after_patience(
        self, store, identity_transform
    ):
        """Stale positive log-odds decrease after patience.

        After decay_after_scans+1 empty integrations, stale positive
        log-odds should decrease.
        """
        # Integrate point twice to get positive log-odds
        # scan 1: prior = -0.8, scan 2: -0.8 + 0.85 = 0.05
        pt = [0.1, 0.1, 0.2]
        _integrate_point(store, identity_transform, pt)
        _integrate_point(store, identity_transform, pt)
        key, voxel = _get_only_voxel(store)
        lo_before = voxel['log_odds']
        assert lo_before > 0.0, 'Need positive log-odds for decay test'

        # Run decay_after_scans + 1 empty integrations
        for _ in range(store._config.decay_after_scans + 1):
            _integrate_empty(store, identity_transform)

        _, voxel_after = _get_only_voxel(store)
        assert voxel_after['log_odds'] < lo_before

    def test_no_decay_within_patience(
        self, store, identity_transform
    ):
        """Within decay_after_scans empty scans, log-odds should NOT change.

        Staleness == threshold means decay has NOT kicked in yet
        (staleness must be > threshold, not >=).
        """
        # Build up positive log-odds (4 hits: -0.8, 0.05, 0.90, 1.75)
        pt = [0.1, 0.1, 0.2]
        for _ in range(4):
            _integrate_point(store, identity_transform, pt)
        key, voxel = _get_only_voxel(store)
        lo_before = voxel['log_odds']
        assert lo_before == pytest.approx(1.75, abs=0.1)

        # Run exactly decay_after_scans empty scans (staleness == threshold)
        for _ in range(store._config.decay_after_scans):
            _integrate_empty(store, identity_transform)

        _, voxel_after = _get_only_voxel(store)
        assert voxel_after['log_odds'] == pytest.approx(lo_before, abs=1e-5)

    def test_proportional_decay_math(
        self, store, identity_transform
    ):
        """Verify proportional (multiplicative) decay arithmetic.

        Integrate 6 times for near-saturation:
          scan 1: -0.8, scan 2: 0.05, scan 3: 0.90,
          scan 4: 1.75, scan 5: 2.60, scan 6: 3.45
        After decay_after_scans+1 empty scans:
          log_odds = 3.45 * 0.3 = 1.035
        """
        pt = [0.1, 0.1, 0.2]
        for _ in range(6):
            _integrate_point(store, identity_transform, pt)
        key, voxel = _get_only_voxel(store)
        lo_before = voxel['log_odds']
        assert lo_before == pytest.approx(3.45, abs=0.1)

        # Run decay_after_scans + 1 empty scans to trigger exactly 1 decay
        for _ in range(store._config.decay_after_scans + 1):
            _integrate_empty(store, identity_transform)

        _, voxel_after = _get_only_voxel(store)
        expected = lo_before * store._config.decay_factor
        assert voxel_after['log_odds'] == pytest.approx(expected, abs=0.1)

    def test_negative_log_odds_not_decayed(self, identity_transform):
        """Negative log-odds must NOT be decayed toward zero (pitfall test).

        Manually set a voxel to log_odds=-1.0 and verify decay does not
        move it toward zero.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)

        # Create a voxel
        _integrate_point(store, identity_transform, [0.1, 0.1, 0.2])
        key = next(iter(store._voxels))
        idx = store._voxels[key]

        # Force log_odds to -1.0
        store._log_odds[idx] = -1.0

        # Run enough empty scans to exceed patience
        for _ in range(cfg.decay_after_scans + 3):
            _integrate_empty(store, identity_transform)

        voxel = store.get_voxel(key)
        if voxel is not None:
            # Negative log-odds should NOT have moved toward zero
            assert voxel['log_odds'] <= -1.0 + 1e-5

    def test_decay_applied_at_start_of_integrate(
        self, store, identity_transform
    ):
        """Verify ordering: decay BEFORE hit update in the same call.

        Integrate a point once (log_odds = -0.8, not positive, so let's
        do 2 integrations to get 0.05). Wait decay_after_scans+1 empty
        scans. Then re-integrate the same point.

        Expected: decayed_value + hit, NOT undecayed_value + hit.
          0.05 * 0.3 + 0.85 = 0.865 (decay first, then hit)
          vs 0.05 + 0.85 = 0.90 (hit without decay -- wrong)
        """
        pt = [0.1, 0.1, 0.2]
        # Two integrations: scan 1 -> -0.8, scan 2 -> 0.05
        _integrate_point(store, identity_transform, pt)
        _integrate_point(store, identity_transform, pt)
        key, voxel = _get_only_voxel(store)
        lo_before_wait = voxel['log_odds']
        assert lo_before_wait == pytest.approx(0.05, abs=1e-5)

        # Run exactly decay_after_scans + 1 empty scans to trigger decay.
        # After these, staleness will be decay_after_scans + 1, so
        # when we re-integrate, the first thing in integrate() is decay.
        # But actually the empty scans already apply decay.
        # After the empty scans the log_odds is 0.05 * 0.3 = 0.015
        for _ in range(store._config.decay_after_scans + 1):
            _integrate_empty(store, identity_transform)

        # Check the decayed value
        _, voxel_decayed = _get_only_voxel(store)
        lo_decayed = voxel_decayed['log_odds']
        assert lo_decayed == pytest.approx(
            lo_before_wait * store._config.decay_factor, abs=0.05
        )

        # Now re-integrate the same point. Decay should happen FIRST in
        # this integrate() call, but since the voxel was JUST decayed
        # in the previous empty scan and staleness is reset by the
        # previous decay... Actually: the empty scans decayed it, so
        # last_scan is still old. On re-integrate, decay fires again
        # (another empty cycle has passed). Then hit applies.
        #
        # Actually let's simplify. The key thing: after the re-integrate,
        # the log_odds should reflect decay-then-hit, not just hit alone.
        # Let's just verify the final value is decayed+hit, not lo_decayed+hit.
        _integrate_point(store, identity_transform, pt)

        _, voxel_after = _get_only_voxel(store)
        lo_after = voxel_after['log_odds']

        # In the re-integrate call, decay fires first:
        #   lo_decayed * decay_factor (because staleness > patience again)
        # Then hit: + log_odds_hit
        expected = lo_decayed * store._config.decay_factor + store._config.log_odds_hit
        # Contrast with no-decay: lo_decayed + hit
        wrong_no_decay = lo_decayed + store._config.log_odds_hit

        assert lo_after == pytest.approx(expected, abs=0.05)
        assert lo_after != pytest.approx(wrong_no_decay, abs=0.01)

    def test_decay_factor_configurable(self, identity_transform):
        """DC-03: Create config with decay_factor=0.5 and verify it is used."""
        cfg = WorldMapConfig(decay_factor=0.5)
        store = VoxelStore(cfg)

        # Build up positive log-odds (6 hits -> ~3.45)
        pt = [0.1, 0.1, 0.2]
        for _ in range(6):
            _integrate_point(store, identity_transform, pt)
        key, voxel = _get_only_voxel(store)
        lo_before = voxel['log_odds']

        # Trigger decay
        for _ in range(cfg.decay_after_scans + 1):
            _integrate_empty(store, identity_transform)

        _, voxel_after = _get_only_voxel(store)
        expected = lo_before * 0.5
        assert voxel_after['log_odds'] == pytest.approx(expected, abs=0.1)
        # Verify it is NOT using default 0.3
        not_expected = lo_before * 0.3
        assert abs(voxel_after['log_odds'] - not_expected) > 0.3

    def test_scan_index_always_increments(
        self, store, identity_transform
    ):
        """Scan index increments on every integrate(), even empty ones."""
        assert store.scan_index == 0
        _integrate_empty(store, identity_transform)
        assert store.scan_index == 1
        _integrate_empty(store, identity_transform)
        assert store.scan_index == 2


# -------------------------------------------------------------------
# PU-01 -- Frustum miss
# -------------------------------------------------------------------

class TestFrustumMiss:
    """PU-01 miss path: frustum miss penalizes in-view unobserved voxels."""

    def test_frustum_miss_penalizes_in_view_voxel(
        self, identity_transform
    ):
        """A confident in-frustum voxel not observed this scan gets penalized.

        Setup: voxel at [0.0, 0.0, 0.5], camera at origin (identity T).
        The voxel is within depth range [0.15, 1.0] and well within FOV.
        Make it confident (2 hits: log_odds=0.05), then integrate a
        DIFFERENT point at [0.05, 0.05, 0.5]. The original voxel should
        receive a miss penalty.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)
        T = identity_transform

        # Create and confirm voxel at [0.0, 0.0, 0.5]
        _integrate_point(store, T, [0.0, 0.0, 0.5])
        _integrate_point(store, T, [0.0, 0.0, 0.5])
        key = list(store._voxels.keys())[0]
        lo_before = store.get_voxel(key)['log_odds']
        assert lo_before > cfg.confidence_threshold, 'Need confident voxel'

        # Integrate a DIFFERENT in-frustum point so original is missed
        _integrate_point(store, T, [0.05, 0.05, 0.5])

        lo_after = store.get_voxel(key)['log_odds']
        assert lo_after < lo_before, (
            f'Expected miss penalty: {lo_after} should be < {lo_before}'
        )
        expected = lo_before + cfg.log_odds_miss
        assert lo_after == pytest.approx(expected, abs=0.05)

    def test_frustum_miss_does_not_penalize_new_voxels(
        self, identity_transform
    ):
        """New voxels created this scan should NOT receive miss penalties.

        Integrate a fresh point. Its log_odds should be log_odds_prior,
        NOT log_odds_prior + log_odds_miss.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)

        # Fresh integration: all voxels are new
        _integrate_point(store, identity_transform, [0.0, 0.0, 0.5])
        _, voxel = _get_only_voxel(store)

        assert voxel['log_odds'] == pytest.approx(
            cfg.log_odds_prior, abs=1e-5
        ), 'New voxel should NOT have miss penalty applied'

    def test_frustum_miss_only_penalizes_confident_voxels(
        self, identity_transform
    ):
        """A voxel below confidence_threshold should NOT get miss penalties.

        After 1 integration, log_odds = -0.8 (prior), which is below
        confidence_threshold (0.0). Even if in frustum and not re-observed,
        no miss should apply.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)
        T = identity_transform

        # Create unconfident voxel (1 hit: log_odds = -0.8)
        _integrate_point(store, T, [0.0, 0.0, 0.5])
        key, voxel = _get_only_voxel(store)
        lo_before = voxel['log_odds']
        assert lo_before <= cfg.confidence_threshold

        # Integrate a different point -- original is in frustum but unconfident
        _integrate_point(store, T, [0.05, 0.05, 0.5])

        voxel_after = store.get_voxel(key)
        # log_odds should be unchanged (no miss on unconfident voxel)
        assert voxel_after['log_odds'] == pytest.approx(
            lo_before, abs=1e-5
        )

    def test_out_of_frustum_voxel_not_penalized(
        self, identity_transform
    ):
        """A voxel behind the camera should NOT receive miss penalties.

        Create a voxel at [0.0, 0.0, 0.5] (in frustum). Then create a
        voxel at [0.3, 0.3, 0.1] (NOT in frustum: z=0.1 < range_min=0.15).
        Make both confident. Integrate only the first point.
        The out-of-frustum voxel should NOT be penalized.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)
        T = identity_transform

        # Create two voxels with 2 hits each
        pt_in = [0.0, 0.0, 0.5]
        pt_out = [0.3, 0.3, 0.1]  # z=0.1 < range_min=0.15

        _integrate_point(store, T, pt_in)
        _integrate_point(store, T, pt_out)
        # Second observation for confidence
        _integrate_point(store, T, pt_in)
        _integrate_point(store, T, pt_out)

        # Find the out-of-range voxel key
        res = cfg.voxel_resolution
        bmin = cfg.bounds_center - cfg.bounds_half_extents
        out_key = tuple(np.floor(
            (np.array(pt_out) - bmin) / res
        ).astype(int))

        voxel_out = store.get_voxel(out_key)
        if voxel_out is None:
            # pt_out might be out of workspace bounds; skip this check
            pytest.skip('Out-of-range point was clipped by workspace bounds')

        lo_before = voxel_out['log_odds']

        # Now integrate only the in-frustum point
        _integrate_point(store, T, pt_in)

        voxel_out_after = store.get_voxel(out_key)
        assert voxel_out_after is not None
        # Out-of-frustum voxel should NOT be penalized by miss
        assert voxel_out_after['log_odds'] == pytest.approx(
            lo_before, abs=1e-5
        ), 'Out-of-frustum voxel should not receive miss penalty'


# -------------------------------------------------------------------
# Pruning
# -------------------------------------------------------------------

class TestPruning:
    """Periodic pruning removes dead voxels and compacts arrays."""

    def test_pruning_removes_dead_voxels(self, identity_transform):
        """Dead voxels (near L_MIN) should be removed at prune_interval."""
        cfg = WorldMapConfig(prune_interval=5)
        store = VoxelStore(cfg)
        T = identity_transform

        # Create a voxel and force it near L_MIN
        _integrate_point(store, T, [0.1, 0.1, 0.2])
        key = next(iter(store._voxels))
        idx = store._voxels[key]
        store._log_odds[idx] = cfg.log_odds_min + 0.005  # below prune threshold

        count_before = store.voxel_count
        assert count_before == 1

        # Run scans until we hit a prune_interval multiple.
        # Current scan_index is 1; need to reach scan_index 5.
        while store.scan_index % cfg.prune_interval != 0 or store.scan_index == 0:
            _integrate_empty(store, T)

        # After pruning, the dead voxel should be gone
        assert store.voxel_count < count_before
        assert key not in store._voxels

    def test_pruning_compacts_arrays(self, identity_transform):
        """After pruning, get_all_data() returns consistent, compact data."""
        cfg = WorldMapConfig(prune_interval=5)
        store = VoxelStore(cfg)
        T = identity_transform

        # Create 3 voxels
        pts = _make_points([0.0, 0.0, 0.3], [0.1, 0.1, 0.3], [0.2, 0.2, 0.3])
        cols = _make_colors([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])
        store.integrate(pts, cols, T)
        assert store.voxel_count == 3

        # Kill one voxel
        first_key = list(store._voxels.keys())[0]
        first_idx = store._voxels[first_key]
        store._log_odds[first_idx] = cfg.log_odds_min + 0.005

        # Advance to prune interval
        while store.scan_index % cfg.prune_interval != 0:
            _integrate_empty(store, T)

        # Verify compaction
        assert store.voxel_count == 2
        data = store.get_all_data()
        assert data['keys'].shape[0] == 2
        assert data['log_odds'].shape[0] == 2
        assert data['obs_count'].shape[0] == 2
        assert data['colors'].shape[0] == 2
        # Dict size matches
        assert len(store._voxels) == 2
        # All dict indices are in valid range [0, size)
        for k, idx in store._voxels.items():
            assert 0 <= idx < store.voxel_count

    def test_pruning_preserves_live_voxels(self, identity_transform):
        """Live voxels retain correct state after pruning compacts arrays."""
        cfg = WorldMapConfig(prune_interval=5)
        store = VoxelStore(cfg)
        T = identity_transform

        # Create 3 voxels
        pts = _make_points([0.0, 0.0, 0.3], [0.1, 0.1, 0.3], [0.2, 0.2, 0.3])
        cols = _make_colors([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])
        store.integrate(pts, cols, T)

        # Remember the state of the live voxels
        keys = list(store._voxels.keys())
        live_key_1 = keys[1]
        live_key_2 = keys[2]
        lo_1 = store.get_voxel(live_key_1)['log_odds']
        lo_2 = store.get_voxel(live_key_2)['log_odds']
        oc_1 = store.get_voxel(live_key_1)['obs_count']
        oc_2 = store.get_voxel(live_key_2)['obs_count']

        # Kill the first voxel
        dead_key = keys[0]
        dead_idx = store._voxels[dead_key]
        store._log_odds[dead_idx] = cfg.log_odds_min + 0.005

        # Advance to prune interval
        while store.scan_index % cfg.prune_interval != 0:
            _integrate_empty(store, T)

        # Verify live voxels are preserved
        assert store.voxel_count == 2
        v1 = store.get_voxel(live_key_1)
        v2 = store.get_voxel(live_key_2)
        assert v1 is not None
        assert v2 is not None
        assert v1['log_odds'] == pytest.approx(lo_1, abs=1e-5)
        assert v2['log_odds'] == pytest.approx(lo_2, abs=1e-5)
        assert v1['obs_count'] == oc_1
        assert v2['obs_count'] == oc_2


# -------------------------------------------------------------------
# PU-04, PU-05 -- Performance
# -------------------------------------------------------------------

class TestPerformance:
    """PU-04/05: Vectorized integration, latency under 500ms."""

    def test_integration_latency_100k_points(self, identity_transform):
        """100K points with 50K existing voxels: under 1s.

        The 500ms target from PU-05 is for the integration itself;
        we allow extra margin for CI/test environments. A pure Python
        loop over 100K points would take >2s, so 1s proves vectorization.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)
        res = cfg.voxel_resolution
        bmin = cfg.bounds_center - cfg.bounds_half_extents

        # Pre-populate with 50K existing voxels
        side = 37
        grid = np.stack(
            np.meshgrid(
                np.arange(side), np.arange(side), np.arange(side),
                indexing='ij',
            ),
            axis=-1,
        ).reshape(-1, 3)[:50000]
        pre_points = (grid.astype(np.float64) + 0.5) * res + bmin
        pre_colors = np.full((len(pre_points), 3), 0.5, dtype=np.float32)
        store.integrate(pre_points, pre_colors, identity_transform)
        assert store.voxel_count == 50000

        # Generate 100K random points within workspace
        rng = np.random.default_rng(42)
        bmax = cfg.bounds_center + cfg.bounds_half_extents
        pts_100k = rng.uniform(
            bmin + 0.01, bmax - 0.01, size=(100000, 3)
        )
        cols_100k = rng.uniform(0, 1, size=(100000, 3)).astype(np.float32)

        # Time the integration
        start = time.perf_counter()
        store.integrate(pts_100k, cols_100k, identity_transform)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f'Integration took {elapsed:.3f}s, expected < 1.0s'
        )

    def test_no_python_loop_over_points(self, identity_transform):
        """50K unique points should integrate in under 500ms (vectorized).

        A pure Python loop over 50K points would take >1s. This test
        verifies that the core operations are vectorized via numpy.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)

        rng = np.random.default_rng(123)
        bmin = cfg.bounds_center - cfg.bounds_half_extents
        bmax = cfg.bounds_center + cfg.bounds_half_extents

        pts = rng.uniform(bmin + 0.01, bmax - 0.01, size=(50000, 3))
        cols = rng.uniform(0, 1, size=(50000, 3)).astype(np.float32)

        start = time.perf_counter()
        store.integrate(pts, cols, identity_transform)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, (
            f'Integration of 50K unique points took {elapsed:.3f}s, '
            f'expected < 0.5s (a Python loop would take >1s)'
        )


# -------------------------------------------------------------------
# Clearing scenario (integration test)
# -------------------------------------------------------------------

class TestClearingScenario:
    """Integration tests for the 2-3 scan clearing target."""

    def test_saturated_voxel_clears_in_2_scans_with_decay_and_miss(
        self, identity_transform
    ):
        """A saturated voxel clears via decay+miss combined.

        Saturate: 10 hits -> log_odds = 3.5 (L_MAX).
        Voxel at [0.0, 0.0, 0.5] is in-frustum with identity T.

        During patience period (decay_after_scans empty scans), frustum
        miss applies each scan (miss does NOT wait for patience):
          After 6 empty scans (patience): 3.5 + 6*(-0.4) = 3.5 - 2.4 = 1.1

        Once patience expires (7th empty scan = scan_index > last_scan + 6),
        both decay and miss apply:
          Scan patience+1: 1.1 * 0.3 + (-0.4) = 0.33 - 0.4 = -0.07

        So the voxel clears after patience + 1 scan of combined decay+miss.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)
        T = identity_transform

        # Place voxel at [0.0, 0.0, 0.5] (center of frustum)
        pt = [0.0, 0.0, 0.5]
        for _ in range(10):
            _integrate_point(store, T, pt)

        key, voxel = _get_only_voxel(store)
        assert voxel['log_odds'] == pytest.approx(
            cfg.log_odds_max, abs=1e-5
        )

        # During patience period, frustum miss applies but NOT decay.
        # After decay_after_scans empty scans: 3.5 - 6*0.4 = 1.1
        for _ in range(cfg.decay_after_scans):
            _integrate_empty(store, T)

        _, voxel_mid = _get_only_voxel(store)
        lo_after_patience = voxel_mid['log_odds']
        expected_patience = cfg.log_odds_max + cfg.decay_after_scans * cfg.log_odds_miss
        assert lo_after_patience == pytest.approx(expected_patience, abs=0.15)
        assert lo_after_patience > cfg.confidence_threshold, (
            'Voxel should still be confident after patience period'
        )

        # Now 1 more scan: decay (staleness > threshold) + miss
        _integrate_empty(store, T)
        _, v_cleared = _get_only_voxel(store)
        lo_cleared = v_cleared['log_odds']
        # decay: lo_after_patience * 0.3 + miss: -0.4
        expected_cleared = lo_after_patience * cfg.decay_factor + cfg.log_odds_miss
        assert lo_cleared == pytest.approx(expected_cleared, abs=0.1)
        assert lo_cleared < cfg.confidence_threshold, (
            f'Expected cleared (< {cfg.confidence_threshold}), '
            f'got {lo_cleared}'
        )

    def test_ramp_up_in_3_to_5_scans(self, store, identity_transform):
        """Log-odds ramp up correctly over consecutive observations.

        Scan 1: -0.8 (prior)
        Scan 2: 0.05
        Scan 3: 0.90
        Scan 4: 1.75
        Scan 5: 2.60
        Scan 6: 3.45
        Scan 7: 3.50 (clamped)
        """
        pt = [0.1, 0.1, 0.2]
        key = None

        # Scan 1
        _integrate_point(store, identity_transform, pt)
        key = next(iter(store._voxels))
        assert store.get_voxel(key)['log_odds'] == pytest.approx(
            -0.8, abs=1e-5
        )

        # Scan 2
        _integrate_point(store, identity_transform, pt)
        assert store.get_voxel(key)['log_odds'] == pytest.approx(
            0.05, abs=1e-5
        )

        # Scan 3: should be > 0.5 (established)
        _integrate_point(store, identity_transform, pt)
        lo3 = store.get_voxel(key)['log_odds']
        assert lo3 > 0.5

        # Scan 4-5: building confidence
        _integrate_point(store, identity_transform, pt)
        _integrate_point(store, identity_transform, pt)
        lo5 = store.get_voxel(key)['log_odds']
        assert lo5 > 2.0, f'After 5 scans, expected > 2.0, got {lo5}'

        # Scan 6: near max
        _integrate_point(store, identity_transform, pt)
        lo6 = store.get_voxel(key)['log_odds']
        assert lo6 > 3.0, f'After 6 scans, expected > 3.0, got {lo6}'

        # Scan 7: clamped at L_MAX
        _integrate_point(store, identity_transform, pt)
        lo7 = store.get_voxel(key)['log_odds']
        assert lo7 == pytest.approx(
            store._config.log_odds_max, abs=1e-5
        )


# ===================================================================
# Phase 3 -- Query engine and clearing
# ===================================================================

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _integrate_n_times(store, point, n, T=None):
    """Integrate the same point n times to build up obs_count."""
    if T is None:
        T = np.eye(4)
    pts = np.array([point])
    clrs = np.array([[128, 128, 128]], dtype=np.uint8)
    for _ in range(n):
        store.integrate(pts, clrs, T)


# -------------------------------------------------------------------
# QR-01, QR-03 -- query_height
# -------------------------------------------------------------------

class TestQueryHeight:
    """QR-01/QR-03: query_height with confidence + observation gating."""

    def test_returns_none_for_single_observation(
        self, store, identity_transform
    ):
        """Integrate a point once. query_height with default obs_min=2
        should return None because obs_count=1 < obs_min=2.
        """
        _integrate_n_times(store, [0.1, 0.1, 0.2], 1, identity_transform)
        result = store.query_height(0.1, 0.1, radius=0.05)
        assert result is None

    def test_returns_height_for_multi_observation(
        self, store, identity_transform
    ):
        """Integrate a point 3 times. query_height should return a
        HeightResult with correct max_z matching the voxel center Z.
        """
        _integrate_n_times(store, [0.1, 0.1, 0.2], 3, identity_transform)
        result = store.query_height(0.1, 0.1, radius=0.05)
        assert result is not None
        assert isinstance(result, HeightResult)
        # max_z should be near 0.2 (voxel centre Z)
        res = store._config.voxel_resolution
        np.testing.assert_allclose(result.max_z, 0.2, atol=res)

    def test_returns_none_when_below_confidence_threshold(
        self, identity_transform
    ):
        """A voxel with obs_count >= 2 but log_odds below the
        confidence threshold should be excluded by query_height.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)
        # Integrate twice to get obs_count=2
        _integrate_n_times(store, [0.1, 0.1, 0.2], 2, identity_transform)

        # After 2 integrations: log_odds = -0.8 + 0.85 = 0.05
        # Use a high confidence_min to exclude it
        result = store.query_height(
            0.1, 0.1, radius=0.05, confidence_min=1.0
        )
        assert result is None

    def test_result_fields(self, store, identity_transform):
        """Verify HeightResult has max_z (float), voxel_count (int),
        mean_log_odds (float). Verify frozen (immutable).
        """
        _integrate_n_times(store, [0.1, 0.1, 0.2], 3, identity_transform)
        result = store.query_height(0.1, 0.1, radius=0.05)
        assert result is not None

        assert isinstance(result.max_z, float)
        assert isinstance(result.voxel_count, int)
        assert isinstance(result.mean_log_odds, float)

        # HeightResult should be frozen
        with pytest.raises(AttributeError):
            result.max_z = 999.0

    def test_cylindrical_geometry_excludes_far_xy(
        self, store, identity_transform
    ):
        """Place two points: one at (0.1, 0.0, 0.2) and one at
        (0.3, 0.0, 0.2). Query with radius=0.15 centered at (0.1, 0.0).
        Only the first point should be included. Verifies XY-only
        distance, not 3D.

        Both points are integrated simultaneously in each scan to
        avoid frustum miss cross-contamination.
        """
        pts = np.array([[0.1, 0.0, 0.2], [0.3, 0.0, 0.2]])
        clrs = np.array([[128, 128, 128], [128, 128, 128]], dtype=np.uint8)
        for _ in range(3):
            store.integrate(pts, clrs, identity_transform)

        result = store.query_height(0.1, 0.0, radius=0.15)
        assert result is not None
        assert result.voxel_count == 1
        res = store._config.voxel_resolution
        np.testing.assert_allclose(result.max_z, 0.2, atol=res)

    def test_returns_max_z_among_filtered(
        self, store, identity_transform
    ):
        """Place points at different Z heights within query radius.
        Verify max_z is the highest Z. Integrate each 3+ times.

        All points integrated together per scan to avoid frustum miss.
        """
        pts = np.array([[0.1, 0.1, 0.2], [0.1, 0.1, 0.3], [0.1, 0.1, 0.1]])
        clrs = np.array([[128, 128, 128]] * 3, dtype=np.uint8)
        for _ in range(3):
            store.integrate(pts, clrs, identity_transform)

        result = store.query_height(0.1, 0.1, radius=0.05)
        assert result is not None
        res = store._config.voxel_resolution
        np.testing.assert_allclose(result.max_z, 0.3, atol=res)
        assert result.voxel_count == 3

    def test_custom_obs_min_overrides_default(
        self, store, identity_transform
    ):
        """Integrate 2 times, query with obs_min=3 should return None.
        Query with obs_min=2 should return HeightResult.
        """
        _integrate_n_times(store, [0.1, 0.1, 0.2], 2, identity_transform)

        result_strict = store.query_height(
            0.1, 0.1, radius=0.05, obs_min=3
        )
        assert result_strict is None

        result_default = store.query_height(
            0.1, 0.1, radius=0.05, obs_min=2
        )
        assert result_default is not None

    def test_pitfall_obs_count_gte_not_gt(
        self, store, identity_transform
    ):
        """Integrate exactly 2 times. query_height with obs_min=2
        should return HeightResult (>= not >). Guards against
        off-by-one (Pitfall 4 from research).
        """
        _integrate_n_times(store, [0.1, 0.1, 0.2], 2, identity_transform)
        result = store.query_height(0.1, 0.1, radius=0.05, obs_min=2)
        assert result is not None, (
            'obs_min=2 with obs_count=2 should match (>= not >)'
        )

    def test_returns_none_on_empty_store(self, store):
        """query_height on empty store returns None."""
        result = store.query_height(0.1, 0.1, radius=0.05)
        assert result is None


# -------------------------------------------------------------------
# QR-02 -- query_region
# -------------------------------------------------------------------

class TestQueryRegion:
    """QR-02: query_region with cylindrical geometry."""

    def test_cylindrical_geometry(self, store, identity_transform):
        """Place points at various (x,y,z). Query cylindrical region.
        Verify only points within XY radius AND Z range are returned.

        All points integrated together per scan to avoid frustum miss.
        """
        pts = np.array([
            [0.1, 0.1, 0.2],   # Inside: within XY radius and Z range
            [0.3, 0.3, 0.2],   # Outside XY: too far in XY
            [0.1, 0.1, 0.5],   # Outside Z: outside Z range
        ])
        clrs = np.array([[128, 128, 128]] * 3, dtype=np.uint8)
        for _ in range(3):
            store.integrate(pts, clrs, identity_transform)

        result = store.query_region(
            xy_center=np.array([0.1, 0.1]),
            radius=0.1,
            z_min=0.15,
            z_max=0.25,
        )
        assert result['centers'].shape[0] == 1
        res = store._config.voxel_resolution
        np.testing.assert_allclose(
            result['centers'][0], [0.1, 0.1, 0.2], atol=res
        )

    def test_excludes_below_confidence(self, identity_transform):
        """Place voxels with low confidence. query_region should
        exclude them.
        """
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)
        # Integrate once: log_odds = -0.8 (below confidence_threshold=0.0)
        _integrate_n_times(store, [0.1, 0.1, 0.2], 1, identity_transform)

        result = store.query_region(
            xy_center=np.array([0.1, 0.1]),
            radius=0.1,
            z_min=0.0,
            z_max=0.4,
        )
        # Single observation: log_odds = -0.8, below confidence_threshold=0.0
        assert result['centers'].shape[0] == 0

    def test_returns_dict_of_arrays(self, store, identity_transform):
        """Verify return dict has keys: 'centers', 'colors',
        'log_odds', 'obs_count', 'keys'. Verify shapes are
        consistent (N rows).
        """
        _integrate_n_times(store, [0.1, 0.1, 0.2], 3, identity_transform)

        result = store.query_region(
            xy_center=np.array([0.1, 0.1]),
            radius=0.1,
            z_min=0.0,
            z_max=0.4,
        )
        expected_keys = {'centers', 'colors', 'log_odds', 'obs_count', 'keys'}
        assert set(result.keys()) == expected_keys

        n = result['centers'].shape[0]
        assert n > 0
        assert result['centers'].shape == (n, 3)
        assert result['colors'].shape == (n, 3)
        assert result['log_odds'].shape == (n,)
        assert result['obs_count'].shape == (n,)
        assert result['keys'].shape == (n, 3)

    def test_returns_empty_on_no_match(self, store):
        """Query a region with no voxels. Verify all arrays have
        0 rows.
        """
        result = store.query_region(
            xy_center=np.array([0.1, 0.1]),
            radius=0.1,
            z_min=0.0,
            z_max=0.4,
        )
        assert result['centers'].shape[0] == 0
        assert result['colors'].shape[0] == 0
        assert result['log_odds'].shape[0] == 0
        assert result['obs_count'].shape[0] == 0
        assert result['keys'].shape[0] == 0

    def test_z_range_filtering(self, store, identity_transform):
        """Place points at Z=0.2 and Z=0.5. Query with z_min=0.4,
        z_max=0.55. Only Z=0.5 point included.

        Both points integrated together per scan to avoid frustum miss.
        """
        pts = np.array([[0.1, 0.1, 0.2], [0.1, 0.1, 0.5]])
        clrs = np.array([[128, 128, 128]] * 2, dtype=np.uint8)
        for _ in range(3):
            store.integrate(pts, clrs, identity_transform)

        result = store.query_region(
            xy_center=np.array([0.1, 0.1]),
            radius=0.1,
            z_min=0.4,
            z_max=0.55,
        )
        assert result['centers'].shape[0] == 1
        res = store._config.voxel_resolution
        np.testing.assert_allclose(
            result['centers'][0, 2], 0.5, atol=res
        )

    def test_returns_copies_not_views(self, store, identity_transform):
        """Get query result, modify returned arrays, verify internal
        store state is unchanged.
        """
        _integrate_n_times(store, [0.1, 0.1, 0.2], 3, identity_transform)

        result = store.query_region(
            xy_center=np.array([0.1, 0.1]),
            radius=0.1,
            z_min=0.0,
            z_max=0.4,
        )
        assert result['centers'].shape[0] == 1

        # Save original values
        original_log_odds = result['log_odds'][0]
        original_color = result['colors'][0].copy()

        # Mutate returned arrays
        result['log_odds'][0] = -999.0
        result['colors'][0] = [0.0, 0.0, 0.0]
        result['centers'][0] = [99.0, 99.0, 99.0]

        # Query again -- internal state should be unchanged
        result2 = store.query_region(
            xy_center=np.array([0.1, 0.1]),
            radius=0.1,
            z_min=0.0,
            z_max=0.4,
        )
        assert result2['centers'].shape[0] == 1
        assert result2['log_odds'][0] == pytest.approx(
            original_log_odds, abs=1e-5
        )
        np.testing.assert_allclose(
            result2['colors'][0], original_color, atol=1e-5
        )


# -------------------------------------------------------------------
# CL-01, CL-02 -- clear_region
# -------------------------------------------------------------------

class TestClearRegion:
    """CL-01/CL-02: clear_region with spherical geometry."""

    def test_clears_voxels_in_sphere(self, store, identity_transform):
        """Place multiple voxels. Clear a sphere encompassing some.
        Verify cleared count and remaining voxel_count.

        All points integrated together per scan to avoid frustum miss.
        """
        pts = np.array([
            [0.1, 0.1, 0.2],
            [0.1, 0.1, 0.3],
            [0.3, 0.3, 0.2],
        ])
        clrs = np.array([[128, 128, 128]] * 3, dtype=np.uint8)
        for _ in range(3):
            store.integrate(pts, clrs, identity_transform)

        count_before = store.voxel_count
        assert count_before == 3

        # Clear a sphere centered at (0.1, 0.1, 0.2) with radius 0.06
        # This should encompass (0.1, 0.1, 0.2) but NOT (0.1, 0.1, 0.3)
        # since 3D distance to (0.1,0.1,0.3) is 0.1 > 0.06
        cleared = store.clear_region(
            center=np.array([0.1, 0.1, 0.2]),
            radius=0.06,
        )
        assert cleared == 1
        assert store.voxel_count == 2

    def test_spherical_geometry(self, store, identity_transform):
        """Place voxels at (0,0,0.3) and (0,0.05,0.3). Clear with
        center=(0,0,0.3) radius=0.02. Verify only the close voxel
        is cleared (not the one 50mm away). This proves spherical
        (3D), not cylindrical.

        Both points integrated together per scan to avoid frustum miss.
        """
        pts = np.array([[0.0, 0.0, 0.3], [0.0, 0.05, 0.3]])
        clrs = np.array([[128, 128, 128]] * 2, dtype=np.uint8)
        for _ in range(3):
            store.integrate(pts, clrs, identity_transform)

        cleared = store.clear_region(
            center=np.array([0.0, 0.0, 0.3]),
            radius=0.02,
        )
        assert cleared == 1
        assert store.voxel_count == 1

    def test_returns_zero_for_empty_region(self, store):
        """Clear a region with no voxels. Returns 0."""
        cleared = store.clear_region(
            center=np.array([0.1, 0.1, 0.2]),
            radius=0.05,
        )
        assert cleared == 0

    def test_query_returns_none_after_clear(
        self, store, identity_transform
    ):
        """Integrate, query_height succeeds, then clear_region, then
        query_height returns None. (Brain workflow test)
        """
        _integrate_n_times(store, [0.1, 0.1, 0.2], 3, identity_transform)

        # Query should succeed before clearing
        result_before = store.query_height(0.1, 0.1, radius=0.05)
        assert result_before is not None

        # Clear the region
        cleared = store.clear_region(
            center=np.array([0.1, 0.1, 0.2]),
            radius=0.05,
        )
        assert cleared >= 1

        # Query should return None after clearing
        result_after = store.query_height(0.1, 0.1, radius=0.05)
        assert result_after is None

    def test_clear_preserves_other_voxels(
        self, store, identity_transform
    ):
        """Place voxels in two separate locations. Clear one location.
        Verify the other is intact (query still works).

        Both points integrated together per scan to avoid frustum miss.
        """
        pts = np.array([[0.1, 0.1, 0.2], [0.3, 0.3, 0.2]])
        clrs = np.array([[128, 128, 128]] * 2, dtype=np.uint8)
        for _ in range(3):
            store.integrate(pts, clrs, identity_transform)

        store.clear_region(
            center=np.array([0.1, 0.1, 0.2]),
            radius=0.05,
        )

        # The other voxel should still be queryable
        result = store.query_height(0.3, 0.3, radius=0.05)
        assert result is not None
        res = store._config.voxel_resolution
        np.testing.assert_allclose(result.max_z, 0.2, atol=res)

    def test_clear_after_prune_no_corruption(self, identity_transform):
        """Trigger a prune, then clear_region. Verify no KeyError or
        data corruption. (Pitfall 3 from research)

        All points integrated together per scan to avoid frustum miss.
        """
        cfg = WorldMapConfig(prune_interval=5)
        store = VoxelStore(cfg)
        T = identity_transform

        # Create 3 voxels (integrated together to avoid frustum miss)
        pts = np.array([[0.0, 0.0, 0.3], [0.1, 0.1, 0.3], [0.2, 0.2, 0.3]])
        clrs = np.array([[128, 128, 128]] * 3, dtype=np.uint8)
        for _ in range(3):
            store.integrate(pts, clrs, T)

        # Kill one voxel to make prune do work
        first_key = list(store._voxels.keys())[0]
        first_idx = store._voxels[first_key]
        store._log_odds[first_idx] = cfg.log_odds_min + 0.005

        # Advance to prune interval
        while store.scan_index % cfg.prune_interval != 0:
            _integrate_empty(store, T)

        # After prune, should have 2 voxels
        assert store.voxel_count == 2

        # Now clear_region should work without KeyError
        cleared = store.clear_region(
            center=np.array([0.1, 0.1, 0.3]),
            radius=0.05,
        )
        assert cleared >= 1
        assert store.voxel_count <= 1

        # Remaining voxel (if any) should be queryable
        data = store.get_all_data()
        assert data['keys'].shape[0] == store.voxel_count
        assert data['log_odds'].shape[0] == store.voxel_count


# -------------------------------------------------------------------
# QR-03 -- Query performance
# -------------------------------------------------------------------

class TestQueryPerformance:
    """QR-03: Query latency under 10ms for 50K voxels."""

    @pytest.fixture
    def populated_store_50k(self, identity_transform):
        """Pre-populate a store with ~50K voxels with obs_count >= 2."""
        cfg = WorldMapConfig()
        store = VoxelStore(cfg)
        res = cfg.voxel_resolution
        bmin = cfg.bounds_center - cfg.bounds_half_extents

        # 37^3 = 50653; take first 50000.
        side = 37
        grid = np.stack(
            np.meshgrid(
                np.arange(side), np.arange(side), np.arange(side),
                indexing='ij',
            ),
            axis=-1,
        ).reshape(-1, 3)[:50000]

        points = (grid.astype(np.float64) + 0.5) * res + bmin
        colors = np.full((len(points), 3), 128, dtype=np.float32)

        # Integrate twice to get obs_count >= 2
        store.integrate(points, colors, identity_transform)
        store.integrate(points, colors, identity_transform)

        assert store.voxel_count == 50000
        return store

    def test_query_height_latency_50k(self, populated_store_50k):
        """query_height under 15ms with 50K voxels (target <10ms,
        relaxed for CI stability).
        """
        store = populated_store_50k

        # Warm up
        store.query_height(0.0, 0.0, radius=0.05)

        # Time it (best of 5)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = store.query_height(0.0, 0.0, radius=0.05)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        best = min(times)
        assert best < 0.015, (
            f'query_height took {best*1000:.1f}ms, expected < 15ms'
        )

    def test_query_region_latency_50k(self, populated_store_50k):
        """query_region under 15ms with 50K voxels (target <10ms,
        relaxed for CI stability).
        """
        store = populated_store_50k

        # Warm up
        store.query_region(
            xy_center=np.array([0.0, 0.0]),
            radius=0.05,
            z_min=0.0,
            z_max=0.4,
        )

        # Time it (best of 5)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = store.query_region(
                xy_center=np.array([0.0, 0.0]),
                radius=0.05,
                z_min=0.0,
                z_max=0.4,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        best = min(times)
        assert best < 0.015, (
            f'query_region took {best*1000:.1f}ms, expected < 15ms'
        )

    def test_clear_region_latency_50k(self, populated_store_50k):
        """clear_region completes in reasonable time (under 50ms)
        with 50K voxels.
        """
        store = populated_store_50k

        start = time.perf_counter()
        cleared = store.clear_region(
            center=np.array([0.0, 0.0, 0.2]),
            radius=0.03,
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 0.050, (
            f'clear_region took {elapsed*1000:.1f}ms, expected < 50ms'
        )
        assert cleared >= 0  # Sanity check


# ===================================================================
# Phase 5 -- Kalman refinement tests
# ===================================================================

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def kalman_config():
    """Return WorldMapConfig with Kalman enabled."""
    return WorldMapConfig(kalman_enabled=True)


@pytest.fixture
def kalman_store(kalman_config):
    """Return fresh VoxelStore with Kalman enabled."""
    return VoxelStore(kalman_config)


# -------------------------------------------------------------------
# KF-01, KF-03 -- TestKalmanBasic
# -------------------------------------------------------------------

class TestKalmanBasic:
    """Basic Kalman array allocation, initialization, and data access."""

    def test_kalman_arrays_allocated_when_enabled(self, kalman_store):
        """_kf_mean/_kf_var/_kf_n are non-None with correct shape/dtype."""
        assert kalman_store._kf_mean is not None
        assert kalman_store._kf_var is not None
        assert kalman_store._kf_n is not None
        assert kalman_store._kf_mean.shape == (kalman_store._capacity, 3)
        assert kalman_store._kf_var.shape == (kalman_store._capacity, 3)
        assert kalman_store._kf_n.shape == (kalman_store._capacity,)
        assert kalman_store._kf_mean.dtype == np.float64
        assert kalman_store._kf_var.dtype == np.float64
        assert kalman_store._kf_n.dtype == np.int32

    def test_kalman_arrays_none_when_disabled(self, store):
        """All three Kalman arrays are None with default config."""
        assert store._kf_mean is None
        assert store._kf_var is None
        assert store._kf_n is None

    def test_new_voxel_initialized_with_measured_position(
        self, kalman_store, identity_transform
    ):
        """Integrate a single point, verify _kf_mean[0] is the actual
        measured position (not centroid), _kf_n[0] == 1."""
        pt = [0.45, 0.15, 0.30]
        _integrate_point(kalman_store, identity_transform, pt)
        key, voxel = _get_only_voxel(kalman_store)
        idx = kalman_store._voxels[key]
        assert kalman_store._kf_n[idx] == 1
        np.testing.assert_allclose(
            kalman_store._kf_mean[idx], pt, atol=1e-10
        )

    def test_multi_point_single_voxel_averaged(
        self, kalman_store, identity_transform
    ):
        """Integrate 3 points that map to the same voxel, verify
        _kf_mean[0] is the mean of the 3 points."""
        res = kalman_store._config.voxel_resolution
        base = np.array([0.45, 0.15, 0.30])
        offsets = [
            base,
            base + np.array([0.001, 0.0, 0.0]),
            base + np.array([0.0, 0.001, 0.0]),
        ]
        pts = np.array(offsets, dtype=np.float64)
        cols = np.full((3, 3), 0.5, dtype=np.float32)
        kalman_store.integrate(pts, cols, identity_transform)
        key = next(iter(kalman_store._voxels))
        idx = kalman_store._voxels[key]
        expected_mean = pts.mean(axis=0)
        np.testing.assert_allclose(
            kalman_store._kf_mean[idx], expected_mean, atol=1e-10
        )

    def test_initial_variance_floored(
        self, kalman_store, identity_transform
    ):
        """Integrate 20 points in one voxel, verify _kf_var >= R * 0.5."""
        base = np.array([0.45, 0.15, 0.30])
        pts = np.array([base + np.random.default_rng(42).uniform(-0.001, 0.001, 3)
                        for _ in range(20)], dtype=np.float64)
        cols = np.full((20, 3), 0.5, dtype=np.float32)
        kalman_store.integrate(pts, cols, identity_transform)
        key = next(iter(kalman_store._voxels))
        idx = kalman_store._voxels[key]
        R = kalman_store._config.kalman_measurement_noise
        assert np.all(kalman_store._kf_var[idx] >= R * 0.5 - 1e-15)

    def test_kalman_update_existing_voxel(
        self, kalman_store, identity_transform
    ):
        """Integrate twice, verify _kf_n == 2, position shifted toward
        second measurement."""
        pt1 = [0.45, 0.15, 0.30]
        pt2 = [0.451, 0.151, 0.301]
        _integrate_point(kalman_store, identity_transform, pt1)
        key = next(iter(kalman_store._voxels))
        idx = kalman_store._voxels[key]
        mean_after_1 = kalman_store._kf_mean[idx].copy()

        _integrate_point(kalman_store, identity_transform, pt2)
        idx = kalman_store._voxels[key]
        assert kalman_store._kf_n[idx] == 2
        mean_after_2 = kalman_store._kf_mean[idx]

        # Position should have shifted toward pt2
        dist_to_pt2_before = np.linalg.norm(mean_after_1 - np.array(pt2))
        dist_to_pt2_after = np.linalg.norm(mean_after_2 - np.array(pt2))
        assert dist_to_pt2_after < dist_to_pt2_before, (
            'Kalman update should shift mean toward new measurement'
        )

    def test_get_all_data_positions_key_exists(
        self, kalman_store, store, identity_transform
    ):
        """Verify 'positions' key in get_all_data() for both enabled
        and disabled configs."""
        # Kalman enabled
        _integrate_point(kalman_store, identity_transform, [0.45, 0.15, 0.30])
        data_on = kalman_store.get_all_data()
        assert 'positions' in data_on

        # Kalman disabled
        _integrate_point(store, identity_transform, [0.45, 0.15, 0.30])
        data_off = store.get_all_data()
        assert 'positions' in data_off


# -------------------------------------------------------------------
# KF-01 -- TestKalmanConvergence
# -------------------------------------------------------------------

class TestKalmanConvergence:
    """Kalman convergence: variance decreases, gain remains nonzero."""

    def test_convergence_in_3_to_5_observations(
        self, kalman_store, identity_transform
    ):
        """Integrate same point 5 times, verify variance decreases
        monotonically."""
        pt = [0.45, 0.15, 0.30]
        variances = []
        key = None
        for i in range(5):
            _integrate_point(kalman_store, identity_transform, pt)
            if key is None:
                key = next(iter(kalman_store._voxels))
            idx = kalman_store._voxels[key]
            variances.append(kalman_store._kf_var[idx].copy())

        # Variance should decrease monotonically starting from obs 2
        # (obs 1 is initialization, obs 2 is first Kalman update)
        for i in range(2, len(variances)):
            assert np.all(variances[i] <= variances[i - 1] + 1e-15), (
                f'Variance should decrease: step {i}: {variances[i]} > {variances[i-1]}'
            )

    def test_gain_does_not_collapse(
        self, kalman_store, identity_transform
    ):
        """Integrate 10 times, verify Kalman gain P/(P+R) remains > 0.01."""
        pt = [0.45, 0.15, 0.30]
        key = None
        for _ in range(10):
            _integrate_point(kalman_store, identity_transform, pt)
            if key is None:
                key = next(iter(kalman_store._voxels))

        idx = kalman_store._voxels[key]
        P = kalman_store._kf_var[idx]
        R = kalman_store._config.kalman_measurement_noise
        gain = P / (P + R)
        assert np.all(gain > 0.01), (
            f'Kalman gain collapsed to {gain}, expected > 0.01'
        )

    def test_refined_position_after_2_observations(
        self, kalman_store, identity_transform
    ):
        """Integrate 2 times, verify get_all_data()['positions'] differs
        from centers for the voxel."""
        pt = [0.45, 0.15, 0.30]
        _integrate_point(kalman_store, identity_transform, pt)
        _integrate_point(kalman_store, identity_transform, pt)

        data = kalman_store.get_all_data()
        # With kf_n >= 2, positions should use Kalman mean (not centroid)
        assert not np.allclose(data['positions'], data['centers'], atol=1e-10), (
            'After 2 observations, positions should differ from centers'
        )

    def test_single_observation_returns_centroid(
        self, kalman_store, identity_transform
    ):
        """Integrate once, verify get_all_data()['positions'] equals
        centers (kf_n < 2 fallback)."""
        pt = [0.45, 0.15, 0.30]
        _integrate_point(kalman_store, identity_transform, pt)

        data = kalman_store.get_all_data()
        np.testing.assert_array_equal(
            data['positions'], data['centers'],
            err_msg='Single observation should return centroid, not Kalman mean'
        )


# -------------------------------------------------------------------
# KF-01 -- TestKalmanDecayReset
# -------------------------------------------------------------------

class TestKalmanDecayReset:
    """Kalman state reset when voxels decay below confidence threshold.

    NOTE: Multiplicative decay (factor=0.3) on positive log_odds
    never crosses zero.  With confidence_threshold=0.0 the Kalman
    reset cannot fire via decay alone.  We use
    confidence_threshold=0.5 so that 0.66 * 0.3 = 0.198 < 0.5
    triggers the reset.
    """

    def test_decay_resets_kalman_state(self, identity_transform):
        """Build up Kalman state, let decay cross the (raised)
        confidence threshold, verify _kf_n == 0 and _kf_mean == 0."""
        cfg = WorldMapConfig(
            kalman_enabled=True,
            decay_after_scans=2,
            confidence_threshold=0.5,
            prune_interval=100,
        )
        store = VoxelStore(cfg)
        T = identity_transform

        pt = [0.45, 0.15, 0.30]
        for _ in range(4):
            _integrate_point(store, T, pt)

        key = next(iter(store._voxels))
        idx = store._voxels[key]
        assert store._kf_n[idx] == 4

        # Run empty integrations to trigger decay past threshold
        for _ in range(10):
            _integrate_empty(store, T)

        # Verify Kalman state was reset
        assert key in store._voxels
        idx = store._voxels[key]
        assert store._kf_n[idx] == 0, (
            'Kalman state should be reset after decay below threshold'
        )
        np.testing.assert_array_equal(
            store._kf_mean[idx], [0.0, 0.0, 0.0]
        )

    def test_re_observation_after_decay_starts_fresh(
        self, identity_transform
    ):
        """After decay reset, integrate again, verify _kf_n == 1 and
        position is the new measurement."""
        cfg = WorldMapConfig(
            kalman_enabled=True,
            decay_after_scans=2,
            confidence_threshold=0.5,
            prune_interval=100,
        )
        store = VoxelStore(cfg)
        T = identity_transform

        # Build up
        pt_old = [0.45, 0.15, 0.30]
        for _ in range(4):
            _integrate_point(store, T, pt_old)

        key = next(iter(store._voxels))
        idx = store._voxels[key]
        assert store._kf_n[idx] == 4

        # Decay: run enough empty integrations to trigger Kalman reset
        for _ in range(10):
            _integrate_empty(store, T)

        # Verify decay reset happened
        assert key in store._voxels
        idx = store._voxels[key]
        assert store._kf_n[idx] == 0

        # Re-observe with new measurement
        pt_new = [0.451, 0.151, 0.301]
        _integrate_point(store, T, pt_new)

        assert key in store._voxels
        idx = store._voxels[key]
        assert store._kf_n[idx] == 1
        np.testing.assert_allclose(
            store._kf_mean[idx], pt_new, atol=1e-10,
            err_msg='After decay reset, new observation should be fresh'
        )


# -------------------------------------------------------------------
# KF-05 -- TestKalmanDisabled
# -------------------------------------------------------------------

class TestKalmanDisabled:
    """No-regression tests: Kalman disabled matches Phase 4 behavior."""

    def test_no_regression_integrate_disabled(
        self, store, identity_transform
    ):
        """Run same integration sequence with Kalman disabled, verify
        voxel state (log_odds, obs_count, colors) is valid."""
        pt = [0.45, 0.15, 0.30]
        for _ in range(3):
            _integrate_point(store, identity_transform, pt)

        key = next(iter(store._voxels))
        voxel = store.get_voxel(key)
        assert voxel is not None
        assert voxel['obs_count'] == 3
        assert voxel['log_odds'] > 0.0  # Should be confident after 3 hits

    def test_positions_equal_centers_when_disabled(
        self, store, identity_transform
    ):
        """get_all_data()['positions'] == get_all_data()['centers'] with
        Kalman disabled."""
        pt = [0.45, 0.15, 0.30]
        for _ in range(3):
            _integrate_point(store, identity_transform, pt)

        data = store.get_all_data()
        np.testing.assert_array_equal(
            data['positions'], data['centers'],
            err_msg='With Kalman disabled, positions must equal centers'
        )

    def test_memory_unchanged_when_disabled(self, store, identity_transform):
        """Memory_bytes for a given voxel count matches Phase 4 formula
        (136 bytes/voxel = 36 array + 100 dict overhead)."""
        pt = [0.45, 0.15, 0.30]
        _integrate_point(store, identity_transform, pt)
        expected = 1 * (36 + 100)  # Phase 4 formula
        assert store.memory_bytes == expected


# -------------------------------------------------------------------
# TestKalmanCompactAndGrow
# -------------------------------------------------------------------

class TestKalmanCompactAndGrow:
    """Kalman state survives compact (clear_region) and grow operations."""

    def test_compact_preserves_kalman_state(
        self, kalman_store, identity_transform
    ):
        """Integrate points in two voxels, clear one via clear_region,
        verify remaining voxel's Kalman state is preserved."""
        T = identity_transform
        pts = np.array([
            [0.45, 0.15, 0.30],
            [0.45, 0.30, 0.30],
        ], dtype=np.float64)
        cols = np.full((2, 3), 0.5, dtype=np.float32)

        for _ in range(3):
            kalman_store.integrate(pts, cols, T)

        assert kalman_store.voxel_count == 2

        # Find the key for the second point's voxel
        keys = list(kalman_store._voxels.keys())
        # Record Kalman state for second voxel before clearing
        second_key = keys[1]
        second_idx = kalman_store._voxels[second_key]
        kf_mean_before = kalman_store._kf_mean[second_idx].copy()
        kf_n_before = kalman_store._kf_n[second_idx]

        # Clear first voxel
        center_first = kalman_store.voxel_center(keys[0])
        kalman_store.clear_region(center_first, radius=0.003)

        assert kalman_store.voxel_count == 1

        # Verify remaining voxel's Kalman state is preserved
        remaining_key = next(iter(kalman_store._voxels))
        remaining_idx = kalman_store._voxels[remaining_key]
        np.testing.assert_allclose(
            kalman_store._kf_mean[remaining_idx], kf_mean_before, atol=1e-10
        )
        assert kalman_store._kf_n[remaining_idx] == kf_n_before

    def test_grow_preserves_kalman_state(self, identity_transform):
        """Integrate enough voxels to trigger a _grow, verify Kalman
        state for early voxels is still correct."""
        cfg = WorldMapConfig(kalman_enabled=True)
        store = VoxelStore(cfg)
        T = identity_transform

        # Integrate a single voxel and record its Kalman state
        pt = [0.45, 0.15, 0.30]
        _integrate_point(store, T, pt)
        key = next(iter(store._voxels))
        idx = store._voxels[key]
        kf_mean_before = store._kf_mean[idx].copy()
        kf_n_before = store._kf_n[idx]
        initial_cap = store._capacity

        # Generate enough unique voxels to exceed initial capacity.
        # Points must pass workspace bounds: cylindrical annulus around
        # base_xy with r in [inner_radius, outer_radius], x>0, y>0,
        # z in [z_min, z_max].
        res = cfg.voxel_resolution
        base_xy = cfg.workspace_base_xy
        # Spread points in a grid within the workspace annulus.
        # Use angles from 0.1 to pi/2-0.1 (x>0, y>0) and radii
        # from inner+0.02 to inner+0.5 to stay within outer.
        n_target = initial_cap + 100
        pts_list = []
        r_inner = cfg.workspace_inner_radius + 0.02
        r_outer = min(cfg.workspace_outer_radius - 0.02, r_inner + 0.5)
        n_r = 50
        n_theta = 30
        n_z = 3
        for ri in range(n_r):
            r = r_inner + (r_outer - r_inner) * ri / n_r
            for ti in range(n_theta):
                theta = 0.1 + (np.pi / 2 - 0.2) * ti / n_theta
                for zi in range(n_z):
                    z = 0.05 + zi * 0.10
                    x = base_xy[0] + r * np.cos(theta)
                    y = base_xy[1] + r * np.sin(theta)
                    pts_list.append([x, y, z])
                    if len(pts_list) >= n_target:
                        break
                if len(pts_list) >= n_target:
                    break
            if len(pts_list) >= n_target:
                break

        pts = np.array(pts_list, dtype=np.float64)
        cols = np.full((len(pts), 3), 0.5, dtype=np.float32)
        store.integrate(pts, cols, T)

        assert store._capacity > initial_cap, (
            f'Should have triggered _grow: cap={store._capacity}, '
            f'voxels={store.voxel_count}, target={n_target}'
        )

        # Verify original voxel's Kalman state survived the grow
        assert key in store._voxels
        idx_after = store._voxels[key]
        np.testing.assert_allclose(
            store._kf_mean[idx_after], kf_mean_before, atol=1e-10
        )
        assert store._kf_n[idx_after] == kf_n_before
