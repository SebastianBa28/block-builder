import os
import pytest

from legobuilder.brain.verify_assembly_plan import AssemblyPlanVerifier

ASSEMBLIES_DIR = os.path.join(
    os.path.dirname(__file__), '../../src/legobuilder', 'assemblies/tests'
)


def _verify(filename, **kwargs):
    path = os.path.join(ASSEMBLIES_DIR, filename)
    return AssemblyPlanVerifier(path, **kwargs).verify()


# ── Passing tests ──────────────────────────────────────────────


class TestPassingAssemblies:

    def test_single_block(self):
        report = _verify('pass_single_block.json')
        assert report.passed
        assert report.errors == []

    def test_tower(self):
        report = _verify('pass_tower.json')
        assert report.passed
        assert report.errors == []

    def test_l_shape(self):
        report = _verify('pass_l_shape.json')
        assert report.passed
        assert report.errors == []

    def test_cantilever_at_limit(self):
        report = _verify('pass_cantilever_at_limit.json')
        assert report.passed
        assert report.errors == []

    def test_bridge(self):
        report = _verify('pass_bridge.json')
        assert report.passed
        assert report.errors == []
        # Bridge has transient disconnected warnings during build
        assert len(report.warnings) > 0

    def test_staircase(self):
        report = _verify('pass_staircase.json')
        assert report.passed
        assert report.errors == []


# ── Failing tests (errors) ─────────────────────────────────────


class TestFailingAssemblies:

    def test_out_of_bounds(self):
        report = _verify('fail_out_of_bounds.json')
        assert not report.passed
        assert any('out of grid bounds' in e for e in report.errors)

    def test_z_zero(self):
        report = _verify('fail_z_zero.json')
        assert not report.passed
        assert any('z must be >= 1' in e for e in report.errors)

    def test_duplicate_position(self):
        report = _verify('fail_duplicate_position.json')
        assert not report.passed
        assert any('duplicate position' in e for e in report.errors)

    def test_duplicate_id(self):
        report = _verify('fail_duplicate_id.json')
        assert not report.passed
        assert any('duplicate ID' in e for e in report.errors)

    def test_invalid_color(self):
        report = _verify('fail_invalid_color.json')
        assert not report.passed
        assert any("unknown color 'purple'" in e for e in report.errors)

    def test_floating(self):
        report = _verify('fail_floating.json')
        assert not report.passed
        assert any('no adjacent support' in e for e in report.errors)

    def test_build_order(self):
        report = _verify('fail_build_order.json')
        assert not report.passed
        assert any('no adjacent support' in e for e in report.errors)

    def test_cantilever(self):
        report = _verify('fail_cantilever.json')
        assert not report.passed
        assert any('cantilever distance 3 exceeds limit 2' in e for e in report.errors)

    def test_staircase(self):
        report = _verify('fail_staircase.json')
        assert not report.passed
        assert any('cantilever distance 3 exceeds limit 2' in e for e in report.errors)


# ── Disconnected (warning only, still passes) ──────────────────


class TestWarningAssemblies:

    def test_disconnected(self):
        report = _verify('fail_disconnected.json')
        assert report.passed  # warnings don't cause failure
        assert report.errors == []
        assert any('not connected to ground' in w for w in report.warnings)
