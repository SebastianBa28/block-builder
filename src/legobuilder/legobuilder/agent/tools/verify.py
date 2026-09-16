"""Assembly verification tool for the design agent."""

import json
from pathlib import Path

from legobuilder.brain.verify_assembly_plan import AssemblyPlanVerifier

_ASSEMBLIES_DIR = Path(__file__).resolve().parents[3] / "assemblies"
_TMP_DIR = _ASSEMBLIES_DIR / "tmp"


def verify_assembly(assembly_json: str) -> str:
    """Verify a magnetic block assembly JSON for physical buildability.

    Runs 8 checks: grid bounds, duplicate positions, duplicate IDs,
    valid colors, build order support, cantilever limits (max 2),
    ground connectivity, and reachability.

    Args:
        assembly_json: Complete assembly JSON string with metadata and blocks.

    Returns:
        Verification report: PASSED/FAILED with error and warning details.
    """
    try:
        json.loads(assembly_json)
    except json.JSONDecodeError as e:
        return f"FAILED: Invalid JSON — {e}"

    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _TMP_DIR / "agent_verify_tmp.json"

    try:
        tmp_path.write_text(assembly_json)
        verifier = AssemblyPlanVerifier(
            str(tmp_path), max_cantilever=2,
        )
        report = verifier.verify()
        return str(report)
    finally:
        tmp_path.unlink(missing_ok=True)
