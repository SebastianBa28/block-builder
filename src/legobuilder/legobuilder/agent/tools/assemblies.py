"""Assembly file management tools for the design agent."""

import json
from pathlib import Path

from legobuilder.brain.verify_assembly_plan import AssemblyPlanVerifier

_ASSEMBLIES_DIR = Path(__file__).resolve().parents[3] / "assemblies"
_TMP_DIR = _ASSEMBLIES_DIR / "tmp"


def list_existing_assemblies() -> str:
    """List all saved assembly JSON files with their metadata.

    Scans the assemblies directory (excluding old/, tests/, tmp/ subdirs)
    and returns a summary of each assembly.

    Returns:
        Formatted list of assemblies with title, description, block count,
        and colors used.
    """
    entries = []
    for path in sorted(_ASSEMBLIES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        meta = data.get("metadata", {})
        blocks = data.get("blocks", [])
        colors = sorted({b.get("color", "?") for b in blocks})

        entries.append(
            f"- {path.name}\n"
            f"    Title: {meta.get('title', '(none)')}\n"
            f"    Description: {meta.get('description', '(none)')}\n"
            f"    Blocks: {len(blocks)}\n"
            f"    Colors: {', '.join(colors)}"
        )

    if not entries:
        return "No assembly files found."
    return '\n'.join(entries)


def load_assembly(filename: str) -> str:
    """Load an existing assembly JSON file by filename.

    Args:
        filename: Name of the JSON file (e.g., 'flower.json').

    Returns:
        The full JSON content of the assembly file, or an error message.
    """
    path = (_ASSEMBLIES_DIR / filename).resolve()
    if not path.is_relative_to(_ASSEMBLIES_DIR.resolve()):
        return "Error: invalid filename (path traversal rejected)."
    if not path.exists():
        return f"Error: '{filename}' not found in assemblies directory."
    return path.read_text()


def save_assembly(filename: str, assembly_json: str) -> str:
    """Save an assembly JSON to disk after verifying it passes all checks.

    The assembly must pass verification before it will be saved.

    Args:
        filename: Name for the JSON file (e.g., 'tower.json').
        assembly_json: Complete assembly JSON string.

    Returns:
        Success message with file path, or verification failure details.
    """
    if not filename.endswith(".json"):
        filename += ".json"

    path = (_ASSEMBLIES_DIR / filename).resolve()
    if not path.is_relative_to(_ASSEMBLIES_DIR.resolve()):
        return "Error: invalid filename (path traversal rejected)."

    # Validate JSON
    try:
        data = json.loads(assembly_json)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON — {e}"

    # Verify before saving
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _TMP_DIR / "agent_save_tmp.json"
    try:
        tmp_path.write_text(assembly_json)
        verifier = AssemblyPlanVerifier(
            str(tmp_path), max_cantilever=2,
        )
        report = verifier.verify()
    finally:
        tmp_path.unlink(missing_ok=True)

    if not report.passed:
        return (
            f"Cannot save — verification failed:\n{report}\n\n"
            f"Fix the errors and try again."
        )

    # Pretty-print and save
    path.write_text(json.dumps(data, indent=2) + '\n')
    return f"Saved to {path}"
