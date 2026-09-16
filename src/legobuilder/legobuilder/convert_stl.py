"""Convert ASCII STL files to binary format for RViz2 compatibility.

RViz2 requires binary STL files for mesh markers. This CLI tool converts
one or more ASCII STL files in-place, backing up the originals with an
``.ascii_bak`` suffix.
"""

import stl
import sys
import os


def main():
    """Convert each STL file given on the command line from ASCII to binary.

    Reads each path from ``sys.argv``, loads the mesh, renames the
    original file with an ``.ascii_bak`` extension, and saves the
    binary version at the original path.
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_stl.py <file1.stl> [file2.stl ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"Converting {path} from ASCII to binary...")
        mesh = stl.mesh.Mesh.from_file(path)
        os.rename(path, path + ".ascii_bak")
        mesh.save(path)
        print(f"  Done: {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
