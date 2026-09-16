"""Camera device discovery utilities.

Provides helpers for enumerating V4L2 camera devices and categorising
them by type (depth vs regular) for use by the detector node.
"""

import subprocess


def get_camera_devices(v4l2_output=None):
    """Parse v4l2 device list and categorise cameras by type.

    Runs v4l2-ctl --list-devices (or accepts pre-captured output)
    and returns device paths grouped as 'depth' or 'regular'
    based on keywords in the camera name.

    Arguments
    ---------
    v4l2_output : str or None
        Raw stdout from v4l2-ctl --list-devices.  When *None* the
        command is executed as a subprocess.

    Returns
    -------
    dict[str, list[str]]
        Mapping of 'depth' and/or 'regular' to lists of
        /dev/videoN device paths.
    """
    if v4l2_output is None:
        try:
            result = subprocess.run(
                ['v4l2-ctl', '--list-devices'],
                capture_output=True,
                text=True,
                check=True
            )
            v4l2_output = result.stdout
        except FileNotFoundError:
            print("Error: v4l2-ctl command not found. Ensure 'v4l-utils' is installed.")
            return {}
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            return {}

    cameras = {}
    current_key = None

    for line in v4l2_output.splitlines():
        if not line.strip():
            continue

        if not line[0].isspace():
            name_lower = line.lower()
            if "realsense" in name_lower or "depth" in name_lower:
                current_key = "depth"
            else:
                current_key = "regular"
            if current_key not in cameras:
                cameras[current_key] = []
        else:
            if current_key is not None:
                cameras[current_key].append(line.strip())

    return cameras


if __name__ == "__main__":
    mock_data = """Intel(R) RealSense(TM) Depth Ca (usb-0000:00:14.0-3):
        /dev/video2
        /dev/video3
        /dev/video6
        /dev/video7
        /dev/video8
        /dev/video9
        /dev/media1
        /dev/media2

HD Pro Webcam C920 (usb-0000:00:14.0-4):
        /dev/video0
        /dev/video1
        /dev/media0"""

    device_dict = get_camera_devices(mock_data)

    import json
    print(json.dumps(device_dict, indent=4))
