"""Launch the USB camera node and camera detector.

This launch file is intended show how the pieces come together.
Please copy the relevant pieces.

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node

from detectors.config import FOCUS, EXPOSURE, AUTO_EXPOSURE, \
    GAIN, WHITE_BALANCE, AUTOWHITEBALANCE, BRIGHTNESS, CONTRAST, SATURATION, SHARPNESS, AUTO_FOCUS

#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure the USB camera node
    node_usbcam = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'usb_cam',
        output     = 'screen',
        parameters = [{'camera_name':  'logitech'},
                      {'video_device': '/dev/video0'},
                      {'pixel_format': 'yuyv2rgb'},
                      {'image_width':  640},
                      {'image_height': 480},
                      {'framerate':    15.0},
                      {'brightness':   BRIGHTNESS},
                      {'constrast':    CONTRAST},
                      {'saturation':   SATURATION},
                      {'sharpness':    SHARPNESS},
                      {'gain':         GAIN},
                      {'autoexposure': AUTO_EXPOSURE},
                      {'exposure':     EXPOSURE},
                      {'auto_white_balance': AUTOWHITEBALANCE},
                      {'white_balance':      WHITE_BALANCE},
                      {'autofocus':    AUTO_FOCUS},
                      {'focus':        FOCUS},
                      ])

    # Configure the camera detector node
    node_camera_detector = Node(
        name       = 'detectors', 
        package    = 'detectors',
        executable = 'detector',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])

    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the nodes.
        node_usbcam,
        node_camera_detector,
    ])
