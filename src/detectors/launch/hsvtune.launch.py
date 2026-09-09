"""Launch the USB camera node and HSV tuning utility.

This launch file is intended show how the pieces come together.
Please copy the relevant pieces.

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown, IncludeLaunchDescription
from launch_ros.actions                import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

from detectors.config import FOCUS, EXPOSURE, AUTO_EXPOSURE, \
    GAIN, WHITE_BALANCE, AUTOWHITEBALANCE, BRIGHTNESS, CONTRAST, SATURATION, SHARPNESS, AUTO_FOCUS

CAMERA_TYPE = 'depth'  # depth, regular

#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure the USB camera node
    node_cam_reg = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'usb_cam',
        output     = 'screen',
        parameters = [{'camera_name':  'logitech'},
                      {'video_device': '/dev/video6'},
                      {'pixel_format': 'yuyv2rgb'},
                      {'image_width':  1920},
                      {'image_height': 1080},
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

    # Depth
    rsfile = os.path.join(pkgdir('realsense2_camera'), 'launch/rs_launch.py')
    rsargs = {'camera_name':             'camera',  # camera unique name
              'depth_module.profile':    '0,0,30',   # depth W, H, FPS
              'rgb_camera.profile':      '0,0,30',   # color W, H, FPS
              'enable_color':            'true',    # enable color stream
              'enable_infra1':           'false',   # enable infra1 stream
              'enable_infra2':           'false',   # enable infra2 stream
              'enable_depth':            'true',    # enable depth stream
              'align_depth.enable':      'false',   # enabled aligned depth
              'pointcloud.enable':       'false',   # Turn on point cloud
              'allow_no_texture_points': 'true'}    # All points without texture
    node_cam_depth = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rsfile),
        launch_arguments=rsargs.items())

    # Select the camera node and remappings based on the CAMERA_TYPE
    node_cam = None
    remappings = None
    if CAMERA_TYPE == 'depth':
        node_cam = node_cam_depth
        remappings = [('/image_raw', '/camera/camera/color/image_raw')]
    elif CAMERA_TYPE == 'regular':
        node_cam = node_cam_reg
        remappings = [('/image_raw', '/usb_cam/image_raw')]
    else:
        raise ValueError(f"Invalid CAMERA_TYPE: {CAMERA_TYPE}.")

    # Configure the HSV tuning utility node
    node_hsvtune = Node(
        name       = 'hsvtuner', 
        package    = 'detectors',
        executable = 'hsvtune',
        output     = 'screen',
        # remappings = [('/image_raw', '/usb_cam/image_raw')])
        remappings = remappings)

    # Configure the camera settings
    node_cameracontrol = Node(
        name       = 'cameracontrol', 
        package    = 'detectors',
        executable = 'cameracontrol',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the nodes.
        node_cam,
        node_hsvtune,
        # node_cameracontrol,
    ])
