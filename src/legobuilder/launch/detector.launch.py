"""Launch the USB camera node and camera detector.

This launch file is intended show how the pieces come together.
Please copy the relevant pieces.

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir


from datetime import datetime

from launch                            import LaunchDescription
from launch.actions                    import Shutdown, IncludeLaunchDescription, ExecuteProcess
from launch_ros.actions                import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

from legobuilder.config import EE_DEPTH_SCAN, RECORD_BAG, VISUAL_GRIP_CHECK, TEST_GRIPPER
from legobuilder.vision.utils import get_camera_devices

FOCUS = 0
AUTO_FOCUS = False
EXPOSURE = 90
AUTO_EXPOSURE = False
GAIN = 81
WHITE_BALANCE = 3488
AUTOWHITEBALANCE = False # 0=OFF, 1=ON
BRIGHTNESS = 28
CONTRAST = 182
SATURATION = 255
SHARPNESS = 255

#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS
    
    rvizcfg = os.path.join(pkgdir('legobuilder'), 'rviz/urdf_image_pointcloud.rviz')

    device_dict = get_camera_devices()
    regular_device = device_dict['regular'][0]
    depth_device = device_dict['depth'][0] if (EE_DEPTH_SCAN or VISUAL_GRIP_CHECK or TEST_GRIPPER) else None
    
    print(f"Regular camera device: {regular_device}")
    print(f"Depth camera device: {depth_device}")
    
    # Regular
    node_cam_reg = Node(    
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'usb_cam',
        output     = 'screen',
        parameters = [{'camera_name':  'logitech'},
                      {'camera_info_url': '/home/robot/.ros/camera_info/logitech.yaml'},
                      {'video_device': device_dict['regular'][0]},
                      {'pixel_format': 'yuyv2rgb'},
                      {'image_width':  1920},
                      {'image_height': 1080},
                      {'framerate':    15.0},
                      {'brightness':   BRIGHTNESS},
                      {'contrast':     CONTRAST},
                      {'saturation':   SATURATION},
                      {'sharpness':    SHARPNESS},
                      {'gain':         GAIN},
                      {'autoexposure': AUTO_EXPOSURE},
                      {'exposure':     EXPOSURE},
                      {'auto_white_balance': AUTOWHITEBALANCE},
                      {'white_balance':      WHITE_BALANCE},
                      {'autofocus':    AUTO_FOCUS},
                      {'focus':        FOCUS},
                      ]
    )
    
    # Depth
    rsfile = os.path.join(pkgdir('realsense2_camera'), 'launch/rs_launch.py')
    rsargs = {'camera_name':             'camera',  # camera unique name
              'video_device':            device_dict['depth'][0],  # depth camera device
              'depth_module.profile':    '0,0,0',   # depth W, H, FPS
              'rgb_camera.profile':      '0,0,0',   # color W, H, FPS
              'enable_color':            'true',    # enable color stream
              'enable_infra1':           'false',   # enable infra1 stream
              'enable_infra2':           'false',   # enable infra2 stream
              'enable_depth':            'true',    # enable depth stream
              'align_depth.enable':      'false',   # enabled aligned depth
              'pointcloud.enable':       'true',   # Turn on point cloud
              'allow_no_texture_points': 'true'}    # All points without texture
    node_cam_depth = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rsfile),
        launch_arguments=rsargs.items())

    # Select the camera node and remappings based on the CAMERA_TYPE
    node_cam = None
    remappings = [('/image_raw', '/usb_cam/image_raw')]
    if EE_DEPTH_SCAN:
        remappings.extend([
            ('/ee_cam/depth/points', '/camera/camera/depth/color/points'),
        ])
    if VISUAL_GRIP_CHECK or TEST_GRIPPER:
        remappings.extend([
            ('/ee_cam/color/image_raw', '/camera/camera/color/image_raw'),
        ])

    # Configure the camera detector node
    node_camera_detector = Node(
        name       = 'detector',
        package    = 'legobuilder',
        executable = 'detector',
        output     = 'screen',
        remappings = remappings,
    )
    
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST

    nodes = [
        node_rviz,
        node_cam_reg,
        node_camera_detector
    ] + ([node_cam_depth] if (EE_DEPTH_SCAN or VISUAL_GRIP_CHECK or TEST_GRIPPER) else [])

    if RECORD_BAG and EE_DEPTH_SCAN:
        bag_dir = os.path.expanduser(
            f'~/robotws/data/rosbags/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        nodes.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'record',
                 '/joint_states',
                 '/camera/camera/depth/color/points',
                 '-o', bag_dir],
            output='screen',
        ))

    return LaunchDescription(nodes)
