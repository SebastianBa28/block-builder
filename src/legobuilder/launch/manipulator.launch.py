import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node

from legobuilder.config import JOINT_NAMES, MOTOR_NAMES


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('legobuilder'), 'rviz/urdf_image.rviz')

    # Locate/load the robot's URDF file (XML).
    urdf = os.path.join(pkgdir('legobuilder'), 'urdf/fivedof.urdf')
    with open(urdf, 'r') as file:
        robot_description = file.read()


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for RVIZ.
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher_manipulator', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}],
        remappings = [('/joint_states', '/joint_states_viz')])

    # Configure a node for the hebi interface.
    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family':   'robotlab'},
                      {'motors':   MOTOR_NAMES},
                      {'joints':   JOINT_NAMES},
                      {'lifetime': 200.0}   # 200ms instead of 50ms,
                    ],  
        on_exit    = Shutdown())

    # Configure a trajectory node.
    node_trajectory = Node(
        name       = 'manipulator', 
        package    = 'legobuilder',
        executable = 'manipulator',
        output     = 'screen',
    )


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([
        # node_rviz,
        node_robot_state_publisher,
        node_hebi,
        node_trajectory,
    ])
