"""Launch the brain node.

The brain node coordinates between detector and manipulator:
- Subscribes to: detector/contour_info (raw contour data)
- Publishes to: brain/trajectory_states (pre-computed trajectories with IK)

Requires:
- robot_state_publisher to be running (for URDF/KinematicChain)
- detector node to be running (publishes contour_info)

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate/load the robot's URDF file (XML).
    urdf = os.path.join(pkgdir('legobuilder'), 'urdf/fivedof.urdf')
    with open(urdf, 'r') as file:
        robot_description = file.read()


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the robot_state_publisher.
    # Required for brain to access URDF for kinematic chain.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher_brain',
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    # Configure the brain node
    node_brain = Node(
        name       = 'brain',
        package    = 'legobuilder',
        executable = 'brain',
        output     = 'screen',
    )

    # Configure the IK solver node (runs IK in a separate process
    # so that the brain's executor is never blocked by IK computation)
    node_ik_solver = Node(
        name       = 'ik_solver',
        package    = 'legobuilder',
        executable = 'ik_solver',
        output     = 'screen',
    )


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST

    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the nodes.
        # node_robot_state_publisher,
        node_ik_solver,
        node_brain,
    ])
