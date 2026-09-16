"""Example Launch File (Possible Pieces to Use)

   This launch file is intended show how the pieces come together.

   This could start
     1)  RVIZ, ready to view the robot
     2)  The robot_state_publisher (likely listening to default /joint_states)
     3)  The HEBI node to communicate with the motors
     4a) The GUI,
      (i)  by default sending /joint_states (substituting the robot)
      (ii) sending /joint_commmands at 100Hz (in place of your code)
     4b) Or the trajectory code (sending /joint_commands at 100Hz)

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

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('threedof'), 'rviz/viewurdf.rviz')

    # Locate/load the robot's URDF file (XML).
    # urdf = os.path.join(pkgdir('threedof'), 'urdf/threedof.urdf')
    urdf = os.path.join(pkgdir('threedof'), 'urdf/threedofexample.urdf')
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
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    # Configure a node for the hebi interface.
    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family':   'robotlab'},
                      {'motors':   ['7.5',  '7.4',      '7.3']},
                      {'joints':   ['base', 'shoulder', 'elbow']}],
        on_exit    = Shutdown())

    # Configure a node for the GUI.  By default this sends to
    # /joint_states at 10Hz and substitutes for the robot.
    # Alternatively, send to /joint_commands at 100Hz to drive the
    # robot.
    node_gui_REPLACE_ROBOT = Node(
        name       = 'gui', 
        package    = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        output     = 'screen',
        on_exit    = Shutdown())

    node_gui_DRIVE_ROBOT = Node(
        name       = 'gui', 
        package    = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        output     = 'screen',
        parameters = [{'rate': 100}],
        remappings = [('/joint_states', '/joint_commands')],
        on_exit    = Shutdown())

    # Configure a trajectory node.  PLACEHOLDER FOR YOUR CODE!!
    node_trajectory = Node(
        name       = 'touchtable', 
        package    = 'threedof',
        executable = 'touchtable',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # STEP 1: Test the URDF.
        node_rviz,
        node_robot_state_publisher,
        node_gui_REPLACE_ROBOT,

        # # STEP 2: Watch/move robot by hand.
        # node_rviz,
        # node_robot_state_publisher,
        # node_hebi,

        # # STEP 3: Watch/move the robot by GUI.
        # node_rviz,
        # node_robot_state_publisher,
        # node_hebi,
        # node_gui_DRIVE_ROBOT,

        # # STEP 4: Let your code drive the robot.
        # node_rviz,
        # node_robot_state_publisher,
        # node_hebi,
        # node_trajectory,
    ])
