"""Launch the 134 HEBI demo

This instantiates the HEBI node and runs the simple 134 demo node.

"""

from launch                            import LaunchDescription
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the hebi interface.
    node_hebi = Node(
        name       = 'hebi', 
        package    = 'hebiros',
        executable = 'hebinode',
        output     = 'screen',
        parameters = [{'family': 'robotlab'},
                      {'motors': ['4.5', '4.4', '4.3']},
                      {'joints': ['base', 'shoulder', 'elbow']}])

    # Configure a node for the simple demo.
    node_demo = Node(
        name       = 'demo', 
        package    = 'demo134',
        executable = 'goals1',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the hebi and demo nodes.
        node_hebi,
        node_demo,
    ])
