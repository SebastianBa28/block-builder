'''demo134.py

   Demonstration of how to receive point commands!

   This simply reports the received point and does nothing else.  Copy
   the relevant pieces into your code.


   Node:        /receivepoint
   Subscribe:   /point                  geometry_msgs/Point
   Publish:

'''

import rclpy

from asyncio            import Future
from rclpy.node         import Node
from geometry_msgs.msg  import Point


#
#   DEMO Node Class
#
#   This inherits all the standard ROS node stuff, but adds callback
#   function to report the received point message.
#
#   Arguments are the node name and a future object (to force a shutdown).
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        # Create a subscriber to receive point messages.
        self.create_subscription(Point, '/point', self.recvpoint, 10)

        # Report.
        self.get_logger().info("Running %s" % name)

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Receive a point message - called by incoming messages.
    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        
        # Report.
        self.get_logger().info("Received point %r, %r, %r" % (x,y,z))
        
    


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal the node wants to stop.
    future = Future()

    # Instantiate the DEMO node.
    node = DemoNode('receivepoint', future)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the node is complete
    # (as signaled by the future object).
    rclpy.spin_until_future_complete(node, future)

    # Report the reason for shutting down.
    if future.done():
        node.get_logger().info("Stopping: " + future.result())
    else:
        node.get_logger().info("Stopping: Interrupted")

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
