'''hsvtune.py

   Create trackbars to continually adjust the HSV limits.


   Node:        /hsvtune
   Subscribe:   /image_raw                      Source image
   Publish:     /hsvtune/image_raw              Output image
                /hsvtune/binary                 HSV thresholded binary image
'''

import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from asyncio            import Future
from rclpy.node         import Node
from sensor_msgs.msg    import Image

from detectors.config import MIN_H, MAX_H, MIN_S, MAX_S, MIN_V, MAX_V

#
#  Use Trackbars to Vary HSV Limits
#
# A single trackbar object
class TrackBar():
    def __init__(self, winname, barname, hsvlimits, channel, element, maximum):
        # Store the parameters.
        self.winname   = winname
        self.barname   = barname
        self.hsvlimits = hsvlimits
        self.channel   = channel
        self.element   = element
        # Create the trackbar.
        cv2.createTrackbar(barname, winname,
                           hsvlimits[channel,element], maximum, self.CB)

    def CB(self, val):
        # Make sure the threshold doesn't pass the opposite limit.
        if self.element == 0:  val = min(val, self.hsvlimits[self.channel,1])
        else:                  val = max(val, self.hsvlimits[self.channel,0])
        # Update the threshold and the tracker position.
        self.hsvlimits[self.channel,self.element] = val
        cv2.setTrackbarPos(self.barname, self.winname, val)

# A combined HSV limit tracker.
class HSVTracker():
    def __init__(self, hsvlimits):
        # Create a controls window for the trackbars.
        winname = 'Controls'
        cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

        # Show the control window.  Note this won't actually appear/
        # update (draw on screen) until waitKey(1) is called below.
        cv2.imshow(winname, np.zeros((1, 500, 3), np.uint8))

        # Create trackbars for each limit.
        TrackBar(winname, 'Lower H', hsvlimits, 0, 0, 179)
        TrackBar(winname, 'Upper H', hsvlimits, 0, 1, 179)
        TrackBar(winname, 'Lower S', hsvlimits, 1, 0, 255)
        TrackBar(winname, 'Upper S', hsvlimits, 1, 1, 255)
        TrackBar(winname, 'Lower V', hsvlimits, 2, 0, 255)
        TrackBar(winname, 'Upper V', hsvlimits, 2, 1, 255)

    def update(self):
        # Call waitKey(1) to force the window to update.
        cv2.waitKey(1)


#
#  HSVTune Node Class
#
class HSVTuneNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        # Thresholds in Hmin/max, Smin/max, Vmin/max
        self.hsvlimits = np.array([[MIN_H, MAX_H], [MIN_S, MAX_S], [MIN_V, MAX_V]])

        # Create trackbars to vary the thresholds.
        self.tracker = HSVTracker(self.hsvlimits)
        self.get_logger().info("Allowing HSV limits to vary...")

        # Create a publisher for the processed (debugging) images.
        # Store up to three images, just in case.
        self.pubrgb = self.create_publisher(Image, '~/image_raw', 3)
        self.pubbin = self.create_publisher(Image, '~/binary',    3)

        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.create_subscription(Image, '/image_raw', self.process, 1)

        # Report.
        self.get_logger().info("HSV tuner running...")

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Process the image (detect the ball).
    def process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # Update the HSV limits (updating the control window).
        self.tracker.update()

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary = cv2.inRange(hsv, self.hsvlimits[:,0], self.hsvlimits[:,1])

        # Grab the image shape, determine the center pixel.
        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # Draw the center lines.  Note the row is the first dimension.
        frame = cv2.line(frame, (uc,0), (uc,H-1), (255, 255, 255), 1)
        frame = cv2.line(frame, (0,vc), (W-1,vc), (255, 255, 255), 1)

        # Report the center HSV values.  Note the row comes first.
        # self.get_logger().info(
        #     "Center pixel HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))

        # Convert the frame back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Also publish the thresholded binary (black/white) image.
        self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    # Instantiate the detector node.
    node = HSVTuneNode('hsvtune', future)

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
