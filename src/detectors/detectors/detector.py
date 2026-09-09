#!/usr/bin/env python3
#
#   balldetector.py
#
#   Detect the tennis balls with OpenCV.
#
#   Node:           /balldetector
#   Subscribers:    /usb_cam/image_raw          Source image
#   Publishers:     /balldetector/binary        Intermediate binary image
#                   /balldetector/image_raw     Debug (marked up) image
#
import cv2
import numpy as np
from pydantic import BaseModel
from typing import Union

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point, Pose, Quaternion

from detectors.config import MIN_H, MAX_H, MIN_S, MAX_S, MIN_V, MAX_V
from detectors.config import ARUCO_SETTINGS
from detectors.config import WORLD_U, WORLD_V
from detectors.config import CIRCLE_STABILITY_THRESHOLD, RECTANGLE_STABILITY_THRESHOLD
from detectors.config import CIRCLE_BUFFER_DURATION, RECTANGLE_BUFFER_DURATION, T_CHECK_CIRCLE, T_CHECK_RECTANGLE, USE_MULTIPLE_PERSPECTIVES

class ContourInfo(BaseModel):
    t: float  # timestamp in seconds
    is_circle: bool = False
    is_rectangle: bool = False

class CircleContourInfo(ContourInfo):
    center_uv: tuple[int, int]
    center_xy: tuple[float, float]
    radius: int
    
class RectangleContourInfo(ContourInfo):
    center_uv: tuple[int, int]
    center_xy: tuple[float, float]
    length: int
    width: int
    corner_uvs: list[tuple[int, int]] = None
    corner_xys: list[tuple[float, float]] = None
    angle: float = None            # rad north of x-axis
    quaternion: tuple[float, float, float, float] = None  # [x, y, z, w]

#
#  Detector Node Class
#
class DetectorNode(Node):
    # Pick some colors, assuming RGB8 encoding.
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)

    # Initialization.
    def __init__(self, name):
        super().__init__(name)

        self.hsvlimits = np.array([[MIN_H, MAX_H], [MIN_S, MAX_S], [MIN_V, MAX_V]])
        self.perspective_transforms: dict[str, np.ndarray] = {}   # {tl/br/... : np.ndarray}
        self.circle_contour_buffer = {
            'duration': CIRCLE_BUFFER_DURATION,  # seconds
            'contour_infos': []   # list of CircleContourInfo
        }
        self.rectangle_contour_buffer = {
            'duration': RECTANGLE_BUFFER_DURATION,  # seconds
            'contour_infos': []   # list of RectangleContourInfo
        }
        self.start_time = self.get_clock().now()

        ### Publishers ###
        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)
        self.pubbin = self.create_publisher(Image, name+'/binary',    3)   # filter for orangy color

        # Create a publisher to send object loca
        self.pub_disk = self.create_publisher(Point, name+'/disk_info', 10)
        self.pub_strip = self.create_publisher(Pose, name+'/strip_info', 10)

        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

        # Report.
        self.get_logger().info("Ball detector running...")
        
    def get_t(self):
        now = self.get_clock().now()
        return (now - self.start_time).nanoseconds * 1e-9

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()
        
    def set_perspective_transforms(self, frame: np.ndarray | None):
        if frame is None:
            self.get_logger().warning("set_perspective_transform: Frame is None")
            return
        
        perspective_transforms = {}
        missing_markers = False
        
        for name, _ in ARUCO_SETTINGS.items():
            if name == 'tl':
                mask_incl = np.zeros(frame.shape[:2], dtype="uint8")
                mask_incl[0:frame.shape[0]//2, 0:frame.shape[1]//2] = 1
            elif name == 'tr':
                mask_incl = np.zeros(frame.shape[:2], dtype="uint8")
                mask_incl[0:frame.shape[0]//2, frame.shape[1]//2:frame.shape[1]] = 1
            elif name == 'bl':
                mask_incl = np.zeros(frame.shape[:2], dtype="uint8")
                mask_incl[frame.shape[0]//2:frame.shape[0], 0:frame.shape[1]//2] = 1
            elif name == 'br':
                mask_incl = np.zeros(frame.shape[:2], dtype="uint8")
                mask_incl[frame.shape[0]//2:frame.shape[0], frame.shape[1]//2:frame.shape[1]] = 1
            else:
                self.get_logger().warning(f"set_perspective_transform: Unknown Aruco setting name {name}")
                continue
            new_frame = cv2.bitwise_and(frame, frame, mask=mask_incl)
            
            # Detect the Aruco markers (using the 4X4 dictionary).
            markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
                new_frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

            # Abort if not all markers are detected.
            if (markerIds is None or len(markerIds) != 4 or
                set(markerIds.flatten()) != set([1,2,3,4])):
                self.get_logger().warning("set_perspective_transform: Not all markers detected")
                missing_markers = True
                continue

            # Determine the center of the marker pixel coordinates.
            uvMarkers = np.zeros((4,2), dtype='float32')
            for i in range(4):
                uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)
                
            # Calculate the matching World coordinates of the 4 Aruco markers.
            x0 = 0
            y0 = 0
            DX = 0.1016
            DY = 0.06985
            xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
                                    [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])
        
            M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)
            perspective_transforms[name] = M
        
        if missing_markers and USE_MULTIPLE_PERSPECTIVES:
            self.get_logger().warning("set_perspective_transform: Missing markers for multiple perspectives, not setting transforms")
            return
        else:
            self.perspective_transforms = perspective_transforms
            
        self.get_logger().info(f"Perspective transforms: {self.perspective_transforms}")
        
    def pixel_to_world(self, u: int, v: int, use_multiple_perspectives: bool = True) -> tuple[float, float]:
        def _pixel_to_world(uv: tuple[int, int], M: np.ndarray, aruco_cnt_xy: tuple[float, float]) -> tuple[float, float]:
            u, v = uv
            cnt_x, cnt_y = aruco_cnt_xy
            uvObj = np.float32([u, v])
            xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)
            world_coord = (cnt_x + xyObj[0], cnt_y + xyObj[1])
            return world_coord
        
        # Interpolate the pixel to world coordinate using multiple perspective transforms
        if use_multiple_perspectives:
            world_coords = []
            for name, M in self.perspective_transforms.items():
                aruco_settings = ARUCO_SETTINGS[name]
                world_coord = _pixel_to_world((u, v), M, (aruco_settings['cnt_x'], aruco_settings['cnt_y']))
                world_coords.append(world_coord)
            world_coords = np.array(world_coords)
            avg_world_coord = (np.mean(world_coords[:,0]), np.mean(world_coords[:,1]))
            return avg_world_coord
        else:
            # use top-left perspective transform only
            first_name = list(self.perspective_transforms.keys())[0]
            aruco_settings = ARUCO_SETTINGS[first_name]
            M = self.perspective_transforms[first_name]
            aruco_cnt_xy = (aruco_settings['cnt_x'], aruco_settings['cnt_y'])
            return _pixel_to_world((u, v), M, aruco_cnt_xy)
    
    def _get_frame_info(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # Convert to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue
        
        return frame, frame_hsv
    
    def _process_on_center_pixel(self, frame: np.ndarray):
        # Grab the image shape, determine the center pixel.
        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # Help to determine the HSV range...
        if True:
            # Draw the center lines.  Note the row is the first dimension.
            frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
            frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)

            # Report the center HSV values.  Note the row comes first.
            # self.get_logger().info(
            #     "HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))
    
    def _get_binary(self, frame_hsv: np.ndarray) -> np.ndarray:
        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary = cv2.inRange(frame_hsv, self.hsvlimits[:,0], self.hsvlimits[:,1])
        return binary
    
    def get_contour_info(self, contour: np.ndarray) -> Union[CircleContourInfo, None]:
        def _is_contour_circle(contour: np.ndarray, tolerance: float = 0.2) -> (bool, CircleContourInfo):
            ((ur, vr), radius) = cv2.minEnclosingCircle(contour)
            ur     = int(ur)
            vr     = int(vr)
            radius = int(radius)
            
            theoretical_area = np.pi * (radius ** 2)
            actual_area = cv2.contourArea(contour)
            is_circle = abs(theoretical_area - actual_area) / theoretical_area < tolerance
            
            if not is_circle:
                return is_circle, None
            
            center_xy: tuple[float, float] = self.pixel_to_world(ur, vr)
            circle_contour_info = CircleContourInfo(
                t=self.get_t(),
                is_circle=True,
                center_uv=(ur, vr),
                radius=radius,
                center_xy=center_xy
            )
            
            return is_circle, circle_contour_info
        
        def _is_contour_rectangle(contour: np.ndarray, tolerance: float = 0.2) -> (bool, RectangleContourInfo):
            rect = cv2.minAreaRect(contour)
            (ur, vr), (width, length), angle = rect
            corners = cv2.boxPoints(rect)
            
            # Check if area of enclosed circle much different than area of rectangle
            ((cu, cv), radius) = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            rectangle_area = width * length
            is_rectangle = abs(circle_area - rectangle_area) / rectangle_area > (1 - tolerance)
            
            if not is_rectangle:
                return is_rectangle, None
            
            corner_uvs = [(int(c[0]), int(c[1])) for c in corners]
            corner_xys = [self.pixel_to_world(c[0], c[1]) for c in corners]
            center_xy: tuple[float, float] = self.pixel_to_world(int(ur), int(vr))
            
            # Find corner xy with lowest y
            bottom_corner_index = np.argmin([c[1] for c in corner_xys])
            bottom_corner = corner_xys[bottom_corner_index]
            
            # Find distance to other corners
            dists = []
            for i, c in enumerate(corner_xys):
                if i != bottom_corner_index:
                    dist = np.sqrt((c[0] - bottom_corner[0])**2 + (c[1] - bottom_corner[1])**2)
                    dists.append(dist)
                else:
                    dists.append(0)  # Ignore self distance
            
            # Get index of long edge corner
            long_edge_index = np.argsort(dists)[2]  # 2nd largest distance
            long_edge_corner = corner_xys[long_edge_index]

            # Get the angle between the bottom corner and the second corner of the long edge using atan2
            delta_x = corner_xys[long_edge_index][0] - bottom_corner[0]
            delta_y = corner_xys[long_edge_index][1] - bottom_corner[1]
            angle = np.arctan2(delta_y, delta_x)  # radians
            
            # Compute quaternion
            half_angle = angle / 2.0
            qx = 0.0
            qy = 0.0
            qz = np.sin(half_angle)
            qw = np.cos(half_angle)
            quaternion = (qx, qy, qz, qw)
            
            rectangle_contour_info = RectangleContourInfo(
                t=self.get_t(),
                is_rectangle=True,
                center_uv=(int(ur), int(vr)),
                center_xy=center_xy,
                length=length,
                width=width,
                corner_uvs=corner_uvs,
                corner_xys=corner_xys,
                angle=angle,
                quaternion=quaternion
            )
            
            return is_rectangle, rectangle_contour_info
        
        is_circle, circle_contour_info = _is_contour_circle(contour)
        is_rectangle, rectangle_contour_info = _is_contour_rectangle(contour)
        
        if is_circle and is_rectangle:
            self.get_logger().warning("Contour is both circle and rectangle?")
        
        if is_circle and circle_contour_info is not None:
            return circle_contour_info
        elif is_rectangle and rectangle_contour_info is not None:
            return rectangle_contour_info
        
    def _update_contour_buffers(
        self,
        contour_infos: list[ContourInfo]
    ):
        for contour_info in contour_infos:
            if contour_info.is_circle:
                self.circle_contour_buffer['contour_infos'].append(contour_info)
            elif contour_info.is_rectangle:
                self.rectangle_contour_buffer['contour_infos'].append(contour_info)
        
        t = self.get_t()
        while \
          len(self.circle_contour_buffer['contour_infos']) > 0 \
          and t - self.circle_contour_buffer['contour_infos'][0].t > self.circle_contour_buffer['duration']:
            self.circle_contour_buffer['contour_infos'].pop(0)
        while \
          len(self.rectangle_contour_buffer['contour_infos']) > 0 \
          and t - self.rectangle_contour_buffer['contour_infos'][0].t > self.rectangle_contour_buffer['duration']:
            self.rectangle_contour_buffer['contour_infos'].pop(0)
        
    def draw_annotated_shape_contour(self, frame: np.ndarray, contour_info: ContourInfo):
        if contour_info.is_circle:
            (ur, vr) = contour_info.center_uv
            radius = contour_info.radius
            center_xy = contour_info.center_xy
        
            cv2.circle(frame, (ur, vr), int(radius), self.yellow,  2)
            cv2.circle(frame, (ur, vr), 3,           self.red,    -1)
            
            cv2.putText(
                frame, 
                f"({center_xy[0]:.2f}, {center_xy[1]:.2f})", 
                (ur-50, vr-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2, cv2.LINE_AA)
            
        elif contour_info.is_rectangle:
            (ur, vr) = contour_info.center_uv
            corner_uvs = contour_info.corner_uvs
            center_xy = contour_info.center_xy
            
            for i in range(len(corner_uvs)):
                u1, v1 = corner_uvs[i]
                u2, v2 = corner_uvs[(i+1) % len(corner_uvs)]
                cv2.line(frame, (u1, v1), (u2, v2), self.yellow, 2)
            cv2.circle(frame, (ur, vr), 3, self.red, -1)
            
            cv2.putText(
                frame, 
                f"({center_xy[0]:.2f}, {center_xy[1]:.2f}), {np.degrees(contour_info.angle):.1f} deg", 
                (ur-50, vr-15), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2, cv2.LINE_AA
            )
                
        else:
            self.get_logger().warning("Contour is neither circle nor rectangle?")

    def _should_publish_contour(self, contour_info: ContourInfo) -> bool:
        """
        Decide whether to publish the detected contour based on stability over time.
        """
        cnt_x, cnt_y = contour_info.center_xy

        # Prepare the contour array and parameters based on shape type
        if contour_info.is_circle:
            cnt_arr = np.array([cnt_x, cnt_y])
            # Check if last t and most recent t are within some t_check
            t_check = T_CHECK_CIRCLE
            stability_threshold = CIRCLE_STABILITY_THRESHOLD
            buffer = self.circle_contour_buffer['contour_infos']
        elif contour_info.is_rectangle:
            x, y, z, w = contour_info.quaternion
            cnt_arr = np.array([cnt_x, cnt_y, x, y, z, w])
            # Check if last t and most recent t are within some t_check
            t_check = T_CHECK_RECTANGLE
            stability_threshold = RECTANGLE_STABILITY_THRESHOLD
            buffer = self.rectangle_contour_buffer['contour_infos']
        else:
            return False
        
        # Check if enough elements in buffer and buffer spans T_CHECK
        if len(buffer) > 2 and \
           buffer[-1].t - buffer[0].t >= t_check:
            # If so, ensure that each element in the buffer is approximately the same
            total_diff = 0
            for buffer_contour in buffer:
                if contour_info.is_circle:
                    buffer_cnt_arr = np.array(buffer_contour.center_xy)
                elif contour_info.is_rectangle:
                    buffer_cnt_arr = np.array([
                        buffer_contour.center_xy[0],
                        buffer_contour.center_xy[1],
                        buffer_contour.quaternion[0],
                        buffer_contour.quaternion[1],
                        buffer_contour.quaternion[2],
                        buffer_contour.quaternion[3]
                    ])
                total_diff += np.linalg.norm(buffer_cnt_arr - cnt_arr)
            # self.get_logger().info(f"Total diff for publishing check: {total_diff:.4f}")
            if total_diff < stability_threshold:
                return True
        return False
    
    def _get_contours(self, binary: np.ndarray) -> tuple[np.ndarray]:
        # Erode and Dilate. Definitely adjust the iterations!
        iter = 2
        binary = cv2.erode( binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode( binary, None, iterations=iter)

        # Find contours in the mask and initialize the current
        # (x, y) center of the ball
        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # TODO: send hierarchy when needed in future
        return contours
        
    
    def publish_detected_shape(self, contour_info: ContourInfo):
        # Only publish if all elements in buffer are approximately the same
        if self._should_publish_contour(contour_info):
            if contour_info.is_circle:
                cnt_x, cnt_y = contour_info.center_xy
                self.pub_disk.publish(
                    Point(
                        x=cnt_x, 
                        y=cnt_y, 
                        z=0.0
                    )
                )
                # self.get_logger().info(f"Published disk at ({cnt_x:.2f}, {cnt_y:.2f})")
            elif contour_info.is_rectangle:
                cnt_x, cnt_y = contour_info.center_xy
                x, y, z, w = contour_info.quaternion
                self.pub_strip.publish(
                    Pose(
                        position=Point(x=cnt_x, y=cnt_y, z=0.0), 
                        orientation=Quaternion(
                            x=x, y=y, z=z, w=w
                        )
                    )
                )
                # self.get_logger().info(f"Published strip at ({cnt_x:.2f}, {cnt_y:.2f}), angle {np.degrees(contour_info.angle):.1f} deg")
            else:
                self.get_logger().warning("Contour is neither circle nor rectangle?")
    
    # Process the image (detect the ball).
    def process(self, msg):
        # Get last frame + hsv version
        frame, frame_hsv = self._get_frame_info(msg)
        
        # Draw where world origin is
        # H, W, D = frame.shape
        # cv2.circle(frame, (WORLD_U, H - WORLD_V), 5, self.green, -1)
        
        # Set perspective transform (to convert uv to xy)
        
        if self.perspective_transforms == {}:
            self.set_perspective_transforms(frame)
            
            if self.perspective_transforms == {}:
                self.get_logger().warning("Perspective transform not set. Either markers not present or obstructed by objects.")
                return
            
            self.get_logger().info("Perspective transform set")
        
        # Optional to show center line
        # self._process_on_center_pixel(frame)
        
        # Filter for orange color
        binary = self._get_binary(frame_hsv)
        
        # Get contours (should be disk and/or strip)
        contours = self.detector_get_contours(binary)
        # cv2.drawContours(frame, contours, -1, self.blue, 2)
        
        # Process each contour
        for contour in contours:
            contour_info = self.get_contour_info(contour)
            if contour_info is None:
                continue
            
            self._update_contour_buffers([contour_info])
            self.draw_annotated_shape_contour(frame, contour_info)
            self.publish_detected_shape(contour_info)

        # Convert the frame back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Also publish the binary (black/white) image.
        self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))
        

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('detector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
