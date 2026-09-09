"""Pure-function ROS message builders for the Detector node."""

import math

import numpy as np

from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import ColorRGBA, Header
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

from legobuilder_interfaces.msg import ContourInfoArray, ContourInfoMsg

from legobuilder.config import INNER_RADIUS, OUTER_RADIUS, BLOCK_SIZE
from legobuilder.schemas import ContourInfo


# ── Contour Serialization ─────────────────────────────────────────


def build_contour_info_array(
    contour_infos: list[ContourInfo],
    stamp,
) -> ContourInfoArray:
    """Serialize a ContourInfo list to a ContourInfoArray ROS message.

    Arguments
    ---------
    contour_infos : list[ContourInfo]
        Detected contours from the current frame.
    stamp
        A ROS builtin_interfaces/Time message (e.g. from
        clock.now().to_msg()).

    Returns
    -------
    ContourInfoArray
        ROS message ready for publishing.
    """
    msg = ContourInfoArray()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = 'world'

    for ci in contour_infos:
        contour_msg = ContourInfoMsg()
        contour_msg.timestamp = ci.t
        contour_msg.frame_idx = ci.frame_idx
        contour_msg.color = ci.color.value

        if ci.is_circle:
            contour_msg.shape_type = ContourInfoMsg.CIRCLE
            contour_msg.center_x = float(ci.center_xy[0])
            contour_msg.center_y = float(ci.center_xy[1])
            contour_msg.radius_px = ci.radius
        elif ci.is_rectangle:
            contour_msg.shape_type = ContourInfoMsg.RECTANGLE
            contour_msg.center_x = float(ci.center_xy[0])
            contour_msg.center_y = float(ci.center_xy[1])
            contour_msg.length_px = int(ci.length_px)
            contour_msg.width_px = int(ci.width_px)
            contour_msg.angle = (
                float(ci.angle)
                if ci.angle is not None else 0.0)
            if ci.quaternion is not None:
                contour_msg.quaternion = Quaternion(
                    x=ci.quaternion[0],
                    y=ci.quaternion[1],
                    z=ci.quaternion[2],
                    w=ci.quaternion[3],
                )
            if ci.corner_xys is not None:
                contour_msg.corner_world_xs = [
                    float(c[0]) for c in ci.corner_xys]
                contour_msg.corner_world_ys = [
                    float(c[1]) for c in ci.corner_xys]
        elif ci.is_square:
            contour_msg.shape_type = ContourInfoMsg.SQUARE
            contour_msg.center_x = float(ci.center_xy[0])
            contour_msg.center_y = float(ci.center_xy[1])
            contour_msg.center_z = float(ci.center_z)
            contour_msg.center_u = int(ci.center_uv[0])
            contour_msg.center_v = int(ci.center_uv[1])
            contour_msg.length_px = int(ci.size_px) 
            contour_msg.width_px = int(ci.size_px)
            contour_msg.angle = float(ci.angle) if ci.angle is not None else 0.0
            contour_msg.corner_world_xs = [float(c[0]) for c in ci.corner_xys]
            contour_msg.corner_world_ys = [float(c[1]) for c in ci.corner_xys]
            contour_msg.corner_world_us = [int(c[0]) for c in ci.corner_uvs]
            contour_msg.corner_world_vs = [int(c[1]) for c in ci.corner_uvs]
            if ci.face_conns_xyzs:
                contour_msg.face_conns_xs = [float(c[0]) for c in ci.face_conns_xyzs]
                contour_msg.face_conns_ys = [float(c[1]) for c in ci.face_conns_xyzs]
                contour_msg.face_conns_zs = [float(c[2]) for c in ci.face_conns_xyzs]
            if ci.quaternion is not None:
                contour_msg.quaternion = Quaternion(
                    x=ci.quaternion[0],
                    y=ci.quaternion[1],
                    z=ci.quaternion[2],
                    w=ci.quaternion[3],
                )

        msg.contours.append(contour_msg)

    return msg


# ── World Map Helpers ──────────────────────────────────────────────


def process_ee_pointcloud(msg, logger):
    """Extract XYZRGB array from a PointCloud2 message.

    Arguments
    ---------
    msg : PointCloud2
        End-effector depth camera point cloud.
    logger
        ROS logger for debug output.

    Returns
    -------
    tuple[np.ndarray, Header] or None
        (Nx6 XYZRGB array, depth stamp header) on success, or
        None if the point cloud is empty.
    """
    xyz = pc2.read_points_numpy(
        msg, field_names=('x', 'y', 'z'), skip_nans=True)
    if xyz.size == 0:
        return None

    # RealSense packs RGB as a float32 viewed as uint32: 0x00RRGGBB
    rgb_packed = pc2.read_points_numpy(
        msg, field_names=('rgb',), skip_nans=True)
    rgb_uint32 = rgb_packed.view(np.uint32).reshape(-1)
    r = ((rgb_uint32 >> 16) & 0xFF).astype(np.float64)
    g = ((rgb_uint32 >> 8) & 0xFF).astype(np.float64)
    b = (rgb_uint32 & 0xFF).astype(np.float64)

    xyzrgb = np.column_stack((xyz, r, g, b))

    stamp_msg = Header()
    stamp_msg.stamp = msg.header.stamp
    stamp_msg.frame_id = msg.header.frame_id

    logger.debug(
        f"[RECV ee_pointcloud] {len(xyz)} points (XYZRGB)")

    return xyzrgb, stamp_msg


def build_world_map_msg(
    world_map, stamp, enable_caching,
    pointcloud_dirty, cached_msg, logger,
):
    """Build a PointCloud2 message from the world map.

    Pure function: takes state in, returns state out.  The caller
    publishes the returned message and stores the updated cache.

    Arguments
    ---------
    world_map : WorldMap
        The 3D voxel map instance.
    stamp
        A ROS builtin_interfaces/Time message for the header.
    enable_caching : bool
        Whether PointCloud caching is enabled.
    pointcloud_dirty : bool
        True if the cache needs rebuilding.
    cached_msg
        Previously cached PointCloud2 message, or None.
    logger
        ROS logger for debug output.

    Returns
    -------
    tuple or None
        (msg_to_publish, new_cached_msg, new_dirty_flag) on success,
        or None if the world map has no data.
    """
    if not world_map.has_data:
        return None

    if (enable_caching
            and not pointcloud_dirty
            and cached_msg is not None):
        return (cached_msg, cached_msg, False)

    msg = world_map.to_pointcloud2_msg(stamp=stamp)
    if msg is not None:
        logger.debug(
            f"[PUB detector/world_map] {msg.width} points"
        )
        return (msg, msg, False)

    return None


def build_world_bounds_marker(world_map_config, stamp):
    """Build workspace bounds as a wireframe annular cylinder Marker.

    Arguments
    ---------
    world_map_config : WorldMapConfig
        World map configuration with workspace bounds.
    stamp
        A ROS builtin_interfaces/Time message for the header.

    Returns
    -------
    Marker
        A LINE_LIST Marker representing the workspace bounds.
    """
    cx = float(world_map_config.workspace_base_xy[0])
    cy = float(world_map_config.workspace_base_xy[1])
    r_in = INNER_RADIUS
    r_out = OUTER_RADIUS
    z_lo = world_map_config.workspace_z_min
    z_hi = world_map_config.workspace_z_max

    m = Marker()
    m.header.frame_id = 'world'
    m.header.stamp = stamp
    m.ns = 'world_bounds'
    m.id = 0
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = 0.01
    m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)
    m.pose.orientation.w = 1.0

    n_seg = 128

    def arc_pts(radius, z):
        """Return arc points clipped to x>0, y>0."""
        pts = []
        for i in range(n_seg):
            angle = 2.0 * math.pi * i / n_seg
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            if x > 0.0 and y > 0.0:
                pts.append(Point(x=x, y=y, z=z))
        return pts

    for radius in (r_in, r_out):
        for z in (z_lo, z_hi):
            pts = arc_pts(radius, z)
            for i in range(len(pts) - 1):
                m.points.append(pts[i])
                m.points.append(pts[i + 1])

    n_vert = 24
    for i in range(n_vert):
        angle = 2.0 * math.pi * i / n_vert
        for radius in (r_in, r_out):
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            if x > 0.0 and y > 0.0:
                m.points.append(
                    Point(x=x, y=y, z=z_lo))
                m.points.append(
                    Point(x=x, y=y, z=z_hi))

    return m


# ── Worldmap Contour Markers ─────────────────────────────────────────


_COLOR_RGBA = None


def _get_color_rgba_map():
    """Lazy-load color-to-RGBA mapping."""
    global _COLOR_RGBA
    if _COLOR_RGBA is None:
        from legobuilder.schemas import Color
        _COLOR_RGBA = {
            Color.YELLOW: ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8),
            Color.BLUE:   ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),
            Color.GREEN:  ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),
            Color.RED:    ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
            Color.ORANGE: ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8),
        }
    return _COLOR_RGBA


def build_worldmap_marker_array(
    level_results: list,
    block_height: float,
    stamp,
) -> MarkerArray:
    """Build a MarkerArray of CUBE markers for worldmap-detected blocks.

    Arguments
    ---------
    level_results : list[tuple[int, list[ContourInfo]]]
        (level, contour_infos) pairs from perceive_from_worldmap.
    block_height : float
        Block thickness in meters.
    stamp
        ROS builtin_interfaces/Time message.

    Returns
    -------
    MarkerArray
        Array of CUBE markers for RViz visualization.
    """
    color_map = _get_color_rgba_map()

    marker_array = MarkerArray()

    # DELETE_ALL to clear stale markers
    delete_marker = Marker()
    delete_marker.header.frame_id = 'world'
    delete_marker.header.stamp = stamp
    delete_marker.ns = 'worldmap_blocks'
    delete_marker.id = 0
    delete_marker.action = Marker.DELETEALL
    marker_array.markers.append(delete_marker)

    marker_id = 1
    for level, contour_infos in level_results:
        for ci in contour_infos:
            if not ci.is_square:
                continue

            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = stamp
            m.ns = 'worldmap_blocks'
            m.id = marker_id
            marker_id += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position = Point(
                x=float(ci.center_xy[0]),
                y=float(ci.center_xy[1]),
                z=float(ci.center_z),
            )

            if ci.quaternion is not None:
                m.pose.orientation = Quaternion(
                    x=ci.quaternion[0],
                    y=ci.quaternion[1],
                    z=ci.quaternion[2],
                    w=ci.quaternion[3],
                )
            else:
                m.pose.orientation.w = 1.0

            m.scale.x = block_height
            m.scale.y = block_height
            m.scale.z = block_height

            m.color = color_map.get(
                ci.color,
                ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8),
            )

            marker_array.markers.append(m)

            # Add smaller cylinder markers for face connections moving out of the faces of the block
            for face in ci.face_conns_xyzs:
                face_marker = Marker()
                face_marker.header.frame_id = 'world'
                face_marker.header.stamp = stamp
                face_marker.ns = f'worldmap_blocks_faces'
                face_marker.id = marker_id
                marker_id += 1
                face_marker.type = Marker.CYLINDER
                face_marker.action = Marker.ADD
                face_marker.pose.position = Point(
                    x=float(face[0]),
                    y=float(face[1]),
                    z=float(face[2]),
                )
                # Make the orientation of the face marker move out of the block's face
                # The cylinder's z axis should be 
                face_marker.pose.orientation = m.pose.orientation
                face_marker.scale.x = block_height * 0.5
                face_marker.scale.y = block_height * 0.5
                face_marker.scale.z = block_height * 0.1
                face_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)
                marker_array.markers.append(face_marker)
    return marker_array


def build_grid_corner_markers(corners, block_height, stamp) -> MarkerArray:
    """Build purple cylinder markers at each grid corner.

    Arguments
    ---------
    corners : list[tuple[float, float]]
        (x, y) positions of grid corners.
    block_height : float
        Cylinder height in meters.
    stamp
        ROS builtin_interfaces/Time message.

    Returns
    -------
    MarkerArray
        Array of CYLINDER markers for RViz visualization.
    """
    marker_array = MarkerArray()
    for i, (cx, cy) in enumerate(corners):
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = stamp
        m.ns = 'grid_corners'
        m.id = i
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = cx
        m.pose.position.y = cy
        m.pose.position.z = block_height / 2
        m.pose.orientation.w = 1.0
        m.scale.x = 0.01   # diameter (radius 0.005)
        m.scale.y = 0.01
        m.scale.z = block_height
        m.color = ColorRGBA(r=0.5, g=0.0, b=0.5, a=0.8)
        marker_array.markers.append(m)
    return marker_array
