"""Object processing pipeline for the brain node.

Buffers raw contour-derived objects, clusters them spatially with HDBSCAN,
and applies temporal stability filtering to produce reliable detections.
"""

import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import HDBSCAN, DBSCAN

from legobuilder.schemas import ObjectType, Object

from legobuilder.config import (
    BLOCK_SIZE,
    BUFFER_DURATION,
    CLUSTERING_SETTINGS,
    STABILITY_THRESHOLDS,
    PRESENCE_DURATION_THRESHOLD,
    ABSENCE_DURATION_TOLERANCE,
    SMOOTHING_WINDOW_DURATION,
)

# RGB visualization colors keyed by ObjectType
_OBJ_COLORS = {
    ObjectType.YELLOW_BLOCK: (255, 255, 0),
    ObjectType.BLUE_BLOCK:   (0, 0, 255),
    ObjectType.GREEN_BLOCK:  (0, 255, 0),
    ObjectType.RED_BLOCK:    (255, 0, 0),
    ObjectType.DISK:         (255, 165, 0),
    ObjectType.STRIP:        (128, 128, 128),
}


class ObjectProcessor:
    """Buffer, cluster, and detect stable objects from raw contour data.

    Implements a three-stage pipeline: (1) append new observations to a
    time-bounded buffer, (2) cluster buffered observations per object
    type using HDBSCAN, (3) detect stable objects by checking positional
    and angular displacement within each cluster over time.

    Attributes
    ----------
    logger
        ROS logger instance.
    get_t : callable
        Returns current time in seconds since node start.
    object_buffer : dict[ObjectType, list[Object]]
        Per-type ring buffer of recent observations.
    buffer_duration : float
        Maximum age (seconds) of buffered observations.
    clustering_settings : dict
        HDBSCAN min_samples, eps, and max_items per ObjectType.
    stability_thresholds : dict
        Maximum positional and angular displacement per ObjectType.
    presence_duration_threshold : float
        Maximum time gap (seconds) before an object is considered absent.
    absence_duration_tolerance : float
        Maximum gap between consecutive observations within a cluster.
    smoothing_window_duration : float
        Duration (seconds) of the moving-average smoothing window.
    """

    def __init__(self, logger, get_t_func):
        """Initialize the object processor.

        Arguments
        ---------
        logger
            ROS logger for logging messages.
        get_t_func : callable
            Function that returns current time in seconds.
        """
        self.logger = logger
        self.get_t = get_t_func

        self.object_buffer: dict[ObjectType, list[Object]] = {}

        self.buffer_duration = BUFFER_DURATION
        self.clustering_settings = CLUSTERING_SETTINGS
        self.stability_thresholds = STABILITY_THRESHOLDS
        self.presence_duration_threshold = PRESENCE_DURATION_THRESHOLD
        self.absence_duration_tolerance = ABSENCE_DURATION_TOLERANCE
        self.smoothing_window_duration = SMOOTHING_WINDOW_DURATION

    # ── Public API ────────────────────────────────────────────────────

    def process_contours(self, objects: list[Object]) -> list[Object]:
        """Run the full processing pipeline: buffer, cluster, detect.

        Arguments
        ---------
        objects : list[Object]
            New objects from current frame.

        Returns
        -------
        list[Object]
            Stable detected objects.
        """
        self.update_buffers(objects)

        for obj_type in ObjectType:
            self.cluster_buffered_objects(obj_type)

        return self.detect()

    def render_detections(self, detected_objects: list[Object]) -> np.ndarray:
        """Render a top-down bird's-eye view of detected objects.

        Draws filled polygons or rotated squares for each detection
        on a 1920x1080 canvas mapped to the calibrated workspace bounds.

        Arguments
        ---------
        detected_objects : list[Object]
            Objects to visualize.

        Returns
        -------
        np.ndarray
            1080x1920x3 RGB image.
        """
        IMG_H, IMG_W = 1080, 1920

        canvas = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        cv2.putText(
            canvas, f'BRAIN/DETECTIONS ({IMG_W}x{IMG_H})', (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA,
        )

        for obj in detected_objects:
            color = _OBJ_COLORS.get(obj.obj_type, (255, 255, 255))

            cv2.fillPoly(canvas, [np.array(obj.corner_uvs, dtype=np.int32)], color)

            cp = obj.center_uv[0], obj.center_uv[1]
            cv2.circle(canvas, cp, 3, (255, 0, 0), -1)

            # cv2.putText(canvas, label, (cp[0] + 5, cp[1] - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1)
            cv2.putText(
                canvas,
                f"({obj.center_xyz[0]:.2f}, {obj.center_xyz[1]:.2f})"
                f", {np.degrees(obj.angle):.1f} deg",
                (cp[0] - 60, cp[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3,
                cv2.LINE_AA,
            )

        return canvas

    # ── Buffer management ─────────────────────────────────────────────

    def update_buffers(self, objects: list[Object]):
        """Append new objects to the buffer and prune expired entries.

        Arguments
        ---------
        objects : list[Object]
            Newly observed objects to buffer.
        """
        for obj in objects:
            if obj.obj_type not in self.object_buffer:
                self.object_buffer[obj.obj_type] = []
            self.object_buffer[obj.obj_type].append(obj)

        t = self.get_t()
        for obj_type in self.object_buffer:
            while len(self.object_buffer[obj_type]) > 0 and \
              t - self.object_buffer[obj_type][0].t > self.buffer_duration:
                self.object_buffer[obj_type].pop(0)

    def cluster_buffered_objects(self, obj_type: ObjectType):
        """Cluster buffered observations of one type using HDBSCAN.

        Sets the obj_label field on each buffered Object.

        Arguments
        ---------
        obj_type : ObjectType
            The type of object to cluster.
        """
        if obj_type not in self.clustering_settings:
            self.logger.warning(
                f"Clustering settings for object type "
                f"'{obj_type}' not found."
            )
            return

        obj_centers = []
        for obj in self.object_buffer.get(obj_type, []):
            obj_centers.append(np.array(obj.center_xyz))

        if len(obj_centers) == 0:
            return

        settings = self.clustering_settings[obj_type]
        # labels = HDBSCAN(
        #     min_cluster_size=settings['min_samples'],
        #     min_samples=settings['min_samples'],
        #     cluster_selection_epsilon=settings['eps'],
        # ).fit(np.array(obj_centers)).labels_
        labels = DBSCAN(
            # min_cluster_size=settings['min_samples'],
            min_samples=settings['min_samples'],
            eps=settings['eps'],
        ).fit(np.array(obj_centers)).labels_
        labels = [int(label) if label != -1 else None for label in labels]

        for i, obj in enumerate(self.object_buffer.get(obj_type, [])):
            obj.obj_label = labels[i]

    # ── Stability detection ───────────────────────────────────────────

    def detect(self) -> list[Object]:
        """Detect stable objects from the buffer.

        For each cluster, computes smoothed position and angle
        trajectories via a moving average, measures cumulative
        displacement, checks temporal presence, and applies
        per-type stability thresholds.  Returns the top
        max_items most stable candidates per type.

        Returns
        -------
        list[Object]
            Stable detected objects sorted by displacement (ascending).
        """
        t = self.get_t()

        detected_objects: list[Object] = []

        for obj_type, b_objs in self.object_buffer.items():
            label_to_buffered = defaultdict(list)
            for b_obj in b_objs:
                if b_obj.obj_label is not None:
                    label_to_buffered[b_obj.obj_label].append(b_obj)

            type_candidates = []

            for label, l_objs in label_to_buffered.items():
                l_objs.sort(key=lambda x: x.t)

                smoothed_pos = []
                smoothed_angle = []

                for i, current_obj in enumerate(l_objs):
                    window_objs = [
                        o for o in l_objs
                        if current_obj.t - self.smoothing_window_duration
                        <= o.t <= current_obj.t
                    ]

                    avg_xyz = np.mean(
                        [o.center_xyz for o in window_objs], axis=0,
                    )
                    smoothed_pos.append(avg_xyz)

                    def average_square_angles_rad(
                        angles_rad: np.ndarray,
                        threshold=0.5,
                    ):
                        """Average angles with pi/2 symmetry.

                        Maps angles to 4x frequency so that 0 and
                        pi/2 are equivalent, averages the unit
                        vectors, and maps back.
                        """
                        angles = np.asarray(angles_rad, dtype=float)
                        x = np.mean(np.cos(angles * 4))
                        y = np.mean(np.sin(angles * 4))
                        # r = np.hypot(x, y)
                        # if r < threshold:
                        #     return np.nan
                        mean_mapped_angle = np.arctan2(y, x)
                        mean_angle = mean_mapped_angle / 4.0
                        return mean_angle % (np.pi / 2.0)

                    valid_angles = [
                        o.angle for o in window_objs if o.angle is not None
                    ]
                    # sum_angle = np.sum(valid_angles)
                    # avg_angle = sum_angle / len(valid_angles)
                    avg_angle = average_square_angles_rad(valid_angles)
                    smoothed_angle.append(avg_angle)

                pos_displacement = 0.0
                angle_displacement = 0.0

                for i in range(1, len(l_objs)):
                    pos_displacement += np.linalg.norm(
                        smoothed_pos[i] - smoothed_pos[i - 1]
                    )

                    if (smoothed_angle[i] is not None
                            and smoothed_angle[i - 1] is not None):
                        angle_displacement += np.linalg.norm(
                            smoothed_angle[i] - smoothed_angle[i - 1]
                        )

                is_present = True

                if t - l_objs[-1].t >= self.presence_duration_threshold:
                    is_present = False

                if is_present:
                    for i in range(1, len(l_objs)):
                        dt = l_objs[i].t - l_objs[i - 1].t
                        if dt > self.absence_duration_tolerance:
                            is_present = False
                            break

                pos_check = (
                    pos_displacement
                    < self.stability_thresholds[obj_type]['position']
                )
                angle_check = (
                    angle_displacement
                    < self.stability_thresholds[obj_type]['angle']
                )

                if pos_check and angle_check and is_present:
                    # NOTE: intersect method disabled -- seems to be
                    # worse though high-level idea probably has merit
                    # intersect_result = self._intersect_corners(l_objs)
                    intersect_result = None
                    if intersect_result is not None:
                        center_xyz, angle, corner_xys = intersect_result
                    else:
                        center_xyz = tuple(smoothed_pos[-1])
                        angle = smoothed_angle[-1]
                        corner_xys = None

                    detected_obj = Object(
                        t=l_objs[-1].t,
                        frame_idx=l_objs[-1].frame_idx,
                        obj_type=obj_type,
                        center_uv=l_objs[-1].center_uv,
                        center_xyz=center_xyz,
                        angle=l_objs[-1].angle,
                        corner_xys=l_objs[-1].corner_xys,
                        corner_uvs=l_objs[-1].corner_uvs,
                        face_conns_xyzs=l_objs[-1].face_conns_xyzs, # TODO smooth this and check if above threshold
                    )

                    type_candidates.append({
                        'obj': detected_obj,
                        'displacement': pos_displacement
                    })

            type_candidates.sort(key=lambda x: x['displacement'])

            max_items = self.clustering_settings[obj_type]['max_items']
            for candidate in type_candidates[:max_items]:
                detected_objects.append(candidate['obj'])

        return detected_objects

    # ── Private helpers ───────────────────────────────────────────────

    def _intersect_corners(self, objects: list['Object']):
        """Intersect rectangle corners to tighten bounding boxes.

        Iteratively intersects the convex hulls of corner polygons
        from multiple observations of the same cluster, then fits
        a minimum-area rectangle to the intersection to produce a
        refined center, quaternion, and corner set.

        Arguments
        ---------
        objects : list[Object]
            Clustered observations (all with corner_xys).

        Returns
        -------
        tuple or None
            (center_xyz, quaternion, corner_xys) on success,
            or None if fewer than 2 valid polygons or the
            intersection degenerates.
        """
        polys = []
        for obj in objects:
            if obj.corner_xys is not None and len(obj.corner_xys) >= 3:
                poly = np.asarray(obj.corner_xys, dtype=np.float32)
                hull = cv2.convexHull(poly)
                polys.append(hull.reshape(-1, 2))

        if len(polys) < 2:
            return None

        intersection = polys[0].reshape(-1, 1, 2)
        for poly in polys[1:]:
            ret, result = cv2.intersectConvexConvex(
                intersection,
                poly.reshape(-1, 1, 2)
            )
            if ret < 1e-10 or result is None or len(result) < 3:
                return None
            intersection = result

        pts = intersection.reshape(-1, 2).astype(np.float32)

        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), _ = rect
        corners = cv2.boxPoints(rect)

        corner_xys = [(float(c[0]), float(c[1])) for c in corners]

        bottom_corner_index = int(
            np.argmin([c[1] for c in corner_xys])
        )
        bottom_corner = corner_xys[bottom_corner_index]

        dists = []
        for i, c in enumerate(corner_xys):
            if i != bottom_corner_index:
                dist = np.sqrt(
                    (c[0] - bottom_corner[0])**2
                    + (c[1] - bottom_corner[1])**2
                )
                dists.append(dist)
            else:
                dists.append(0.0)

        # 2nd largest distance = long edge partner
        long_edge_index = int(np.argsort(dists)[2])

        delta_x = corner_xys[long_edge_index][0] - bottom_corner[0]
        delta_y = corner_xys[long_edge_index][1] - bottom_corner[1]
        angle = np.arctan2(delta_y, delta_x)
        if angle > np.pi / 2:
            angle -= np.pi

        half_angle = angle / 2.0
        quaternion = (
            0.0, 0.0,
            float(np.sin(half_angle)), float(np.cos(half_angle)),
        )

        center_xyz = (float(cx), float(cy), 0.0)
        return center_xyz, quaternion, corner_xys
