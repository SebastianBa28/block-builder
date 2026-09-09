# Raw contour info sent from detector each frame

float64 timestamp
int32 frame_idx

# Shape type constants
int32 CIRCLE=0
int32 RECTANGLE=1
int32 SQUARE=2
int32 shape_type

# Common fields - world coordinates
float32 center_x
float32 center_y
float32 center_z

int32 center_u
int32 center_v

# Circle-specific (valid if shape_type == CIRCLE)
int32 radius_px

# Rectangle-specific (valid if shape_type == RECTANGLE)
int32 length_px
int32 width_px
float32 angle  # radians
geometry_msgs/Quaternion quaternion

# Rectangle corner world coordinates (valid if shape_type == RECTANGLE)
# 4 corners, ordered from cv2.boxPoints
float32[] corner_world_xs
float32[] corner_world_ys
int32[] corner_world_us
int32[] corner_world_vs

# Face Connections
float32[] face_conns_xs
float32[] face_conns_ys
float32[] face_conns_zs

# Color constants
int32 ORANGE = 1
int32 YELLOW = 2
int32 BLUE = 3
int32 GREEN = 4
int32 RED = 5
int32 color