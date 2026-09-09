# Brain → Detector: trigger scan/detection with request type
std_msgs/Header header
string request_id                    # unique ID for tracking
uint8 request_type
uint8 SIMPLY_ACCUMULATE=0
uint8 DETECT_FROM_ACCUMULATED=1
uint8 DETECT_FROM_LATEST=2
uint8 IN_Z_CALIBRATION=4
uint8 IN_XY_CALIBRATION=5
uint8 DETECT_GRID_COORDS=6
