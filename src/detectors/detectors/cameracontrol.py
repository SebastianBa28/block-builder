'''cameracontrol.py

   Create trackbars to control the camera parameters.

   This is not a ROS node, but directly accesses the camera.
'''

import cv2
import numpy as np
import subprocess
import time


#
#   Define the Interface
#
PROP_AUTOFOCUS  = "focus_automatic_continuous"
PROP_FOCUS      = "focus_absolute"
PROP_AUTOEXP    = "auto_exposure"
PROP_EXPOSURE   = "exposure_time_absolute"
PROP_GAIN       = "gain"
PROP_AUTOWB     = "white_balance_automatic"
PROP_WB         = "white_balance_temperature"
PROP_BRIGHTNESS = "brightness"
PROP_CONTRAST   = "contrast"
PROP_SATURATION = "saturation"
PROP_SHARPNESS  = "sharpness"

TEXT_AUTOFOCUS  = "Autofocus OFF (0) or ON (1)                  "
TEXT_FOCUS      = "    Focus                                            "
TEXT_AUTOEXP    = "Autoexposure OFF (1) or ON (3)            "
TEXT_EXPOSURE   = "    Exposure                                  "
TEXT_GAIN       = "    Gain                                               "
TEXT_AUTOWB     = "Autowhitebalance  OFF (0) or ON (1):  "
TEXT_WB         = "    Whitebalance                          "
TEXT_BRIGHTNESS = "Brightness                                        "
TEXT_CONTRAST   = "Contrast                                           "
TEXT_SATURATION = "Saturation                                        "
TEXT_SHARPNESS  = "Sharpness                                        "


#
#  Camera Control Interface Object
#
class CameraControlInterface():
    def __init__(self, device):
        self.device = device

    def get(self, param):
        # Execute the command.
        cmd = ['v4l2-ctl', '--device', self.device, '-C', param]
        try:
            l = subprocess.check_output(cmd).split()
        except:
            raise Exception(f"Unable to get {param} on {self.device}")

        # Confirm the appropriate return value.
        if (len(l) < 2) or (l[0].decode("utf-8") != param+':'):
            raise Exception(f"Failed to get {param} on {self.device}")

        # Extract the value, report, and return.
        value = int(l[1])
        if lowlevelverbose:
            print(f"Parameter {param} read as {value}")
        return value

    def set(self, param, value):
        # Execute the command.
        cmd = ['v4l2-ctl', '--device', self.device, '-c', param+'='+str(value)]
        try:
            out = subprocess.check_output(cmd)
        except:
            raise Exception(f"Unable to set {param} on {self.device}")

        # Confirm the appropriate return value.        
        if (out != b''):
            raise Exception(f"Failed to set {param} on {self.device}")

        # Report.
        if lowlevelverbose:
            print(f"Parameter {param} set to {value}")


#
#   Slider (TrackBar) Objects
#
class ValueBar():
    def __init__(self, cam, win, name, prop, low, high, step=1):
        # Store the parameters.
        self.cam  = cam
        self.win  = win
        self.name = name
        self.prop = prop
        self.low  = int(low)
        self.high = int(high)
        self.step = int(step)

        # Clear the trigger.
        self.triggers = []

        # Create the trackbar (initialized at the current value).
        cv2.createTrackbar(name, win, self.get(), high, self.set)

    def addtrigger(self, value, callback):
        # Enable an automatic callback on a particular value.
        self.triggers.append((value, callback))

    def reset(self):
        # Reset (update) the trackbar.
        cv2.setTrackbarPos(self.name, self.win, self.get())

    def get(self):
        # Read the camera property
        self.value = int(self.cam.get(self.prop))
        if cameraverbose:
            print("Reading %s as %d" % (self.name, self.value))
        return self.value

    def set(self, value):
        if value != self.value:
            # Enforce the min/max values and step size:
            value = max(self.low, min(self.high, value))
            value = self.low + self.step * int((value - self.low)/self.step)

            # Update the camera property
            if cameraverbose:
                print("Setting %s to %d" % (self.name, value))
            self.cam.set(self.prop, value)

            # Reset the trackbar (in case the camera changes the values).
            self.reset()

            # Also call the trigger callback if set.
            for (trigval, trigcb) in self.triggers:
                if value == trigval:
                    trigcb()


class OnOffBar():
    def __init__(self, cam, win, name, prop, off, on):
        # Create the bar.
        self.bar = ValueBar(cam, win, name, prop,
                               min(off, on), max(off, on), abs(on-off))

    def addtrigger(self, value, callback):
        self.bar.addtrigger(value, callback)

    def reset(self):
        self.bar.reset()


#
#   Camera Setup/Control Window Object
#
class CameraControl():
    def __init__(self, device='/dev/video0', verbose=False):
        # Save the verbosity.
        global lowlevelverbose, cameraverbose
        lowlevelverbose = False
        cameraverbose   = verbose

        # Open the low-level control interface.
        cam = CameraControlInterface(device)

        # Choose the controls window name.
        name = f"Camera Controls for {device}"

        # Create a controls window for the bars.
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

        # Set the header
        header = np.full((50, 800, 3), (235, 236, 237), np.uint8)
        cv2.putText(header, "Toggle AUTO to OFF to see the auto-tuned value",
                    (25, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 1)
        cv2.putText(header, "Click any name to enter, or slide the slider",
                    (25, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 1)

        # Show.
        cv2.imshow(name, header)

        # Add the bars.
        bar = OnOffBar(cam, name, TEXT_AUTOFOCUS,  PROP_AUTOFOCUS,  0,    1)
        val = ValueBar(cam, name, TEXT_FOCUS,      PROP_FOCUS,      0,  255, 1)
        bar.addtrigger(0, val.reset)

        bar = OnOffBar(cam, name, TEXT_AUTOEXP,    PROP_AUTOEXP,    1,    3)
        v1  = ValueBar(cam, name, TEXT_EXPOSURE,   PROP_EXPOSURE,   3, 2047, 1)
        v2  = ValueBar(cam, name, TEXT_GAIN,       PROP_GAIN,       0,  255, 1)
        bar.addtrigger(1, v1.reset)
        bar.addtrigger(1, v2.reset)

        bar = OnOffBar(cam, name, TEXT_AUTOWB,     PROP_AUTOWB,     0,    1)
        val = ValueBar(cam, name, TEXT_WB,         PROP_WB,      2000, 6500, 1)
        bar.addtrigger(0, val.reset)

        val = ValueBar(cam, name, TEXT_BRIGHTNESS, PROP_BRIGHTNESS, 0,  255, 1)
        val = ValueBar(cam, name, TEXT_CONTRAST,   PROP_CONTRAST,   0,  255, 1)
        val = ValueBar(cam, name, TEXT_SATURATION, PROP_SATURATION, 0,  255, 1)

        val = ValueBar(cam, name, TEXT_SHARPNESS,  PROP_SHARPNESS,  0,  255, 1)

    def update(self):
        # Call waitKey(1) to force the window to update.
        cv2.waitKey(1)


#
#   Main Code
#
def main(args=None):
    # Create the Camera Control Window.
    control = CameraControl('/dev/video0', True)

    # Keep updating
    while True:
        # Update and sleep 20ms
        control.update()
        time.sleep(0.02)

if __name__ == "__main__":
    main()
