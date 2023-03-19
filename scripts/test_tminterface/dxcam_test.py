# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:01:35 2023

@author: chopi
"""

import dxcam
import win32con
import win32gui

# sct.stop()

W = 640
H = 480  # Just as Terry Davis would have wanted it

trackmania_window = win32gui.FindWindow("TmForever", None)
# trackmania_window = win32gui.FindWindow('TrackMania Nations Forever (TMInterface 1.4.1)',None)


def get_window_position():
    trackmania_window = win32gui.FindWindow("TmForever", None)
    rect = win32gui.GetWindowRect(trackmania_window)
    left = rect[0] + margins["left"]
    top = rect[1] + margins["top"]
    right = rect[2] - margins["right"]
    bottom = rect[3] - margins["bottom"]

    # left = rect[0] + round(
    #     ((rect[2] - rect[0]) - W) / 2
    # )  # Could there be a 1 pixel error with these roundings?
    # top = rect[1] + round(((rect[3] - rect[1]) - H) * 0.76)
    # right = left + W
    # bottom = top + H
    return (left, top, right, bottom)


# Windows 10 has thin invisible borders on left, right, and bottom, it is used to grip the mouse for resizing.
# The borders might look like this: 7,0,7,7 (left, top, right, bottom)
margins = {"left": 7, "top": 34, "right": 7, "bottom": 7}

# To get 640x460 à la louche
# rect width : 654
# rect height : 487


# # Add this import
# import win32com.client

# # Add this to __ini__
# shell = win32com.client.Dispatch("WScript.Shell")

# # And SetAsForegroundWindow becomes
# shell.SendKeys('%')
# win32gui.SetForegroundWindow(trackmania_window)

win32gui.SetWindowPos(
    trackmania_window,
    win32con.HWND_TOPMOST,
    2560 - 740,
    100,
    640 + margins["left"] + margins["right"],
    480 + margins["top"] + margins["bottom"],
    0,
)


sct = dxcam.create(output_color="RGB", region=get_window_position(), max_buffer_len=1)


def get_screen(sct):
    return sct.get_latest_frame()


sct.start(target_fps=200, video_mode=False)


from matplotlib import pyplot as plt

plt.imshow(get_screen(sct), interpolation="nearest")
plt.show()

# import cv2
# cv2.imshow("zou", get_screen(sct))
# cv2.waitKey(0)

aaaaaaaaaaa

sct.stop()
