# from Agade09
import os

import cv2
import numba
import numpy as np

from . import misc

five_digits_center = np.array([14, 33, 47, 67, 81]) + 1

digits_radius = 6  # int(W*6/640)
time_screen_width_percentage = 0.075
time_screen_height_percentage = 0.025
time_screen_height = 0.975

h_min_time = round((time_screen_height - time_screen_height_percentage) * misc.H) - 2
h_max_time = round((time_screen_height + time_screen_height_percentage) * misc.H) - 2
w_min_time = round((0.5 - time_screen_width_percentage) * misc.W)
w_max_time = round((0.5 + time_screen_width_percentage) * misc.W)


class DigitsLibrary:
    def __init__(self, digits_filename):
        self.digits = np.load(digits_filename, allow_pickle=True)
        self.digit_set = set({tuple(digit.flatten()) for digit, _ in self.digits})
        self.digits_stack = np.stack(self.digits[:, 0])
        self.digits_value_stack = np.stack(self.digits[:, 1])


@numba.njit
def parse_time2(img, digits, digits_value):
    # import pdb; pdb.set_trace()
    time = 0
    for i in range(5):
        digit = get_digit(img, five_digits_center[i])
        diffs = np.array([np.sum(np.abs(d.astype(np.float32) - digit.astype(np.float32))) for d in digits])
        diffs_argmin = np.argmin(diffs)
        best_match = digits_value[diffs_argmin]
        time += best_match * (60000, 10000, 1000, 100, 10)[i]
    return round(time, 2)


def parse_time(img, library):
    # Shape should be (480, 640, 4) in BGRA mode
    a = get_time_screen(img)
    return parse_time2(get_time_screen(img), library.digits_stack, library.digits_value_stack)


def binarise_screen_numbers(img):
    return cv2.bitwise_not(cv2.inRange(img, (255, 251, 255, 0), (255, 251, 255, 255)))  # Writing_Color = (255,251,255)


@numba.njit
def get_digit(img, center):
    return img[:, center - digits_radius : center + digits_radius]


def get_time_screen(img):
    time_screen = img[h_min_time:h_max_time, w_min_time:w_max_time, :]
    return binarise_screen_numbers(time_screen)
