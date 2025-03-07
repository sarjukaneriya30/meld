"""Basic methods to extract and build image frames"""
from __future__ import annotations

from typing import Final, Union

import numpy as np

from intrusiondet.core.types import Frame, FrameOrNone, IntTuple

try:
    import cv2.cv2 as cv
except ImportError:
    import cv2 as cv


IMAGE_FORMATS: Final[tuple[str, ...]] = ("JPEG", "JPG", "PNG")
"""Image file formats"""


def get_frame(img_path: Union[str, int]) -> FrameOrNone:
    """Get a frame from an input image or device ID using an integer (untested)

    :param img_path: The path to the file
    :return: An image frame
    """
    # has_frame: bool
    frame: FrameOrNone = cv.imread(img_path)
    # # The argument of cv.VideoCapture might need to change to 0 if a camera feed
    # cap: cv.VideoCapture = cv.VideoCapture(cv.samples.findFileOrKeep(img_path))
    # has_frame, frame = cap.read()
    return frame


def build_frame(shape: tuple, dtype: np.dtype = np.uint8) -> Frame:
    """Build a frame with the specified shape and data type

    :param shape: Shape tuple
    :param dtype: Data type
    :return: New data frame filled with zeros (black)
    """
    frame = np.zeros(shape, dtype=dtype)
    return frame


def apply_color(frame: Frame, color_bgr: IntTuple) -> Frame:
    """Apply a solid color to a frame. Overrides any previous values

    :param frame: Input frame
    :param color_bgr: Color tuple in (blue, green, red) order
    """
    frame[:, :] = color_bgr
    return frame
