"""Methods to aid with OpenCV frontend highgui"""
from typing import Final, Optional

import numpy as np

from intrusiondet.core.colors import KPMG_BLUE, WHITE
from intrusiondet.core.types import ClassNameID, Frame, FrameIndex, IntTuple, UInt8
from intrusiondet.model.detectedobject import DetectedObject

try:
    import cv2.cv2 as cv
except ImportError:
    import cv2 as cv


FONTS: Final[tuple[int, ...]] = (
    cv.FONT_HERSHEY_SIMPLEX,
    cv.FONT_HERSHEY_DUPLEX,
    cv.FONT_HERSHEY_PLAIN,
    cv.FONT_HERSHEY_COMPLEX,
    cv.FONT_HERSHEY_COMPLEX_SMALL,
    cv.FONT_HERSHEY_SCRIPT_COMPLEX,
    cv.FONT_HERSHEY_TRIPLEX,
    cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv.FONT_ITALIC,
)
"""Available OpenCV fonts"""

WINDOW_FLAGS: Final[tuple[int, ...]] = (
    cv.WINDOW_FULLSCREEN,
    cv.WINDOW_NORMAL,
    cv.WINDOW_OPENGL,
    cv.WINDOW_GUI_NORMAL,
    cv.WINDOW_AUTOSIZE,
    cv.WINDOW_FREERATIO,
    cv.WINDOW_GUI_EXPANDED,
    cv.WINDOW_KEEPRATIO,
)
"""Available OpenCV window options"""


def msg_valid_window_options() -> str:
    """Return the supported OpenCV window options

    :return: Valid OpenCV window options as a human-readable string
    """
    # fmt: off
    return (
        "Only the following window options and combinations are allowed, "
        f" {cv.WINDOW_FULLSCREEN  }: WINDOW_FULLSCREEN,"
        f" {cv.WINDOW_NORMAL      }: WINDOW_NORMAL,"
        f" {cv.WINDOW_OPENGL      }: WINDOW_OPENGL,"
        f" {cv.WINDOW_GUI_NORMAL  }: WINDOW_GUI_NORMAL,"
        f" {cv.WINDOW_AUTOSIZE    }: WINDOW_AUTOSIZE,"
        f" {cv.WINDOW_FREERATIO   }: WINDOW_FREERATIO,"
        f" {cv.WINDOW_GUI_EXPANDED}: WINDOW_GUI_EXPANDED,"
        f" {cv.WINDOW_KEEPRATIO   }: WINDOW_KEEPRATIO"
    )
    # fmt: on


def msg_valid_fonts() -> str:
    """Return the supported OpenCV window options

    :return: Valid OpenCV fonts as a human-readable string
    """
    # fmt: off
    return (
        "Only the following OpenCV fonts are allowed"
        f" {cv.FONT_HERSHEY_SIMPLEX       }: FONT_HERSHEY_SIMPLEX,"
        f" {cv.FONT_HERSHEY_DUPLEX        }: FONT_HERSHEY_DUPLEX,"
        f" {cv.FONT_HERSHEY_PLAIN         }: FONT_HERSHEY_PLAIN,"
        f" {cv.FONT_HERSHEY_COMPLEX       }: FONT_HERSHEY_COMPLEX,"
        f" {cv.FONT_HERSHEY_COMPLEX_SMALL }: FONT_HERSHEY_COMPLEX_SMALL,"
        f" {cv.FONT_HERSHEY_SCRIPT_COMPLEX}: FONT_HERSHEY_SCRIPT_COMPLEX,"
        f" {cv.FONT_HERSHEY_TRIPLEX       }: FONT_HERSHEY_TRIPLEX,"
        f" {cv.FONT_HERSHEY_SCRIPT_SIMPLEX}: FONT_HERSHEY_SCRIPT_SIMPLEX,"
        f" {cv.FONT_ITALIC                }: FONT_ITALIC"
    )
    # fmt: on


def draw_text_box(
    frame: Frame,
    label: str,
    bbox_top: FrameIndex,
    bbox_left: FrameIndex,
    font: Optional[int] = None,
    font_color: Optional[IntTuple] = None,
    font_scale: Optional[float] = None,
    font_thickness: Optional[int] = None,
) -> None:
    """Create a text box with a background in the frame

    :param frame: OpenCV window frame
    :param label: Text string to display
    :param bbox_left: Left bounding box edge
    :param bbox_top: Top bounding box edge
    :param font: Displayed font
    :param font_color: BGR tuple of font color
    :param font_scale: Scale factor is multiplied by the font-specific base size
    :param font_thickness: Thickness of the lines used to draw a text
    :return: None
    :raises ValueError: If the input font number is not in the enumerated font option
    """
    if font is None:
        font = cv.FONT_HERSHEY_SIMPLEX
    elif font not in FONTS:
        raise ValueError(msg_valid_fonts())
    else:
        pass
    if font_color is None:
        font_color = WHITE.as_bgr()
    text_foreground_color = tuple(UInt8.MAX - num for num in font_color)
    if font_scale is None:
        font_scale = 1.0
    if font_thickness is None:
        font_thickness = 2
    label_size, base_line = cv.getTextSize(label, font, font_scale, font_thickness)
    bbox_top = max(bbox_top, label_size[1])
    cv.rectangle(
        img=frame,
        pt1=(bbox_left, bbox_top - label_size[1]),
        pt2=(bbox_left + label_size[0], bbox_top + base_line),
        color=text_foreground_color,
        thickness=cv.FILLED,
    )
    cv.putText(
        img=frame,
        text=label,
        org=(bbox_left, bbox_top),
        fontFace=font,
        fontScale=font_scale,
        color=font_color,
        thickness=font_thickness,
    )


def draw_prediction_from_detected_object(
    frame: Frame, det_obj: DetectedObject, **kwargs
) -> None:
    """Draw bounding boxes and confidence value for the detected object

    :param frame: OpenCV window frame
    :param det_obj: Detected object
    :param kwargs: See `intrusiondet.core.frontend.draw_prediction`
    :return: None
    """
    bbox_left, bbox_top, bbox_width, bbox_height = det_obj.box.astuple()
    return draw_prediction(
        frame,
        det_obj.class_name,
        det_obj.conf,
        bbox_left,
        bbox_top,
        bbox_width,
        bbox_height,
        **kwargs,
    )


def draw_prediction(
    frame: np.ndarray,
    class_name: ClassNameID,
    conf: float,
    bbox_left: FrameIndex,
    bbox_top: FrameIndex,
    bbox_width: FrameIndex,
    bbox_height: FrameIndex,
    font: Optional[int] = None,
    window_name: Optional[str] = None,
    window_flag: Optional[int] = None,
    box_color: Optional[IntTuple] = None,
    box_thickness: Optional[int] = None,
    font_color: Optional[IntTuple] = None,
    font_scale: Optional[float] = None,
    font_thickness: Optional[int] = None,
    class_name_maxlength: Optional[int] = None,
) -> None:
    """Draw bounding boxes and confidence value for the detection

    :param frame: OpenCV window frame
    :param class_name: The name of the object
    :param conf: Confidence percentage of detected class
    :param bbox_left: Left bounding box edge
    :param bbox_top: Top bounding box edge
    :param bbox_width: Width of bounding box
    :param bbox_height: Height of bounding box
    :param font: Displayed font
    :param window_name:  Name of the window frame
    :param window_flag: Window flags
    :param box_color: BGR tuple box color
    :param box_thickness: Thickness (integer) of the bounding box. Negative values fill
     the box
    :param font_color: BGR tuple font color
    :param font_scale: Scale factor is multiplied by the font-specific base size
    :param font_thickness: Thickness (integer) of the lines used to draw a text
    :param class_name_maxlength: Maximum length of name string
    :return: None
    :raises ValueError: If the input window option number is not in the enumerated
     window options
    """
    if window_name is not None:
        if window_flag is None:
            window_flag = cv.WINDOW_NORMAL
        elif window_flag not in WINDOW_FLAGS:
            raise ValueError(msg_valid_window_options())
        else:
            pass
        cv.namedWindow(window_name, window_flag)
    if box_thickness is None:
        box_thickness = 2
    if font_scale is None:
        font_scale = 1.0
    if font_thickness is None:
        font_thickness = 2

    # Draw a bounding box.
    if box_color is None:
        box_color = KPMG_BLUE.as_bgr()
    right = bbox_left + bbox_width
    bottom = bbox_top + bbox_height
    cv.rectangle(
        frame, (bbox_left, bbox_top), (right, bottom), box_color, box_thickness
    )
    # conf_percentage = int(np.floor(conf * 100))
    # # Place a text label with an inverted foreground
    # if class_name_maxlength is not None:
    #     class_name = class_name[0:class_name_maxlength]
    # label = f"{class_name}: {conf_percentage}%"
    # draw_text_box(
    #     frame, label, bbox_top, bbox_left, font, font_color, font_scale, font_thickness
    # )
