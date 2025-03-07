"""Convert images to OpenCV blobs"""
import logging
from typing import Any, Optional, Sequence, Union

from intrusiondet.core.image import get_frame
from intrusiondet.core.types import IntOrFloat, NDArray, NDArrayInt, UInt8

try:
    import cv2.cv2 as cv
except ImportError:
    import cv2 as cv

Blob = NDArray
"""A NumPy array with shape (n_frames, n_channels, width, height)"""


def check_blob(out_size: Union[int, tuple[int, int]]) -> tuple[Any]:
    """Assure that the output blob has correct properties

    :param out_size: The output size of the frame(s) in the blob
    :return: A tuple of the properties. Signature (out_size, )
    :raises AttributeError: If the size specification is not in type integer
    :raises ValueError: If the size is not square (an integer) or 2-tuple
    """
    error_msg = (
        "The image size must be one or two identical, positive definite"
        " integers, not %s",
        out_size,
    )
    if not (
        (hasattr(out_size, "__iter__") and len(out_size) == 2)
        or (isinstance(out_size, int))
    ):
        logging.getLogger().exception(*error_msg)
        raise AttributeError
    if isinstance(out_size, int):
        if out_size <= 0:
            logging.getLogger().exception(*error_msg)
            raise ValueError
        out_size = (out_size, out_size)
    else:
        if len(set(out_size)) != 1:
            logging.getLogger().exception(*error_msg)
            raise ValueError
    # fmt: off
    return out_size,  # pylint: disable=trailing-comma-tuple
    # fmt: on


def frame2blob(
    frame: NDArrayInt,
    out_size: Union[int, tuple[int, int]],
    scale_factor: IntOrFloat = 1.0 / UInt8.MAX,
    swap_rb: Optional[bool] = True,
    crop: Optional[bool] = False,
    data_type: Optional[int] = None,
) -> Blob:
    """Convert a frame to a Blob

    :param frame: Input frame
    :param out_size: The output size of the blob
    :param scale_factor: A rescaling factor for the image pixels (default = 1/2**8)
    :param swap_rb: Flag which indicates that swap first and last channels
    :param crop: Crop the input
    :param data_type: Output data type/depth
    :return: OpenCV blob object for a single frame
    """
    (out_size,) = check_blob(out_size)
    blob = cv.dnn.blobFromImage(  # pylint: disable=no-member
        image=frame,
        scalefactor=scale_factor,
        size=out_size,
        swapRB=swap_rb,
        crop=crop,
        ddepth=data_type,
    )
    return blob


def path2blob(
    img_path: Union[str, int],
    out_size: Union[int, tuple[int, int]],
    scale_factor: IntOrFloat = 1.0 / UInt8.MAX,
    swap_rb: Optional[bool] = True,
    crop: Optional[bool] = False,
    data_type: Optional[int] = None,
) -> Blob:
    """Given an image path and size, convert it to an OpenCV blob

    :param img_path: The path to the image
    :param out_size: Scalar or ArrayLike pixel size for the image
    :param scale_factor: A rescaling factor for the image pixels (default = 1/2**8)
    :param swap_rb: Swap the colors from RGB to black and white (default=True)
    :param crop: If to crop the image to the specified size input. This does nothing if
        the size is unchanged.
    :param data_type: Output data type/depth
    :return: OpenCV blob object for a single image
    """
    frame: NDArrayInt = get_frame(img_path)
    blob = frame2blob(
        frame=frame,
        out_size=out_size,
        scale_factor=scale_factor,
        swap_rb=swap_rb,
        crop=crop,
        data_type=data_type,
    )
    return blob


def frames2blob(
    frames: Sequence[NDArrayInt],
    out_size: Union[int, tuple[int, int]],
    scale_factor: IntOrFloat = 1.0 / UInt8.MAX,
    swap_rb: Optional[bool] = True,
    crop: Optional[bool] = False,
    data_type: Optional[int] = None,
) -> Blob:
    """Convert multiple frames to a single Blob

    :param frames: Input frame
    :param out_size: The output size of the blob
    :param scale_factor: A rescaling factor for the image pixels (default = 1/2**8)
    :param swap_rb: Flag which indicates that swap first and last channels
    :param crop: Crop the input
    :param data_type: Output data type/depth
    :return: OpenCV blob object for multiple frames
    """
    (out_size,) = check_blob(out_size)
    blob = cv.dnn.blobFromImages(  # pylint: disable=no-member
        images=frames,
        scalefactor=scale_factor,
        size=out_size,
        swapRB=swap_rb,
        crop=crop,
        ddepth=data_type,
    )
    return blob
