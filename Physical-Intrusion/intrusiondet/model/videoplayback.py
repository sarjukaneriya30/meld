"""Video playback methods"""
import logging
import os
import threading
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Final, Optional

import numpy as np

from intrusiondet.core.types import (
    FrameIndex,
    FrameOrNone,
    NamedObject,
    NDArrayInt,
    PathLike,
)

try:
    import cv2.cv2 as cv
except ImportError:
    import cv2 as cv


MOVIE_FORMATS: Final[tuple[str, ...]] = ("MP4", "AVI")
"""Movie file formats"""


class BasePlayback(NamedObject):
    """Boilerplate class to define locking mechanism for multiple processes"""

    def __init__(self, name: str):
        """Boilerplate class to define locking mechanism for multiple processes

        :param name: The name of the instance
        """
        super().__init__(name)
        self._logger = logging.getLogger()
        self._lock = threading.Lock()
        self._frame_counter: FrameIndex = 0
        self._open_status: bool = True

    def _acquire_lock(self):
        """Lock the class from changing properties until unlocked"""
        self._lock.acquire()

    def _release_lock(self):
        """Unlock the class from changing properties until unlocked"""
        self._lock.release()

    @abstractmethod
    def release(self):
        """Release the video from memory

        :return: None
        """
        raise NotImplementedError("This method is not implemented for BasePlayback")

    @cached_property
    def frames(self) -> FrameIndex:
        """Get the total number of frames in the playback sequence

        :return: Frame count
        """
        raise NotImplementedError("This method is not implemented for BasePlayback")

    @property
    def frame_counter(self) -> FrameIndex:
        """Return the current frame

        :return: frame counter
        """
        self._acquire_lock()
        ret = int(self._frame_counter)
        self._release_lock()
        return ret

    def is_opened(self) -> bool:
        """Query if the playback has more frames

        :return: True if open, False otherwise
        """
        self._acquire_lock()
        ret = self._open_status
        self._release_lock()
        return ret

    def set_frame_index(self, frame_index: FrameIndex) -> bool:
        """Advance the video to the given frame index

        :param frame_index: Desired frame index
        :return:  True if success, false otherwise
        """
        raise NotImplementedError

    def get_next_frame(self) -> FrameOrNone:
        """Get the next frame in the video

        :return: frame or None
        """
        raise NotImplementedError

    def get_frame_index(self, frame_index: FrameIndex) -> FrameOrNone:
        """Skip to given frame index of video

        :param frame_index: Frame index starting with 1
        :return: A frame, None otherwise
        """
        raise NotImplementedError


class DummyPlayback(BasePlayback):
    """Simulate video playback, but does nothing with video data"""

    def release(self):
        """

        :return:
        """
        self._acquire_lock()
        self._open_status = False
        self._release_lock()

    def set_frame_index(self, frame_index: int) -> bool:
        """Advance the video to the given frame index

        :param frame_index: Desired frame index
        :return:  True if success, false otherwise
        """
        raise NotImplementedError

    def get_next_frame(self) -> FrameOrNone:
        """Get the next frame in the video

        :return: frame or None
        """
        raise NotImplementedError

    def get_frame_index(self, frame_index: int) -> FrameOrNone:
        """Skip to given frame index of video

        :param frame_index: Frame index starting with 1
        :return: A frame, None otherwise
        """
        raise NotImplementedError

    @cached_property
    def frames(self) -> FrameIndex:
        return 0


class FramesPlayback(DummyPlayback):
    """Simulate video playback from chucked or streamed video"""

    def __init__(
        self,
        name: str,
        frames: NDArrayInt,
        fps: Optional[float] = None,
    ):
        """Simulate video playback from chucked or streamed video

        :param name: Name of the frames
        :param frames: Collection of frames
        """
        super().__init__(name)
        self._frames = frames
        self._channels: int = 0
        try:
            if self._frames:
                self._channels = self._frames[0].shape[-1]
        except (IndexError,):
            pass
        self._fps = fps

    def __getitem__(self, item) -> NDArrayInt:
        return self._frames[item]

    @cached_property
    def width(self) -> FrameIndex:
        """Get the width of the frame"""
        return self._frames.shape[2]

    @cached_property
    def height(self) -> FrameIndex:
        """Get the height of the frame"""
        return self._frames.shape[3]

    @cached_property
    def color_channels(self) -> int:
        """Get the number of color channels for the frame"""
        return self._channels

    @cached_property
    def frame_size(self) -> tuple[FrameIndex, FrameIndex]:
        """Get the frame width-by-height tuple"""
        return self.width, self.height

    @cached_property
    def frame_size_full(self) -> tuple[FrameIndex, FrameIndex, int]:
        """Get the frame width-by-height-by-channels tuple"""
        return self.width, self.height, self.color_channels

    @cached_property
    def frames(self) -> FrameIndex:
        """Get the video number of frames

        :return: total number of frames
        """
        return len(self._frames)

    @cached_property
    def fps(self) -> Optional[float]:
        """Get the frames-per-seconds. Maybe None if not set at construction

        :return: Frames-per-second as a float
        """
        return self._fps

    def reset(self) -> bool:
        """Revert the video to the first frame

        :return: Always true
        """
        self._acquire_lock()
        self.release()
        self._frame_counter = int(0)
        self._release_lock()
        return True

    def set_frame_index(self, frame_index: FrameIndex) -> bool:
        """Advance the video to the given frame index

        :param frame_index: Desired frame index
        :return:  True if success, false otherwise
        """
        self._acquire_lock()
        if frame_index < 0:
            self._logger.exception("The frame index must be >= 0")
            raise ValueError
        self._frame_counter = frame_index
        ret = True
        self._release_lock()
        return ret

    def get_next_frame(self) -> FrameOrNone:
        """Get the next frame in the video

        :return: frame or None
        """
        if not self.is_opened():
            return None
        self._acquire_lock()
        frame = self._frames[self._frame_counter]
        self._frame_counter += 1
        self._release_lock()
        return frame

    def get_frame_index(self, frame_index: FrameIndex) -> FrameOrNone:
        """Skip to given frame index of video

        :param frame_index: Frame index starting with 1
        :return: A frame, None otherwise
        """
        if frame_index <= 0:
            self._logger.exception("The input frame index must be >0")
            raise ValueError
        if frame_index > self.frames:
            return None
        if frame_index > self._frame_counter + 1:
            self.set_frame_index(frame_index)
        return self.get_next_frame()


class VideoPlayback(BasePlayback):
    """Commonly accessed video properties in an easy-to-use class"""

    def __init__(
        self,
        name: str,
        vid_path: PathLike,
    ):
        """Commonly accessed video properties in an easy-to-use class

        :param name: Object name
        :param vid_path: Path to video file
        """
        super().__init__(name)
        self.__cap: cv.VideoCapture

        self.path = Path(vid_path)
        if not self.path.exists():
            self._logger.exception("The file %s does not exist", os.fspath(vid_path))
            raise OSError
        self.__cap = cv.VideoCapture(os.fspath(self.path))
        self._channels: int = 3

    def release(self):
        """Release the video capture from memory

        :return: None
        """
        self._acquire_lock()
        if self.__cap:
            self.__cap.release()
        self._release_lock()

    def reset(self) -> bool:
        """Revert the video to the first frame

        :return: Always true
        """
        self._acquire_lock()
        self.release()
        self.__cap = cv.VideoCapture(os.fspath(self.path))
        self._frame_counter = int(0)
        self._release_lock()
        return True

    def __del__(self):
        self.release()
        del self.__cap

    @cached_property
    def fps(self) -> float:
        """Get the video frames-per-second (FSP)

        :return: fps : float
        """
        return float(self.__cap.get(cv.CAP_PROP_FPS))

    @cached_property
    def frames(self) -> FrameIndex:
        """Get the video number of frames

        :return: total number of frames
        """
        return int(self.__cap.get(cv.CAP_PROP_FRAME_COUNT))

    @cached_property
    def width(self) -> int:
        """Get the video frame width

        :return: width : int
        """
        return int(self.__cap.get(cv.CAP_PROP_FRAME_WIDTH))

    @cached_property
    def height(self) -> int:
        """Get the video frame height

        :return: height : int
        """
        return int(self.__cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    @cached_property
    def color_channels(self) -> int:
        """Get the number of color channels. If the video is valid, there should >0.
        Otherwise 0

        :return: Number of color channels
        """
        return self._channels

    @cached_property
    def frame_size(self) -> tuple[int, int]:
        """Get the frame size/dimensions

        :return: width, height
        """
        return self.width, self.height

    @cached_property
    def frame_size_full(self) -> tuple[int, int, int]:
        """Get the frame size/dimensions including color channels

        :return: (width, height, color channels)
        """
        return self.width, self.height, self.color_channels

    @cached_property
    def length_sec(self) -> float:
        """Get the length in seconds

        :return: movie length in seconds : float
        """
        if self.frames >= 0 and self.fps > 0.0:
            return float(self.frames / self.fps)
        return 0.0

    # @cached_property
    # def bitrate_kbps(self) -> float:
    #     """Get the video bitrate in kilobits per second
    #
    #     :return: bitrate (kbps) : float
    #     """
    #     return float(self.__cap.get(cv.CAP_PROP_BITRATE))

    @cached_property
    def size_bytes(self) -> int:
        """Get the video size in bytes"""
        return self.path.stat().st_size

    def is_opened(self) -> bool:
        """Query if the video is open

        :return: True if open, False otherwise
        """
        self._acquire_lock()
        ret = bool(self.__cap.isOpened())
        self._release_lock()
        return ret

    def set_frame_index(self, frame_index: int) -> bool:
        """Advance the video to the given frame index

        :param frame_index: Desired frame index
        :return:  True if success, false otherwise
        """
        self._acquire_lock()
        if frame_index < 0:
            raise ValueError("The frame index must be >= 0")
        self._frame_counter = frame_index
        ret = bool(self.__cap.set(cv.CAP_PROP_POS_FRAMES, self._frame_counter))
        self._release_lock()
        return ret

    def get_next_frame(self) -> FrameOrNone:
        """Get the next frame in the video

        :return: frame or None
        """
        ret: bool
        frame: FrameOrNone
        if not self.is_opened():
            return None
        self._acquire_lock()
        ret = self.__cap.grab()
        if not ret:
            self._logger.error(
                "Unable to open file %s at frame %s",
                os.fspath(self.path),
                self._frame_counter,
            )
            self._release_lock()
            return None
        _, frame = self.__cap.retrieve()
        self._frame_counter += 1
        self._release_lock()
        return frame

    def get_frame_index(self, frame_index: int) -> FrameOrNone:
        """Skip to given frame index of video

        :param frame_index: Frame index starting with 1
        :return: A frame, None otherwise
        """
        if frame_index <= 0:
            raise ValueError("The input frame index must be >0")
        if frame_index > self.frames:
            return None
        if frame_index > self._frame_counter + 1:
            self.set_frame_index(frame_index)
        return self.get_next_frame()

    def get_frame_time(self, time_sec: float) -> FrameOrNone:
        """Skip to given time in seconds

        :param time_sec: Time in seconds
        :return: A frame, None otherwise
        """
        if time_sec < 0:
            raise ValueError("The input time is less than 0!")
        return self.get_frame_index(int(np.ceil(self.fps * time_sec)))

    def set_frame_buffer_size(self, buffer_size: int) -> None:
        """Set the size of the number of cached frames for a get_next_frame operation

        :param buffer_size: A buffer size greater than 0
        """
        self._acquire_lock()
        self.__cap.set(cv.CAP_PROP_BUFFERSIZE, buffer_size)
        self._release_lock()
