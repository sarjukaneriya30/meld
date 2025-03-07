"""Methods that obtain video frames and prepare them for computer vision processing"""
import logging
import logging.handlers
import os
from multiprocessing.queues import Queue
from time import sleep
from typing import Optional, Union, Final

from intrusiondet.core.logging import LOG_LEVEL
from intrusiondet.core.types import (
    FrameIndex,
    FrameOrNone,
    IntTuple,
    NDArrayInt,
    PathLike,
)
from intrusiondet.model.videoplayback import BasePlayback, FramesPlayback, VideoPlayback

try:
    import cv2.cv2 as cv
except ImportError:
    import cv2 as cv


def frame_queueing_configurer(queue: Queue) -> None:
    """Configure the frame queueing process with a queue handler for logging.
    Recipe taken from Python standard library cookbook

    :param queue: Log queue
    :return:
    """
    handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)


def video_frame_queueing(
    index_frame_queue: Queue,
    playback: BasePlayback,
    pool_size: int,
    frame_start_index: FrameIndex = 1,
    frame_index_step: FrameIndex = 1,
    **kwargs,
) -> None:
    """Method to queue video frames.

    :param index_frame_queue: Queue instance to store frames
    :param playback: Playback instance to advance the movie
    :param pool_size: The number of processes that will perform detections. This is used
     as to populate None frames as the same number of processing processes
    :param frame_start_index: The starting frame index
    :param frame_index_step: Frame step size
    :keyword resize: Resize the frame to the specified dimensions (default=None)
    :keyword sleep_time_sec: Sleep time in seconds while a queue is full (default=1e-6)
    :raises KeyError: If any other keys are provided
    :return: None
    """
    logger = logging.getLogger()
    os_pid: Final[int] = os.getpid()
    current_frame_index = frame_start_index
    sleep_time_sec: float = kwargs.get("sleep_item_sec", 1.0e-6)
    resize: Optional[IntTuple] = kwargs.get("resize")
    total_frames_to_queue = (
        playback.frames - (frame_start_index - 1)
    ) // frame_index_step
    info_message = (
        "PID %s: Queuing %d frames of %d possible frames, capturing every %d frames"
        " starting at frame %d"
    )
    info_message_args = (
        os_pid,
        total_frames_to_queue,
        playback.frames,
        frame_index_step,
        frame_start_index,
    )
    logger.info(info_message, *info_message_args)
    while True:
        print("Entered frame queueing loop", flush=True)
        frame: FrameOrNone = playback.get_frame_index(current_frame_index)
        logger.debug("PID %d: Got frame at index %d", os_pid, current_frame_index)
        if playback.frame_counter < playback.frames:
            if frame is not None:
                if resize is not None and frame.shape[:2] != resize:
                    frame = cv.resize(frame, resize)
                while index_frame_queue.full():
                    logger.debug(
                        "PID %d: Frame queueing paused as queue is full at index %d",
                        os_pid,
                        current_frame_index,
                    )
                    sleep(sleep_time_sec)
                logger.debug(
                    "PID %d: Queueing frame index %d",
                    os_pid,
                    current_frame_index
                )
                index_frame_queue.put((current_frame_index, frame))
            else:
                logger.error(
                    "PID %d: Unable to obtain frame at index %d, skipping",
                    os_pid,
                    current_frame_index
                )
            current_frame_index += frame_index_step
            if current_frame_index >= playback.frames:
                print("No more frames to process, breaking from frame queueing", flush=True)
                break
        else:
            logger.info(
                "PID %d: At final index %d, breaking from frame queueing",
                os_pid,
                playback.frame_counter,
            )
            for _ in range(pool_size):
                index_frame_queue.put((0, None))
            break
    playback.release()
    logger.info("PID %d: Video is now closed", os_pid)


def start_video_frame_queueing(
    logging_queue: Queue,
    index_frame_queue: Queue,
    frames: Union[PathLike, NDArrayInt],
    pool_size: int,
    start_frame_index: FrameIndex = 1,
    frame_index_step: FrameIndex = 1,
    **kwargs,
) -> None:
    """Start a frame queue process

    :param logging_queue: Logging queue
    :param index_frame_queue: Frame storage queue
    :param frames: Path to a video with frames or an array of frames
    :param pool_size: The number of processes in the pool.  This is used as to populate
     None frames as the same number of processing processes
    :param start_frame_index: Starting frame index
    :param frame_index_step: Step size between frames
    :param kwargs: Keyword arguments for the video_frame_queueing method
    :return: None
    """
    print("Entered video frame queueing", flush=True)
    frame_queueing_configurer(logging_queue)
    logger = logging.getLogger()
    print("PID %s: Starting video frame queueing", os.getpid(), flush=True)
    playback: BasePlayback
    if isinstance(frames, str):
        playback = VideoPlayback(f"video-playback_PID{os.getpid()}", os.fspath(frames))
    else:
        playback = FramesPlayback(f"frames-playback_PID{os.getpid()}", frames)
    video_frame_queueing(
        index_frame_queue,
        playback,
        pool_size,
        start_frame_index,
        frame_index_step,
        **kwargs,
    )
    logger.info("PID %d: Done with video frame queueing", os.getpid())
