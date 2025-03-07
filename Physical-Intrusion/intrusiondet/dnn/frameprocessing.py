"""Process frame through a DNN"""
import logging
import logging.handlers
import os
from collections import deque
from queue import Empty, Queue
from time import sleep
from typing import Callable, Optional, Sequence, Final

from intrusiondet.core.logging import LOG_LEVEL
from intrusiondet.core.types import FrameIndex, FrameOrNone
from intrusiondet.dnn.basednnmodel import BaseDNNModel
from intrusiondet.model.detectedobject import DetectedObject


def frame_processing_configurer(queue: Queue) -> None:
    """Configure the frame processing process with a queue handler for logging.
    Recipe taken from Python standard library cookbook

    :param queue: Logging queue
    :return: None
    """
    handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)


def put_frame_and_predictions_into_queue(
    priority: FrameIndex,
    frame: FrameOrNone,
    detections: list[DetectedObject],
    frame_predictions_queue: Queue,
    sleep_time_sec: float,
) -> None:
    """Place frames and predictions into the output queue

    :param priority: Frame index
    :param frame: Image frame
    :param detections: List of detections
    :param frame_predictions_queue: Output queue
    :param sleep_time_sec: Sleep time between queue being full
    """
    while frame_predictions_queue.full():
        sleep(sleep_time_sec)
    # Queue items are pickled and pickling causes problems with SQLAlchemy model
    # objects. So convert detected objects to a dictionary for the pickle
    # process
    data = frame, [det_obj.asdict() for det_obj in detections]
    queue_item = (priority, data)
    logging.getLogger().debug(
        "PID %d: Placing predictions at frame %d into queue",
        os.getpid(),
        priority
    )
    frame_predictions_queue.put_nowait(queue_item)


def frame_processing(
    frame_queue: Queue,
    frame_predictions_queue: Queue,
    model: BaseDNNModel,
    conf_thr: float,
    nms_thr: float,
    frame_index_max: FrameIndex,
    **kwargs,
) -> None:
    """Multiprocessing method to process frames from a queue and save the results in
    another queue

    :param frame_queue: Queue instance containing indices and frames
    :param frame_predictions_queue: Priority queue instance to store output of YOLO
        processing. The items are stored the following format:
        ```priority, data = (frame_index, (frame, detected objects))```
    :param model: A DNN model instance to process the frames
    :param conf_thr: Confidence threshold
    :param nms_thr: Non-maximum suppression threshold
    :param frame_index_max: Maximum frame index to accept
    :param kwargs: Supported keywords are sleep_time_sec and full_playback
    :keyword sleep_time_sec: Sleep time in seconds if the detected object queue is full
    :keyword full_playback: Whether to process all frames (True), or for fast
        processing, asynchronously clear the frame queue once a frame is gotten (False)
    :keyword batch_size: Number of frames to process in the model (default=1)
    :raises KeyError: If any other keys are provided
    """
    print("Entered frame processing", flush=True)
    logger = logging.getLogger()
    os_pid: Final[int] = os.getpid()
    frame_queue_deque: Optional[deque] = None
    try:
        frame_queue_deque = frame_queue.queue
    except AttributeError:
        pass
    if frame_queue_deque is None:
        try:
            if hasattr(frame_queue, "get_attribute"):
                frame_queue_deque = frame_queue.get_attribute("queue")
        except (AttributeError,):
            pass

    sleep_time_sec: float = kwargs.get("sleep_time_sec", 1e-6)
    full_playback: bool = kwargs.get("full_playback", True)
    batch_size: int = kwargs.get("batch_size", 1)

    priority: int
    data: tuple[FrameOrNone, list[dict]]
    queue_item: tuple[FrameIndex, tuple[FrameOrNone, list[dict]]]
    det_objects: list[DetectedObject]
    frame_index: FrameIndex
    frame: FrameOrNone

    batch_items_list: list[tuple[FrameIndex, FrameOrNone]] = []
    print("PID %s: Start DNN frame processing loop", os_pid, flush=True)
    while True:
        try:
            # logger.debug(f"Get frame from previous frame {frame_index}")
            frame_index, frame = frame_queue.get_nowait()
            if full_playback is False:
                if frame_queue_deque:
                    frame_queue_deque.clear()
                else:
                    try:
                        while frame_queue.qsize():
                            _ = frame_queue.get_nowait()
                    except (Exception,):
                        pass
        except Empty:
            # logger.debug("Queue is empty")
            # In case the frame_queue is empty
            sleep(sleep_time_sec)
            if not frame_queue.empty():
                continue
            frame = None
            frame_index = 0

        if frame is None or frame_index > frame_index_max or frame_index < 1:
            # No more frames to get, prep to break the while loop
            print("PID %d: No more frames to get", os_pid, flush=True)
            put_frame_and_predictions_into_queue(
                priority=0,
                frame=None,
                detections=[],
                frame_predictions_queue=frame_predictions_queue,
                sleep_time_sec=sleep_time_sec,
            )
            break

        if batch_size == 1:
            print(
                "PID %d: Processing single frame batch at index %d",
                os_pid,
                frame_index,
                flush=True
            )
            det_objects = model.process_frame(frame, conf_thr, nms_thr)
            put_frame_and_predictions_into_queue(
                frame_index, frame, det_objects, frame_predictions_queue, sleep_time_sec
            )
        else:
            # A batch of frames is queued into a list until the batch size requirement
            # is met, then process the batch
            batch_item: tuple[FrameIndex, FrameOrNone]
            if len(batch_items_list) == batch_size:
                indices_in_batch, frames_in_batch = zip(*batch_items_list)
                print(
                    "PID %d: Processing frame batch size %d for frames %s",
                    os_pid,
                    batch_size,
                    str(indices_in_batch),
                    flush=True
                )
                per_frame_det_objects = model.process_frames(
                    frames_in_batch,
                    conf_thr,
                    nms_thr,
                )
                for enum_index, (stored_frame_index, stored_frame) in enumerate(
                    batch_items_list
                ):
                    det_objects = per_frame_det_objects[enum_index]
                    put_frame_and_predictions_into_queue(
                        stored_frame_index,
                        stored_frame,
                        det_objects,
                        frame_predictions_queue,
                        sleep_time_sec,
                    )
                # The batch queue is cleared for another iteration
                batch_items_list.clear()
            batch_item = frame_index, frame
            batch_items_list.append(batch_item)
    print("PID %d: Done with DNN frame processing loop", os_pid, flush=True)
    print("After processing final queue size is ", frame_predictions_queue.qsize())


def start_frame_processing(
    logging_queue: Queue,
    frame_queue: Queue,
    frame_predictions_queue: Queue,
    model_constr: Callable,
    model_args: tuple,
    conf_thr: float,
    nms_thr: float,
    frame_index_max: FrameIndex,
    **frame_processing_kwargs,
) -> None:
    """Start processing frame queue through a DNN

    :param logging_queue: Queue for logging messages
    :param frame_queue: Queue for holding pre-processed frames
    :param frame_predictions_queue: Queue for hold post-processed frames
    :param model_constr: Callable constructor method for a DNN class
    :param model_args: Arguments for the DNN class constructor
    :param conf_thr: Confidence threshold
    :param nms_thr: Non-maximum suppression threshold
    :param frame_index_max: Maximum frame index to accept
    :param frame_processing_kwargs: Keyword arguments for the frame_processing method
    :return:
    """
    frame_processing_configurer(logging_queue)
    logger = logging.getLogger()
    print("PID %d: Start frame DNN processing", os.getpid(), flush=True)
    model: BaseDNNModel = model_constr(*model_args)
    print("Created Base DNN model", flush=True)
    frame_processing(
        frame_queue,
        frame_predictions_queue,
        model,
        conf_thr,
        nms_thr,
        frame_index_max,
        **frame_processing_kwargs,
    )
    print("PID %d: Done with frame DNN processing", os.getpid(), flush=True)
