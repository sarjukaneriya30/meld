"""
Physical Intrusion software that combines warehouse business logic and computer vision for object detection.

usage: PhysicalIntrusion.py [-h] -c CONFIGURATION [--debug]

options:
  -h, --help            show this help message and exit
  -c CONFIGURATION, --configuration CONFIGURATION
                        Path to the configuration file for this run (default: configs/config.toml)
  --debug               Enables debug logging (default: False)

"""
import argparse
import asyncio
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import queue
import time
from asyncio import AbstractEventLoop
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Optional, Sequence
from azure.storage.blob import BlobServiceClient, ContainerClient

import numpy as np
import skvideo.io
from attr import fields
#from azure.eventhub import EventData
#from azure.eventhub.aio import EventHubConsumerClient, PartitionContext
#from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore
from azure.storage.blob import BlobClient, BlobServiceClient
from dotenv import load_dotenv

from intrusiondet import bootstrapper
from intrusiondet.bootstrapper import create_sql_repository_connection
from intrusiondet.config.configstruct import IntrusionDetectionConfig
from intrusiondet.core import alerting
from intrusiondet.core.colors import KPMG_COLOR_PALETTE
from intrusiondet.core.image import get_frame
from intrusiondet.core.logging import listener_configurer, listener_process
from intrusiondet.core.opencvfrontend import draw_prediction_from_detected_object
from intrusiondet.core.serializer import DateTimeJSONEncoder
from intrusiondet.core.types import (
    FrameIndex,
    FrameOrNone,
    IntOrFloat,
    LocationID,
    TimeDifferentialInSeconds,
)
from intrusiondet.dnn.classnames import (
    ClassNameConverter,
    class_converter_from_supernames_directory,
    convert_class_id,
)
from intrusiondet.dnn.frameprocessing import start_frame_processing
from intrusiondet.dnn.framequeueing import start_video_frame_queueing
from intrusiondet.dnn.yolo import YoloModel
from intrusiondet.model import detectedobject, framing, savedvideo
from intrusiondet.model.detectedobject import DetectedObject, \
    sort_key_detected_object_by_datetime_utc
from intrusiondet.model.videoplayback import VideoPlayback
from intrusiondet.model.videoreadercamera import VideoReaderCamera
from intrusiondet.model.warehousemanagement import (
    OrderLine,
    WarehouseWorkLog,
    WorkOrder,
)
from intrusiondet.orm.sqlalchemyrepository import SQLAlchemyRepository

cv = None
try:
    import cv2.cv2 as cv
except ImportError:
    import cv2 as cv
finally:
    assert cv is not None


# fmt: off
BOOTSTRAPPER: Optional[bootstrapper.IntrusionDetectionBootstrapper] = None
"""Class to help bootstrap program setup"""


CONFIG: Optional[IntrusionDetectionConfig] = None
"""Class that define global runtime configuration"""


WARN_ICON: FrameOrNone = None
"""An icon to overlay on a video frame when an anomalous object is detected/predicted"""


LOGGING_QUEUE: Optional[mp.Queue] = None
"""Queue that receives logging messages between processes"""


EVENT_HUB_CONSUMER_NAME: Final[str] = "pi"
"""Azure EventHub consumer name of subscription"""


EVENT_HUB_CONSUMER_GROUP_STRING: Final[str] = "$Default"
"""Azure EventHub consumer group string"""


EVENT_HUB_CONSUMER_CONNECTION_STRING: Final[str] = "Endpoint=sb://eventhubpi.servicebus.usgovcloudapi.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=VXsnB/EVD9xNuy/BF7t3l7gBxBN7GjrINsd7rJFc5ns="
"""Azure EventHub consumer subscription connection string"""


BLOB_STORAGE_CONTAINER_NAME: Final[str] = "computer-vision-videos-testing"
"""Azure BlobStorage container name holding videos"""


BLOB_STORAGE_CONNECTION_STRING: Final[str] = "DefaultEndpointsProtocol=https;AccountName=storagetrigger;AccountKey=CjKu5Fop0BezTdL3+nuCza2JsnloYghF81y87gy9mVnRSDJQ5R+4ijAZLjRYGLa/d08QGG/ihwsq+AStN/nhxw==;EndpointSuffix=core.usgovcloudapi.net"
"""Azure BlobStorage container connection string"""


LOGGING_FILENAME: Final[str] = "PhysicalIntrusion_runtime.log"
"""Runtime output log for the program"""


ACTIVITY_LOG: Final[str] = "PhysicalIntrusion.log"
"""Activity log for physical intrusion"""


WAREHOUSE_WORK_LINES_TABLE_CONNECTION_STRING: Final[str] = """Driver={ODBC Driver 18 for SQL Server};Server=10.13.0.68\SQLHOSTINGSERVE,1433;Database=BYODB;Uid=d365admin;Pwd=!!USMC5G!!;Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"""
"""Connection string to warehouse work order lines tables"""


WAREHOUSE_WORK_LINES_TABLE_SELECT_STRING: str = (
    """SELECT *  FROM "WHSWarehouseWorkLineStaging";"""
)
"""Initial select query from the warehouse work lines table"""


REPOSITORY: Optional[SQLAlchemyRepository] = None
print("REPOSITORY INITIALIZED ON LINE 138 " + str(type(REPOSITORY)))
"""Data repository"""
# fmt: on


def physical_intrusion_listener_configurer(log_queue: mp.Queue, level: int) -> None:
    """Setup logging queue listener process

    :param log_queue: Log queue
    :param level: Log level enumeration
    """
    handler = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)



def update_class_with_converter(
    detected_objects: list[DetectedObject],
    class_converter: ClassNameConverter,
) -> None:
    """Update class IDs for detected objects based on the classname converter

    :param detected_objects: Sequence of detected objects
    :param class_converter: Class name converter class
    """
    if class_converter is None:
        return
    for det_obj_index in range(len(detected_objects)):
        det_obj = detected_objects[det_obj_index]
        new_id = convert_class_id(det_obj.class_id, class_converter)
        new_name = class_converter.new[new_id]
        if new_id is None or new_name in CONFIG.detobj.ignore.list:
            detected_objects[det_obj_index] = detectedobject.set_invalid_class_id(
                det_obj
            )
            continue
        detected_objects[det_obj_index] = detectedobject.set_new_properties(
            det_obj, class_id=new_id, class_name=new_name
        )


def update_frame_with_detections(
    frame: FrameOrNone,
    detected_objects: Sequence[DetectedObject],
    classnames: list[str],
    show: bool = False,
) -> bool:
    """Display the image frame with the detected objects. Press "q" on your keyboard to
    exit the program.

    :param frame: Video image frame
    :param detected_objects: Sequence of detected objects
    :param classnames: Classname list mapped from class IDs
    :param show: Show the frame
    :return: True if frame updated, false if not or user hit escape char
    """
    try:
        global WARN_ICON
        if WARN_ICON is None:
            root_dir = Path(__file__).parent
            WARN_ICON = get_frame(
                os.fspath(root_dir / "assets" / "icons" / "warn_alert_gradient.png")
            )
            WARN_ICON = cv.resize(WARN_ICON, (min(frame.shape[:2]) // 4,) * 2)
        for det_obj in detected_objects:
            if not det_obj.is_valid_class_id():
                continue
            classname = classnames[det_obj.class_id]
            if classname in CONFIG.detobj.ignore.list:
                continue
            class_color = KPMG_COLOR_PALETTE.get(CONFIG.detobj.colors[classname])
            if class_color is None:
                raise KeyError(
                    f"Unable to find the color for the specified class: {classname}"
                )
            if classname in CONFIG.detobj.anomaly.list:
                # fmt: off
                # Update the visual to include a warning icon in the upper right corner
                frame[
                    0: WARN_ICON.shape[0],
                    frame.shape[1] - WARN_ICON.shape[1]:,
                    :,
                ] = WARN_ICON
                # fmt: on
            draw_prediction_from_detected_object(
                frame,
                det_obj,
                window_name=CONFIG.frontend.window_name if show else None,
                # font_scale=font_scale,
                # font_thickness=font_thickness,
                box_color=class_color.as_bgr(),
            )
        if show:
            cv.imshow(CONFIG.frontend.window_name, frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                logger = logging.getLogger()
                logger.debug(
                    "Escape character 'q' was pressed, breaking from " "processing"
                )
                return False
            return True
    except (Exception,) as err:
        logger = logging.getLogger()
        logger.error("Exception occurred: %s", str(err))
        return False
    return True


def filter_for_anomaly_or_person(
    detections: list[DetectedObject]
) -> Optional[DetectedObject]:
    """If any anomaly objects are found, return a detection anomaly that is intrusion
    positive or intrusion agnostic like a person

    :param detections: All detections from video
    :return: Candidate physical intrusion detected object
    """
    ret_det_obj: Optional[DetectedObject] = None
    detections.sort(key=sort_key_detected_object_by_datetime_utc)
    for det_obj in detections:
        if det_obj.class_name in CONFIG.detobj.anomaly.list:
            return det_obj
        if ret_det_obj is None:
            if "person" in det_obj.class_name.lower():
                ret_det_obj = det_obj
    return ret_det_obj


def get_next_frame_with_objects(
    frame_predictions_priority_queue: mp.Queue,
    target_frame_index: FrameIndex,
) -> tuple[FrameIndex, tuple[FrameOrNone, list[DetectedObject]]]:
    """Get the processed items and push them to the priority queue until a target frame
    index, the priority

    :param frame_predictions_priority_queue: Priority queue with frame and list detected
        objects as the data
    :param target_frame_index: Target frame index
    :return: Priority queue item (index, (frame, list-detected-objects))
    """
    frame_index: FrameIndex
    frame: FrameOrNone
    ret_item: tuple[FrameIndex, tuple[FrameOrNone, list[DetectedObject]]]
    queue_item: tuple[FrameIndex, tuple[FrameOrNone, list[dict]]]
    detected_objects: list[dict]
    empty_queue_output: Final[tuple] = (0, (None, []))

    logger = logging.getLogger()
    while True:
        try:
            # logger.debug("Get next frame with predictions")
            queue_item = frame_predictions_priority_queue.get_nowait()
        except queue.Empty:
            # logger.debug("Empty queue")
            continue

        # If there are no more frames to process
        frame_index, (frame, detected_objects) = queue_item
        if frame_index < 1 or frame is None:
            logger.debug("Got a frame, but at index %d is invalid", frame_index)
            return empty_queue_output

        # If processing asynchronously, only get the next sequential frame
        if CONFIG.proc.full_playback:
            if frame_index != target_frame_index:
                frame_predictions_priority_queue.put_nowait(queue_item)
                continue
        # For the smoothest playback possible, ignore all but the most recent frame
        else:
            if frame_index < target_frame_index:
                continue
            frame_predictions_priority_queue.queue.clear()
        logger.debug(
            "Got a frame at index %d with %d predictions",
            frame_index,
            len(detected_objects),
        )
        ret_item = (
            frame_index,
            (
                frame,
                [DetectedObject(**det_obj_dict) for det_obj_dict in detected_objects],
            ),
        )
        return ret_item


def detect(
    saved_video: savedvideo.SavedVideo,
    placeholder_list
):
    """Process a movie using the designated YOLO model.

    :param saved_video: Video metadata as described by a SavedVideo instance
    :return: Sequence of all detected objects from the video
    """
    # Configure repository and logging mechanism
    global LOGGING_QUEUE
    logger = BOOTSTRAPPER.logger
    if logger is None:
        logger = logging.getLogger()
    start_time = datetime.now()
    logger.info("Beginning detections bootstrapping")

    # Try to get camera zone mappings
    zones: Optional[dict[str, framing.FramePartition]]
    try:
        logger.info("Trying to get camera zones if they are defined")
        camera = BOOTSTRAPPER.cameras[saved_video.camera_id]
        zones = camera.zones
    except (KeyError,):
        logger.info("No zones found. Continuing without")
        zones = None

    logger.info("Loading video frames into memory")
    playback = VideoPlayback("video", os.fspath(saved_video.filename))

    # Save an output video of the detections
    result_video_fps = np.ceil(playback.fps)
    result_video_output_path = Path(CONFIG.remote.video.captures_path)
    result_video_output_path.mkdir(exist_ok=True)
    result_video_filename = os.fspath(
        result_video_output_path
        / saved_video.prepend_filename_with_name(CONFIG.model.type)
    )
    # Frames to save for the video with the detection boxes and class names
    result_video_frames: list[tuple[FrameIndex, FrameOrNone]] = []

    logger.info(
        "Building dummy YOLO model to determine if classnames are remapped to"
        " supernames"
    )
    yolo_master_reference = bootstrapper.build_dnn_yolo_model(CONFIG)
    classnames = yolo_master_reference.classes
    class_converter: Optional[ClassNameConverter] = None
    if CONFIG.model.supernames_path:
        old_names_filename = CONFIG.model.path / CONFIG.model.supernames_path.name
        class_converter = class_converter_from_supernames_directory(
            old_names_filename=os.fspath(old_names_filename),
            new_names_directory=os.fspath(CONFIG.model.supernames_path),
        )
        classnames = class_converter.new

    logger.info("Define a new frame size dimensions")
    new_frame_size = playback.frame_size
    if isinstance(CONFIG.proc.image_resize, float):
        new_frame_size = tuple(
            int(CONFIG.proc.image_resize * dim) for dim in new_frame_size
        )

    if CONFIG.frontend.enable:
        logger.info("Show an output window")
        cv.namedWindow(CONFIG.frontend.window_name, cv.WINDOW_AUTOSIZE)
    else:
        logger.info("No output window to display")

    # Determine which frame to start processing at and how many frames to queue in total
    # depending on the skip interval
    frame_start_index: FrameIndex = 0
    frame_index_step: FrameIndex = 1
    if playback.fps is not None:
        if isinstance(CONFIG.proc.start_time_sec, IntOrFloat):
            frame_start_index = max(
                int(np.ceil(CONFIG.proc.start_time_sec * playback.fps)), 1
            )
        if isinstance(CONFIG.proc.skip_every_sec, IntOrFloat):
            frame_index_step = max(
                int(np.ceil(CONFIG.proc.skip_every_sec * playback.fps)), 1
            )
        if frame_index_step + frame_start_index > playback.frames:
            frame_index_step = playback.frames - 1

    logger.info("Create a multiprocessing pool to process frames in YOLO")
    pool = mp.Pool(CONFIG.proc.processes)
    print("Here 1", flush=True)
    multiple_results: list[mp.pool.AsyncResult] = []
    print("Here 2", flush=True)
    #time.sleep(5) # TODO remove, this is a test to see if a delay here stops the program from hanging
    mp_manager = mp.Manager()
    print("Here 3", flush=True)
    #time.sleep(5)
    # Frame queue items are --> tuple[FrameIndex, FrameOrNone]
    frame_queue = mp_manager.Queue(maxsize=CONFIG.proc.queue_maxsize)
    print("Here 4", flush=True)
    print("Start a single pool process to queue frames", flush=True)
    #time.sleep(5) # TODO remove, this is a test to see if a delay here stops the program from hanging
    pool.apply_async(
        func=start_video_frame_queueing,
        args=(
            LOGGING_QUEUE,
            frame_queue,
            os.fspath(saved_video.filename),
            CONFIG.proc.processes,
            frame_start_index,
            frame_index_step,
        ),
        kwds={"sleep_time_sec": CONFIG.proc.sleep_time_sec, "resize": new_frame_size},
    )
    total_frames_to_queue: Final[int] = (
        playback.frames - (frame_start_index - 1)
    ) // frame_index_step
    frame_processing_processes_max: Final[int] = CONFIG.proc.processes - 1
    frame_queue_size = frame_queue.qsize()
    print("About to enter frame_queue_size while loop", flush=True)

    previous_value = None

    while frame_queue_size < 2 * frame_processing_processes_max * CONFIG.proc.batch_size:
        if previous_value != frame_queue_size:
            print(f"Frame queue size: {frame_queue_size}", flush=True)
            previous_value = frame_queue_size
        if frame_queue_size >= total_frames_to_queue - 1:
            break
        time.sleep(CONFIG.proc.sleep_time_sec)
        frame_queue_size = frame_queue.qsize()
    print("Exited frame_queue_size while loop", flush=True)

    # In the same pool above, start processing the queued frames
    # Frame predictions priority queue takes output from YOLO. The items are
    # --> tuple[FrameIndex, tuple[FrameOrNone, Sequence[DetectedObject]
    frame_predictions_priority_queue = mp_manager.Queue(
        maxsize=CONFIG.proc.queue_maxsize
    )
    processing_kwargs: dict[str, Any] = {
        "sleep_time_sec": CONFIG.proc.sleep_time_sec,
        "full_playback": CONFIG.proc.full_playback,
        "batch_size": CONFIG.proc.batch_size,
    }
    # Create YOLO model processes in the pool. If the target processor is not CPU, then
    # the pool is fully populated. Else, just one pool process is started.
    logger.info("Creating child processes in pool to process frames in YOLO")
    model_constr = YoloModel
    for process_counter in range(frame_processing_processes_max):
        model_args = (
            CONFIG.model.type + f"child_{process_counter}",
            CONFIG.model.path,
            CONFIG.model.backend,
            CONFIG.model.target,
        )
        async_result = pool.apply_async(
            func=start_frame_processing,
            args=(
                LOGGING_QUEUE,
                frame_queue,
                frame_predictions_priority_queue,
                model_constr,
                model_args,
                CONFIG.model.confidence,
                CONFIG.model.nms,
                playback.frames,
            ),
            kwds=processing_kwargs,
        )
        multiple_results.append(async_result)
        if CONFIG.model.target != cv.dnn.DNN_TARGET_CPU:
            break

    logger.info(
        "Processing video with %d additional processes...", len(multiple_results)
    )

    print("About to enter frame_predictions_priority_queue while loop", flush=True)
    previous_value = None
    while frame_predictions_priority_queue.qsize() < 1:
        if previous_value != frame_predictions_priority_queue.qsize():
            print(
                f"Frame predictions priority queue size: {frame_predictions_priority_queue.qsize()}",
                flush=True,
            )
            previous_value = frame_predictions_priority_queue.qsize()
        time.sleep(CONFIG.proc.sleep_time_sec)
    print("Exited frame_predictions_priority_queue while loop", flush=True)

    # Get processed frames and detections until all frames are processed. The while-loop
    # below will continuous get-put predictions back into itself until the target frame
    # is obtained
    target_frame_index: FrameIndex = frame_start_index
    detected_objects_all: list[DetectedObject] = []
    detected_objects_dump_sequence: list[dict] = []
    continue_status: bool = True
    frame_is_none_counter: int = 0

    # Hinting outside of loop
    frame: FrameOrNone
    detection_location: LocationID
    detection_zones: list[LocationID]
    detection_time: datetime
    detected_objects: list[DetectedObject]
    frame_with_predictions: tuple[FrameIndex, tuple[FrameOrNone, list[DetectedObject]]]
    saved_detections: list[DetectedObject]

    while continue_status is True:
        frame_with_predictions = get_next_frame_with_objects(
            frame_predictions_priority_queue, target_frame_index
        )
        frame_index, (frame, detected_objects) = frame_with_predictions

        # for the next iteration
        target_frame_index += frame_index_step

        if frame is None:
            frame_is_none_counter += 1
            # No more frames obtained, break from loop. All processes must push None
            # frames. Each None frame obtained is handled once in processing methods
            if frame_is_none_counter >= frame_processing_processes_max:
                break
            time.sleep(CONFIG.proc.sleep_time_sec)
            continue

        # Provide naive detections into aware detections with datetime and physical
        # location metadata
        update_class_with_converter(detected_objects, class_converter)
        saved_detections = []
        for det_obj_index in range(len(detected_objects)):
            det_obj = detected_objects[det_obj_index]
            if not det_obj.is_valid_class_id():
                continue
            detection_zones = []
            detection_time = saved_video.get_datetime_at_index(frame_index)
            try:
                camera = BOOTSTRAPPER.cameras[saved_video.camera_id]
                detection_location = camera.location
                detection_zones = [
                    zone_name
                    for zone_name, partition in zones.items()
                    if partition.has_any_overlap_with(det_obj.box)
                ]
            except (KeyError,):
                detection_location = detectedobject.DETECTED_OBJECT_DEFAULT_STR
            detected_objects[det_obj_index] = detectedobject.set_new_properties(
                det_obj,
                datetime_utc=detection_time,
                zones=detection_zones,
                location=detection_location,
                filename=Path(result_video_filename).name,
            )
            det_obj = detected_objects[det_obj_index]
            saved_detections.append(det_obj)
            logger.info(
                "!!!=== Detected object %s at location %s at time %s, it was found in"
                " the following zones %s ===!!!",
                det_obj.class_name,
                detection_location,
                detection_time.isoformat(),
                str(det_obj.zones),
            )
            # logger.info("Attempting to create DB record for above detection")
        continue_status = update_frame_with_detections(
            frame, saved_detections, classnames, CONFIG.frontend.enable
        )
        if CONFIG.frontend.enable:
            if continue_status is False or (cv.waitKey(1) & 0xFF == ord("q")):
                print("Escape character pressed")
                logger.info(
                    "Escape character 'q' was pressed, breaking from processing"
                )
                try:
                    while frame_queue.qsize():
                        _ = frame_queue.get_nowait()
                except (Exception,):
                    pass
                try:
                    while frame_predictions_priority_queue.qsize():
                        _ = frame_predictions_priority_queue.get_nowait()
                except (Exception,):
                    pass
        if frame is not None and frame_index >= 1:
            write_frame = frame.copy()
            # Swap blue and red channels as OpenCV uses BGR convention and FFMPEG output
            # is set to RGB
            write_frame[:, :, 0] = frame[:, :, 2]
            write_frame[:, :, 2] = frame[:, :, 0]
            item = (frame_index, write_frame)
            result_video_frames.append(item)
            if CONFIG.proc.skip_every_sec > 0.0:
                frames_to_write = int(
                    np.ceil((result_video_fps - 1) * CONFIG.proc.skip_every_sec)
                )
                _ = [result_video_frames.append(item) for _ in range(frames_to_write)]
        detected_objects_all.extend(saved_detections)
        detected_objects_dump_sequence.append(
            {
                "frame_index": frame_index,
                "objects": [det_obj.asdict() for det_obj in saved_detections],
            }
        )
    # With the frame processing completed, either by the user or no more frames found,
    # close the processing pool and save data
    pool.close()
    pool.join()
    mp_manager.shutdown()
    logger.info("Pool closed and done")

    if CONFIG.frontend.enable is True or len(CONFIG.frontend.window_name) > 0:
        cv.destroyAllWindows()

    logger.info(
        "Saving processed frames with detections shown into video %s",
        os.fspath(result_video_filename),
    )
    result_video = skvideo.io.FFmpegWriter(
        filename=os.fspath(result_video_filename),
        inputdict={
            "-r": str(result_video_fps),
            "-s": "{}x{}".format(playback.width, playback.height),
        },
        outputdict={
            "-c": "h264",
            "-r": str(result_video_fps),
            "-pix_fmt": "yuv420p",
            "-c:v": "libx264",
            "-preset": "ultrafast",
            "-crf": "17",
        },
        verbosity=1,
    )
    result_video._proc = None
    result_video_frames.sort(key=lambda priority_frame_tuple: priority_frame_tuple[0])
    for frame_index, frame in result_video_frames:
        if frame is not None and frame_index >= 1:
            result_video.writeFrame(frame)
    result_video.close()
    logger.info("Video %s has been written", os.fspath(result_video_filename))

    if isinstance(CONFIG.output.json_dump_path, Path):
        logger.info(
            "Dumping most recent detections to file %s",
            os.fspath(CONFIG.output.json_dump_path),
        )
        with CONFIG.output.json_dump_path.open(mode="w") as json_dump_fp:
            out_json = {
                "video": Path(saved_video.filename).name,
                "detected_objects": detected_objects_dump_sequence,
            }
            if saved_video:
                out_json.update(saved_video.asdict())
            json.dump(out_json, json_dump_fp, indent=4, cls=DateTimeJSONEncoder)
    logger.info("Committing detections to database repository")
    end_time = datetime.now()
    total_time_seconds = (end_time - start_time).total_seconds()
    logger.info(f"The total time to process the video was {total_time_seconds} seconds")

    # adding code to support process Timeout
    placeholder_list.pop(0) # remove placeholder value in list

    # add detections to placeholder list
    for detection in detected_objects_all:
        placeholder_list.append(detection) # add to placeholder list; must be append method
    logger.info('Placeholder list now contains {} detections; should be {}'.format(len(placeholder_list), len(detected_objects_all)))
    return placeholder_list
    # return detected_objects_all


def update_detections_for_intrusion(
    saved_video: savedvideo.SavedVideo, relevant_work: set[WorkOrder]
) -> None:
    """Set intrusion to true if a video has no associated work orders

    :param saved_video: Video metadata
    :param relevant_work: Any found work orders
    """
    global REPOSITORY
    logger = logging.getLogger()
    filter_filename = Path(
        saved_video.prepend_filename_with_name(CONFIG.model.type)
    ).name

    # If no work, find all detections and mark as intrusion
    if len(relevant_work) == 0:
        try:
            logger.info(
                "No work orders were found corresponding to detections from video %s."
                "Setting all detections as intrusion",
                filter_filename,
            )
            REPOSITORY.session.query(DetectedObject).filter(
                DetectedObject.filename == filter_filename
            ).update({"intrusion": True}, synchronize_session="fetch")
            REPOSITORY.session.commit()
        except Exception as err:
            logger.error("There was an error: %s", str(err))


def determine_if_intrusion(
    saved_video: savedvideo.SavedVideo,
    detection_in_video: DetectedObject,
    work_order_intrusion_window_seconds: TimeDifferentialInSeconds = timedelta(
        hours=0.5
    ).total_seconds(),
    line_limit: int = 100
) -> int:
    """Find if the provided video with detections has corresponding work orders for
    physical intrusion determination

    :param saved_video: Saved video metadata
    :param detection_in_video: Possible intrusion detection from the video
    :param work_order_intrusion_window_seconds: Number of seconds to check for work
        before the video started
    :param line_limit: Number of lines to save in the summary log file
    :return: Physical intrusion state
    """
    global BOOTSTRAPPER
    global CONFIG
    logger = BOOTSTRAPPER.logger
    camera: VideoReaderCamera = BOOTSTRAPPER.cameras[saved_video.camera_id]
    activity_log = Path(ACTIVITY_LOG)
    all_lines: list[str] = []
    if activity_log.exists():
        with activity_log.open("r", encoding="utf-8") as activity_log_file_read_pointer:
            all_lines = activity_log_file_read_pointer.readlines()
            len_all_lines = len(all_lines)
            if len_all_lines > line_limit:
                all_lines = all_lines[len_all_lines-line_limit:len_all_lines]

    error_message: str
    error_message_args: tuple[Any, ...]
    notification: str
    notification_args: tuple[Any, ...]
    with activity_log.open("w") as activity_log_file_write_pointer:
        if camera is None:
            error_message = (
                "%s: "
                "The video %s does NOT have an associated video reader loaded into"
                " memory. This is a critical issue that must be resolved. Please create"
                " a video reader of Camera ID %s"
            )
            error_message_args = (
                datetime.now().isoformat(),
                str(saved_video),
                saved_video.camera_id,
            )
            logger.error(error_message, *error_message_args)
            all_lines.append((error_message + os.linesep) % (*error_message_args,))
            for line in all_lines:
                activity_log_file_write_pointer.write(line)

            return -1
        location_id = camera.location
        notification = (
            "A video was recorded at location %s between %s --> %s. Determining if"
            " intrusions were discovered"
        )
        notification_args = (
            location_id,
            saved_video.start_access_utc.isoformat(),
            saved_video.end_access_utc.isoformat(),
        )
        logger.info(notification, *notification_args)

        if detection_in_video.class_name in CONFIG.detobj.anomaly.list:
            notification = (
                "%s: "
                "Anomaly object %s at location %s between %s --> %s was discovered."
                " Intrusion has occurred!"
            )
            notification_args = (
                datetime.now().isoformat(),
                location_id,
                detection_in_video.class_name,
                saved_video.start_access_utc.isoformat(),
                saved_video.end_access_utc.isoformat(),
            )
            logger.info(notification, *notification_args)
            all_lines.append((notification + os.linesep) % (*notification_args,))
            for line in all_lines:
                activity_log_file_write_pointer.write(line)
            intrusion_in_video = detectedobject.set_new_properties(
                detection_in_video,
                intrusion=True
            )
            REPOSITORY.add(intrusion_in_video)
            REPOSITORY.session.commit()
            return 0

        order_line_fields = fields(OrderLine)
        metadata = order_line_fields.work_creation_datetime_utc.metadata
        work_creation_datetime_string = metadata["key"]
        isoformat_sep = metadata["isoformat_sep"]
        start_datetime_str = (
            (
                saved_video.start_access_utc
                - timedelta(seconds=work_order_intrusion_window_seconds)
            )
            .replace(tzinfo=None)
            .isoformat(sep=isoformat_sep).split('.')[0]
        )
        end_datetime_str = saved_video.end_access_utc.replace(tzinfo=None).isoformat(
            sep=isoformat_sep
        ).split('.')[0]

        querystring = WAREHOUSE_WORK_LINES_TABLE_SELECT_STRING.strip(";")
        querystring += (
            f""" WHERE "{work_creation_datetime_string}" > '{start_datetime_str}'"""
            f""" AND "{work_creation_datetime_string}" < '{end_datetime_str}';"""
        )
        logger.info("Executing SQL query %s", querystring)
        work_log: WarehouseWorkLog = WarehouseWorkLog.from_sql_database(
            WAREHOUSE_WORK_LINES_TABLE_CONNECTION_STRING,
            querystring,
            os.fspath("configs/d365/work_lines_format.json"),
        )
        logger.info(
            "Found work order lines:  %s",
            str(
                [
                    work_order.asdict()
                    for work_order in work_log.work_orders_by_work_id.work_orders
                ]
            ),
        )

        relevant_work_orders: set[WorkOrder] = set()
        logger.info("Location ID: {}".format(location_id))
        logger.info("Camera Regions of Interest: {}".format(camera.rois))
        for work_order in reversed(work_log.work_orders_by_datetime.work_orders):
            # add loop to look at each region of interest
            for roi in camera.rois:
                logger.info("looking at work order {}".format(work_order))
                logger.info("Comparing work order against location {}".format(roi))
                if not work_order.is_associated_with_location(roi): # used to be location_id
                    continue
                relevant_work_orders.add(work_order)

        if len(relevant_work_orders) == 0:
            notification = (
                "%s: "
                "For detection %s at location %s between %s --> %s, No work orders were"
                " found in the last %f-hours. Intrusion has occurred!"
            )
            notification_args = (
                datetime.now().isoformat(),
                detection_in_video.class_name,
                location_id,
                saved_video.start_access_utc.isoformat(),
                saved_video.end_access_utc.isoformat(),
                work_order_intrusion_window_seconds / 3600.0,
            )
            logger.info(notification, *notification_args)
            all_lines.append((notification + os.linesep) % (*notification_args,))
            for line in all_lines:
                activity_log_file_write_pointer.write(line)
            intrusion_in_video = detectedobject.set_new_properties(
                detection_in_video,
                intrusion=True
            )
            REPOSITORY.add(intrusion_in_video)
            REPOSITORY.session.commit()
            return alerting.ObservedActivityWorkOrderFlag.ACTIVITYNONEWORKORDER.value

        notification = (
            "%s: "
            "For detection %s at location %s between %s --> %s, the corresponding work"
            " orders were found in the last %f-hours: %s.%sStatus is normal."
        )
        notification_args = (
            datetime.now().isoformat(),
            detection_in_video.class_name,
            location_id,
            saved_video.start_access_utc.isoformat(),
            saved_video.end_access_utc.isoformat(),
            work_order_intrusion_window_seconds / 3600.0,
            str([work_order.work_id for work_order in relevant_work_orders]),
            os.linesep
        )

        logger.info(notification, *notification_args)
        all_lines.append((notification + os.linesep) % (*notification_args,))
        for line in all_lines:
            activity_log_file_write_pointer.write(line)
    return alerting.ObservedActivityWorkOrderFlag.ACTIVITYFOUNDWORKORDER.value

def physical_intrusion(parsed_args: argparse.Namespace) -> int:
    """Driver method that checks blob storage for new videos periodically

    :param parsed_args: Parsed command line arguments
    :return: 0 for success
    """
    print("Entered function physical intrusion ", flush=True)

    # Configuration runtime
    global BOOTSTRAPPER
    global CONFIG
    global REPOSITORY
    global LOGGING_QUEUE
    load_dotenv(".env")
    BOOTSTRAPPER = bootstrapper.bootstrap(config_path=parsed_args.configuration)
    level = logging.INFO # TODO change to INFO once done debugging
    if parsed_args.debug is True:
        level = logging.DEBUG
    physical_intrusion_listener_configurer(LOGGING_QUEUE, level)
    logger = BOOTSTRAPPER.logger = logging.getLogger()
    CONFIG = BOOTSTRAPPER.config

    logger.info("Environ variables: %s", str(os.environ))
    print("Environ variables: %s" % str(os.environ), flush=True)

    logger.info("Cmd line args: %s", str(vars(parsed_args)))
    print("Cmd line args: %s" % vars(parsed_args), flush=True)

    logger.info("Program configuration: %s", str(CONFIG))
    print(f"Program configuration: {str(CONFIG)}", flush=True)

    # Connect to the storage account.
    account_url = "https://physicalintrusion.blob.rack01.usmc5gsmartwarehouse.com/"
    container_name = "computer-vision-videos"
    credential = {
            "account_name": 'physicalintrusion',
            "account_key": '7l7O77QX69DbHssDjnkARx7PPWlcPf4mxUfXPNDYB/KZbA1RFVdAyzw4JJA99WGsXIeKV6Qny4B7L6SJhPBEjQ=='
        }
    container_client = ContainerClient(account_url = account_url, container_name = container_name, credential = credential, api_version = '2019-02-02')

    # If the container does not already exist, create it.
    if not container_client.exists():
        print("Creating container...", flush=True)
        container_client.create_container()
        print("Container created.", flush=True)
    else:
        print("Container already exists.", flush=True)

    # Continuously check for new videos and process them for physical intrusion

    while True:
        new_blob_found = False

        for blob in container_client.list_blobs():
            print("New blob name is " + str(blob.name), flush=True)
            with open(blob.name, "wb") as my_blob:
                blob_data = container_client.download_blob(blob.name)
                blob_data.readinto(my_blob)
            new_blob_found = True
            break

        if new_blob_found:
            try: # have to add exception catch so process will exit with timeout
                print("New blob found. Processing video...", flush=True)
                #logger.info("Video available. Processing video using computer vision")
                #logger.info("Parsing video for metadata from the filename")
                saved_video = savedvideo.SavedVideo.reconstruct_from_filename(
                    os.fspath(blob.name),
                    CONFIG.video.schema,
                    CONFIG.video.pattern,
                    CONFIG.video.regex,
                )

                print(
                    "Video was recorded between %s and %s with ID %s. Running"
                    " detections",
                    saved_video.start_access_utc.isoformat(),
                    saved_video.end_access_utc.isoformat(),
                    str(saved_video.camera_id), flush=True
                )

                #logger.info("About to create repository if it does not exist")

                # The SQLite database repository and ORM wrappers MUST be created before
                # detections are started
                if REPOSITORY is None:
                    #logger.info("Repository not yet created. Creating now.")
                    database_config = BOOTSTRAPPER.config.database
                    #logger.info("First login into database.")
                    repo_host = database_config.public.data["HOST"]
                    #logger.info("Obtained database host address %s", str(repo_host))
                    REPOSITORY = create_sql_repository_connection(BOOTSTRAPPER.config)

                #logger.info("Done with repository check")

                # This will find objects in the obtained video
                print("About to call detect function", flush=True)
                # detected_objects = detect(saved_video)

                # code to timeout detect function
                TIMEOUT = 10 # how long to process before we throw the towel in on this blob
                print('Starting process of detection with {} second timeout...'.format(TIMEOUT), flush=True)
                timeout_manager = mp.Manager()
                detected_objects = timeout_manager.list(['InstantiatedList']) # create list using manager so data can be passed between processes
                timeout_process = mp.Process(target=detect, args=(saved_video, detected_objects))
                timeout_process.start()
                timeout_process.join(TIMEOUT)
                if timeout_process.is_alive():
                    print('Function is hanging past our {} second timeout.'.format(TIMEOUT), flush=True)
                    timeout_process.terminate()
                    print('Process terminated for this video.', flush=True)
                    raise Exception('Video timed out; move to next video.')

                #logger.info(
               #     "Object detection completed with %d objects discovered. Running"
                #    " intrusion detection algorithms.",
               #     len(detected_objects),
                #)
                # We are concerned with intrusions, and only anomalies or persons are
                # considered
                #logger.info("Finding first candidate intruder or anomaly")
                candidate_physical_intrusion = filter_for_anomaly_or_person(
                    detected_objects
                )

                if candidate_physical_intrusion is not None:
                    # This will attempt to cross-reference work orders with the
                    # detections
                    #logger.info("Found candidate. Proceeding with intrusion logic")
                    determine_if_intrusion(saved_video, candidate_physical_intrusion)
                    #logger.info("Completed intrusion determination.")

                    deleted_records: list[DetectedObject] = (
                        REPOSITORY.keep_most_recent_detections(
                            CONFIG.database.prop.detections_max_rows
                        )
                    )
                    for det_obj in deleted_records:
                      #  logger.info(
                    #        "Intrusion %s at time %s is being removed from database"
                    #        " repository, including its detections video in %s",
                    #        det_obj.class_name,
                   #         det_obj.datetime_utc,
                     #       os.fspath(CONFIG.remote.video.captures_path)
                      #  )
                        detections_video = Path(
                            CONFIG.remote.video.captures_path
                        ) / det_obj.filename
                        if detections_video.exists():
                            os.remove(os.fspath(detections_video))
                        else:
                            logger.error(
                                "Unable to find detections video file %s",
                                os.fspath(detections_video)
                            )
                    REPOSITORY.session.commit()
                else:
                    print("No intrusion candidates found.", flush=True)
                    #logger.info("No intrusion candidates found.")

                # Delete the blob from the storage account.
                container_client.delete_blob(blob.name)
            except:
                print('Error parsing video.', flush=True)
           # logger.info("Running cleanup.")
            #except Exception as physical_intrusion_err:
            #    logger.error(
            #        "Error occurred between parsing video name and physical intrusion"
            #        " detection. The error was %s",
            #        str(physical_intrusion_err),
            #    )

            # Delete the video from local storage once processing is complete to preserve
            # storage
            try: # leave here bc we do want to delete the local copy if it doesnt work
                os.remove(os.fspath(blob.name))
                #logger.info("Successfully deleted %s locally.", blob.name)
            except Exception as err:
                #logger.exception("Unable to delete blob from local storage. %s", str(err))
                cleanup_successful = False




def main() -> int:
    """Main process that parses command line arguments

    :return: 0 for success
    """
    print("Run main for PhysicalIntrusion.py")
    global LOGGING_QUEUE

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "-c",
        "--configuration",
        type=str,
        default=os.fspath(Path(__file__).parent / "configs/config.toml"),
        help="Path to the configuration file for this run",
    )

    arg_parser.add_argument(
        "--debug", action="store_true", help="Enables debug logging"
    )

    parsed_args: argparse.Namespace = arg_parser.parse_args()
    mp_manager_main = mp.Manager()

    # Create a dedicated process for logging, necessary for multiprocessing logging
    LOGGING_QUEUE = mp_manager_main.Queue(-1)
    listener = mp.Process(
        target=listener_process,
        args=(LOGGING_QUEUE, listener_configurer, LOGGING_FILENAME),
    )
    listener.start()

    # Run Physical Intrusion, waiting for new videos asynchronously
    physical_intrusion_process = mp.Process(
        target=physical_intrusion, args=(parsed_args,)
    )
    print("Starting physical intrusion process", flush=True)
    physical_intrusion_process.start()
    physical_intrusion_process.join()

    LOGGING_QUEUE.put_nowait(None)
    listener.join()
    print("PhysicalIntrusion.py program complete")
    return 0


if __name__ == "__main__":
    main()
