"""Model of video metadata received from a video reader"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime as stdlib_datetime
from datetime import timedelta as stdlib_timedelta
from functools import cache, cached_property
from itertools import zip_longest
from pathlib import Path

from attr import asdict, attrib, attrs, validators

from intrusiondet.core.converters import DEFAULT_DATETIME_STR, datetimeify
from intrusiondet.core.types import AttributeKey, CameraID, FrameIndex, PathLike
from intrusiondet.model.videoplayback import VideoPlayback


def prepend_video_filename_with_name(name: str, video_filename: PathLike):
    """Prepend to a video file name the name of the computer vision processing model
    output

    :param name: Prepend name
    :param video_filename: Video file name
    :return: Updated video filename
    """
    in_video_filename = Path(video_filename)
    out_video_filename = in_video_filename.parent / "".join(
        (name, in_video_filename.name)
    )
    return os.fspath(out_video_filename)


@cache
def get_length_sec_cached(
    filename: PathLike,
    start_access_utc: stdlib_datetime,
    end_access_utc: stdlib_datetime,
    fast: bool = False,
) -> float:
    """Get the video length in seconds from a saved video

    :param filename: Filename of the video
    :param start_access_utc: Start access datetime attribute
    :param end_access_utc: End access datetime attribute
    :param fast: Use the fast and integral result stored in the class. False gets the
        microseconds as well
    :return: Length in seconds
    """
    if fast is True:
        delta_t: stdlib_timedelta = end_access_utc - start_access_utc
        return delta_t.total_seconds()
    video_properties = VideoPlayback("length_calc", os.fspath(filename))
    return video_properties.length_sec


@cache
def get_num_frames_cached(
    filename: PathLike,
) -> FrameIndex:
    """Get the number of frame from a saved video

    :param filename: Filename of the video
    :return: Number of frames
    """
    video_properties = VideoPlayback("num_frames_calc", os.fspath(filename))
    return video_properties.frames


@cache
def get_fps_cached(filename: PathLike) -> float:
    """Get the frames-per-second for the video

    :param filename: Path to the video file
    :return: Frames-per-second
    """
    video_properties = VideoPlayback("fps_calc", os.fspath(filename))
    return video_properties.fps


@attrs(kw_only=True, frozen=True)  # , eq=True)
class SavedVideo:
    """Metadata properties for a video prior to detections"""

    camera_id: CameraID = attrib(
        default=-1,
        converter=CameraID,
        validator=validators.instance_of(CameraID),
    )
    """Unique ID for the camera (decimal or hexadecimal unknown)"""

    start_access_utc: stdlib_datetime = attrib(
        converter=datetimeify,
        default=datetimeify(DEFAULT_DATETIME_STR),
        repr=lambda datetime: datetime.timestamp(),
        validator=validators.instance_of(stdlib_datetime),
        #  eq=False,
    )
    """Datetime-aware instance when the video was first captured"""

    end_access_utc: stdlib_datetime = attrib(
        converter=datetimeify,
        default=datetimeify(DEFAULT_DATETIME_STR),
        repr=lambda datetime: datetime.timestamp(),
        validator=validators.instance_of(stdlib_datetime),
        #  eq=False,
    )
    """Datetime-aware instance when the video stop capturing"""

    filename: str = attrib(
        default="", converter=os.fspath, validator=validators.instance_of(str)
    )
    """Path to the image or video"""

    def asdict(
        self,
    ) -> dict[AttributeKey, CameraID | float | str | stdlib_datetime]:
        """Get a dictionary of the dataclass. Converted fields are re-serialized

        :return: Dictionary of the dataclass
        """
        ret = asdict(self)
        ret["start_access_utc"] = ret["start_access_utc"].timestamp()
        ret["end_access_utc"] = ret["end_access_utc"].timestamp()
        return ret

    @cached_property
    def frames(self) -> FrameIndex:
        """Get the number of frames in the video"""
        return get_num_frames_cached(self.filename)

    @cached_property
    def fps(self) -> float:
        """Get the number of frames per second in the video"""
        return get_fps_cached(self.filename)

    def get_datetime_at_index(self, frame_index: FrameIndex) -> stdlib_datetime:
        """Get the datetime for the given frame index

        :param frame_index: Frame index counter
        :return: Datetime when the frame occurred
        """
        fps = self.fps
        if fps > 0:
            return self.start_access_utc + stdlib_timedelta(
                seconds=((frame_index - 1) / fps)
            )
        logging.getLogger().error(
            "The video %s does not a valid fps (%f)", self.filename, self.fps
        )
        return self.start_access_utc

    @classmethod
    def reconstruct_from_filename(
        cls,
        filename: PathLike,
        schema: list[str],
        pattern: str | list[str],
        regex: bool = False,
    ):
        """Try to reconstruct video properties from a filename given a keyword schema
        and delimiter pattern strings. If the pattern is a regular expression, set
        regex to True

        :param filename: Filename
        :param schema:
        :param pattern:
        :param regex:
        :return: SavedVideo

        >>> test_name = "./90_B827EBD43C1E_capture_1661373394.243765-1661373397.239304_90_0_image_1280x720.mp4"
        >>> test_schema = ["ACCOUNT_ID", "CAMERA_ID", "NAME", "START_ACCESS_UTC", "END_ACCESS_UTC", "ACCOUNT_ID", "VIDEO_ID", "IMAGE_TYPE", "RESOLUTION"]
        >>> test_pattern =  ["_", "-"]
        >>> test_regex = False
        >>> sv = SavedVideo.reconstruct_from_filename(test_name, test_schema, test_pattern, test_regex)
        >>> print(sv.asdict())
        {'camera_id': 'B827EBD43C1E', 'start_access_utc': 1661373394.243765, 'end_access_utc': 1661373397.239304, 'filename': '90_B827EBD43C1E_capture_1661373394.243765-1661373397.239304_90_0_image_1280x720.mp4'}
        """
        saved_video_properties = SavedVideo().asdict()
        filename_path = Path(filename)
        saved_video_properties["filename"] = os.fspath(filename_path)
        if regex is False and isinstance(pattern, list):
            pattern = "".join(["[" + "".join(pat) + "]|" for pat in pattern])
            pattern = pattern.rstrip("|")
        elif regex is True and isinstance(pattern, str):
            pass
        else:
            print("Failure")
        try:
            filename_path = Path(filename)
            filename_without_suffix: str = filename_path.name.rstrip(
                filename_path.suffix
            )
            keywords: list[str] = schema.copy()
            values: list[str] = re.split(pattern, filename_without_suffix)
            for key, value in zip_longest(keywords, values, fillvalue=None):
                if not isinstance(key, str):
                    continue
                key_lower = key.lower()
                if key_lower in saved_video_properties:
                    match key_lower:
                        case "start_access_utc":
                            value = float(value)
                        case "end_access_utc":
                            value = float(value)
                        case _:
                            pass
                    saved_video_properties[key_lower] = value
        except Exception as err:
            print(f"Error occurred: {err}")
        return cls(**saved_video_properties)

    @classmethod
    def read_from_json(cls, json_file: PathLike):
        """Read from a JSON metadata file the video properties

        :param json_file: JSON file containing metadata
        :return: SavedVideo
        """
        logging.getLogger().info(
            "Reading JSON metadata file %s to load video properties",
            os.fspath(json_file),
        )
        with Path(json_file).open(encoding="utf-8") as json_file_pointer:
            data = json.load(json_file_pointer)
            return cls(**data)

    def is_datetime_within_start_stop_bounds(
        self, test_datetime_utc: stdlib_datetime
    ) -> bool:
        """Test if a particular datetime is between the start-stop time window

        :return: True if bounded start <= test <= end, false otherwise
        """
        return self.start_access_utc <= test_datetime_utc <= self.end_access_utc

    def prepend_filename_with_name(self, name: str) -> str:
        """Prepend to a video file name the name of the computer vision processing model
        output

        :param name: Prepend name for the file, not the path
        :return: Full path with prepended filename
        """
        return prepend_video_filename_with_name(name, Path(self.filename))


def sort_key_saved_video_by_start_access_utc(
    saved_video: SavedVideo,
) -> stdlib_datetime:
    """Sorting key for SavedVideo instances

    :param saved_video: SavedVideo instance
    :return: Start access datetime at UTC
    """
    return saved_video.start_access_utc
