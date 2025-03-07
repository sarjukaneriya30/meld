"""Details about a detected object from a DNN, including location in frame and time"""
from __future__ import annotations

import datetime as stdlib_datetime
import json
from collections import UserList
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Final, Optional, Sequence, Union

from intrusiondet.core import serializer
from intrusiondet.core.converters import DEFAULT_DATETIME_STR, datetimeify
from intrusiondet.core.types import (
    AttributeKey,
    ClassID,
    ClassNameID,
    DateTimeLike,
    FrameIndex,
    LocationID,
    TimeStampInSeconds,
)
from intrusiondet.model.framing import FrameBox

DETECTED_OBJECT_DEFAULT_STR: Final[str] = "UNKNOWN"
"""Default string for any detected object string attribute"""

DETECTED_OBJECT_DEFAULT_INT: Final[int] = -1
"""Default string for any detected object integer attribute"""

DETECTED_OBJECT_DEFAULT_FLOAT: Final[float] = -1.0
"""Default string for any detected object float attribute"""


class DetectedObjectValidClassIdEnum(IntEnum):
    """Set an invalid class ID value as anything less than 0"""

    INVALID_CLASS_ID: Final[ClassID] = -1
    """Valid class IDs are >=0, equivalently invalid class IDs are always negative"""


class DetectedObject:
    """Store the detected/predicted classes from YOLO computer vision model"""

    def __init__(
        self,
        class_id: Optional[ClassID] = None,
        class_name: Optional[ClassNameID] = None,
        conf: Optional[float] = None,
        bbox_left: Optional[FrameIndex] = None,
        bbox_top: Optional[FrameIndex] = None,
        bbox_width: Optional[FrameIndex] = None,
        bbox_height: Optional[FrameIndex] = None,
        datetime_utc: Optional[DateTimeLike | TimeStampInSeconds] = None,
        location: Optional[LocationID] = None,
        zones: Optional[Union[str, Sequence[str]]] = None,
        intrusion: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """Store the detected/predicted classes from YOLO computer vision model

        :param class_id: Class ID number
        :param class_name: Class name
        :param conf: Confidence value
        :param bbox_left: Left point of bounding box
        :param bbox_top: Top point of bounding box
        :param bbox_width: Width of bounding box
        :param bbox_height: Height of bounding box
        :param datetime_utc: Datetime of the detection
        :param location: Location name of the detection
        :param zones: Sequence of identified zones
        :param intrusion: Indicator if detection constitutes intrusion
        :param filename: Video filename of the detection
        """
        if class_id is None:
            class_id = DETECTED_OBJECT_DEFAULT_INT
        self.class_id: Final[ClassVar[ClassID]] = ClassID(class_id)
        """Class enumerated ID for the detected object"""

        if class_name is None:
            class_name = DETECTED_OBJECT_DEFAULT_STR
        self.class_name: Final[ClassVar[ClassNameID]] = ClassNameID(class_name)
        """Confidence of prediction between 0 and 1"""

        if conf is None:
            conf = DETECTED_OBJECT_DEFAULT_FLOAT
        self.conf: Final[ClassVar[float]] = float(conf)
        """Confidence of prediction between 0 and 1"""

        if bbox_left is None:
            bbox_left = FrameIndex()
        self.bbox_left: Final[ClassVar[FrameIndex]] = FrameIndex(bbox_left)
        """Bounding (frame) box left-point for the object"""

        if bbox_top is None:
            bbox_top = FrameIndex()
        self.bbox_top: Final[ClassVar[FrameIndex]] = FrameIndex(bbox_top)
        """Bounding (frame) box top-point for the object"""

        if bbox_width is None:
            bbox_width = FrameIndex()
        self.bbox_width: Final[ClassVar[FrameIndex]] = FrameIndex(bbox_width)
        """Bounding (frame) box width for the object"""

        if bbox_height is None:
            bbox_height = FrameIndex()
        self.bbox_height: Final[ClassVar[FrameIndex]] = FrameIndex(bbox_height)
        """Bounding (frame) box height for the object"""

        if datetime_utc is None:
            datetime_utc = datetimeify(DEFAULT_DATETIME_STR)
        self.datetime_utc: Final[ClassVar[stdlib_datetime.datetime]] = datetimeify(
            datetime_utc
        )
        """The datetime of the observed/predicted object"""

        if location is None:
            location = DETECTED_OBJECT_DEFAULT_STR
        self.location: Final[ClassVar[LocationID]] = LocationID(location)
        """The approximate location ID of where the video was captured"""

        if zones is None:
            zones = json.dumps([])
        self.zones: Final[ClassVar[str]] = (
            json.dumps(zones) if not isinstance(zones, str) else zones
        ).replace('"', "'")
        """A string listing the identified zones in where the object box intersects"""

        self.intrusion: Final[ClassVar[bool]] = bool(intrusion)
        """Indicator if detection constitutes intrusion (True) or not (False)"""

        self.filename: Final[ClassVar[str]] = (
            Path(filename).name if filename is not None else ""
        )
        """Video filename of the detection"""

    def asdict(self) -> dict[AttributeKey, Any]:
        """Return a dictionary of the data

        :return: dictionary of the dataclass
        """
        ret = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "conf": self.conf,
            "bbox_left": self.bbox_left,
            "bbox_top": self.bbox_top,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "datetime_utc": self.datetime_utc,
            "location": self.location,
            "zones": self.zones,
            "intrusion": self.intrusion,
            "filename": self.filename,
        }
        for key, old_value in ret.items():
            try:
                new_value = serializer.serialize_with_exception(old_value)
                ret[key] = new_value
            except serializer.SerializeError:
                continue
        return ret

    def __eq__(self, other: DetectedObject):
        return self.asdict() == other.asdict()

    @property
    def box(self) -> FrameBox:
        """Convert the box coordinates into a FrameBox instance

        :return: Framebox
        """
        return FrameBox(
            left=self.bbox_left,
            top=self.bbox_top,
            width=self.bbox_width,
            height=self.bbox_height,
        )

    def unpack_zones(self) -> list[LocationID]:
        """Take the zones string and convert to a list of location IDs

        :return: List of location IDs

        >>> det_obj = DetectedObject(zones=["abc", "123"])
        >>> det_obj.zones
        "['abc', '123']"
        >>> det_obj.unpack_zones()
        ['abc', '123']
        """
        return json.loads(self.zones.replace("'", '"'))

    def prettyprint(self, sep: str = " ") -> str:
        """Pretty __str__ version of the class with the confidence set to an integer

        :param sep: A separator string between metadata
        :return: An easier to reader string representation
        """
        return (
            f"DetectedObject: {sep}"
            f"Class: {self.class_id},{sep}"
            f"Confidence: {int(self.conf * 100)}%,{sep}"
            f"FrameBox: {self.box.asdict()},"
            f"Detection datetime UTC: {self.datetime_utc},"
            f"Detected Zones: {self.zones}"
        )

    def is_valid_class_id(self) -> bool:
        """Runs a check if the class ID is >= 0, useful to distinguish between instances
        with the default constructor

        :return: True if a valid class id >= 0, false otherwise

        >>> DetectedObject(class_id=0).is_valid_class_id()
        True
        >>> DetectedObject(class_id=-1).is_valid_class_id()
        False
        """
        if self.class_id is None:
            return False
        return self.class_id > DetectedObjectValidClassIdEnum.INVALID_CLASS_ID


class DetectedObjectsList(UserList[DetectedObject]):
    """Simple list class to test equality of detected objects"""

    def __init__(self, detections: Sequence[DetectedObject]) -> None:
        """Simple list class to test equality of detected objects

        :param detections: A sequence of detected objects
        """
        super().__init__(detections)

    def __eq__(self, other: DetectedObjectsList):
        if len(self.data) != len(other.data):
            return False
        return all(
            self.data[index] == other.data[index] for index in range(len(self.data))
        )


def sort_key_detected_object_by_datetime_utc(
    det_obj: DetectedObject,
) -> stdlib_datetime:
    """Sorting key for DetectedObject instances

    :param det_obj: DetectedObject instance
    :return: Detection datetime at UTC
    """
    return det_obj.datetime_utc


def set_new_properties(other: DetectedObject, **new_props) -> DetectedObject:
    """Get a new detected object with new properties

    :param other: Input detected object instance
    :param new_props: Properties dictionary
    :return: New detected object

    >>> det = DetectedObject(class_id=0)
    >>> det.class_id
    0
    >>> det = set_new_properties(det, **{"class_id": 1})
    >>> det.class_id
    1
    """
    other_dict = other.asdict()
    other_dict.update(new_props)
    return DetectedObject(**other_dict)


def set_invalid_class_id(other: DetectedObject) -> DetectedObject:
    """Changes the value of the class ID to an invalid (-1) value

    :returns: New detected object with invalid class ID
    """
    return set_new_properties(
        other, class_id=ClassID(DetectedObjectValidClassIdEnum.INVALID_CLASS_ID)
    )


def correct_detections_datetime_naive_to_utc(
    detections: Sequence[DetectedObject],
) -> DetectedObjectsList:
    """Replace any detected object instance with a naive datetime instance to UTC-aware

    :param detections: Sequence of detected objects
    :return: New detected object sequence
    """
    return DetectedObjectsList(
        [
            set_new_properties(det_obj, datetime_utc=datetimeify(det_obj.datetime_utc))
            if det_obj.datetime_utc.tzinfo is None
            else det_obj
            for det_obj in detections
        ]
    )


def get_earliest_latest_detection_datetimes_from_sequence(
    detections: Sequence[DetectedObject],
) -> Union[tuple, tuple[stdlib_datetime, stdlib_datetime]]:
    """Get the earliest and latest detected objects detection datetimes

    :param detections: Input sequence of detections that are datetime-aware
    :return: Earliest and latest detections, if any are available
    """
    if len(detections) == 0:
        return tuple()
    detections = sorted(detections, key=sort_key_detected_object_by_datetime_utc)
    return detections[0].datetime_utc, detections[-1].datetime_utc


def check_detections_for_activity_at_locations(
    detected_objects: Sequence[DetectedObject],
    location_ids: Sequence[LocationID],
) -> dict[LocationID, list[DetectedObject]]:
    """For a sequence of detected objects, cross-reference a sequence of locations
    to determine which locations have which detections. This is aggregating values by
    location ID keys. If no detected objects are found, none is used instead

    :param detected_objects: Sequence of detected objects (values)
    :param location_ids: Sequence of location IDs (keys)
    :return: Mapping location ID --> Sorted list of detected objects by earliest to
     latest datetime
    """
    out_dict = {
        loc_id: sorted(
            [det_obj for det_obj in detected_objects if det_obj.location == loc_id],
            key=sort_key_detected_object_by_datetime_utc,
        )
        for loc_id in location_ids
    }
    return out_dict
