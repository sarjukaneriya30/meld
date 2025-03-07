"""
Define regions and objects within an image frame. While this package uses classes that
inherit from the attrs dataclasses, they also have methods to interrogate each other.

# Classes

 * The `FramePoint` class is a Euclidean 2D-point that exists on an image grid. A set of
    two `FramePoint`'s defines a line segment on the plane. A point can also reside
    inside a polygon, a set of >2 points. In order to determine if a point lies within
    a polygon, a series of Winding Number tests are utilized on the polygon's line
    segments. The polygon class is called the `FramePartition`, which as the name
    implies, partitions an image frame.
 * The `FramePartition` class is a sequence of `FramePoint` objects that define image
    zones or regions. To ensure the polygon is closed, the first and last points
    argument must be equal meaning  ```(x0, y0) == (xN, yN)``` for a sequence N+1
    points. You can test if a point is inside the polygon, which then utilizes the
    Winding Number test mentioned on the `FramePoint` class.
 * The `FrameBox` class provides the role of defining a detected object box as output by
    the DNN. A box has a 2D-position called left and upper, width, and height
    dimensions. Since detected objects are presumably contained by a box, we are
    interested if the object box is inside a particular zone/region of an image.
    Therefore, a box has `FramePartition` functionality to test membership.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from enum import IntEnum
from pathlib import Path
from typing import Final, Optional, Sequence, Union

import attr

from intrusiondet.core.types import (
    AttributeKey,
    FrameIndex,
    FrameOrNone,
    LocationID,
    PathLike,
)

INVALID_FRAME_POINT_INDEX: Final[FrameIndex] = -1
"""Default values for a FramePoint"""


class FramePointLeftOfLineEnum(IntEnum):
    """Define the output result for the `is_left` test"""

    LEFT: Final = 1
    """Left of an infinite line is >=1"""

    ON: Final = 0
    """On an infinite line is exactly == 0"""

    RIGHT: Final = -1
    """Right of an infinite line is <=-1"""


class WindingNumberTestResultEnum(IntEnum):
    """Define the output result of the `FramePartition._winding_number_test` method
    test"""

    OUTSIDE: Final = 0
    """A point is outside a closed polygon only-if the winding number is 0"""


@attr.attrs(frozen=True)
class FramePoint:
    """A 2D-point representing a pixel coordinate on an image frame.
    __Signature__: FramePoint(hor, ver)"""

    hor: FrameIndex = attr.attrib(
        default=INVALID_FRAME_POINT_INDEX,
        validator=attr.validators.instance_of(FrameIndex),
        converter=FrameIndex,
        eq=True,
    )
    """Horizontal point"""

    ver: FrameIndex = attr.attrib(
        default=INVALID_FRAME_POINT_INDEX,
        validator=attr.validators.instance_of(FrameIndex),
        converter=FrameIndex,
        eq=True,
    )
    """Vertical point"""

    def __getitem__(self, item_index) -> FrameIndex:
        """Get the index appropriate value for the point

        :param item_index: The sequence index
        :return: Horizontal point if the argument is 0 or -2, vertical point if +/-1
        :raises IndexError: If the index is >2 or <-1.

        >>> p0 = FramePoint(1, 2)
        >>> p0[0], p0[0] == p0[-2]
        (1, True)
        >>> p0[1], p0[1] == p0[-1]
        (2, True)
        """
        return self.astuple()[item_index]

    def in_frame(self, test_frame: FrameOrNone) -> bool:
        """Is the point inside an image frame

        :param test_frame: The frame in question
        :return: True if inside the image, false otherwise
        """
        if test_frame is None:
            return False
        if min(self.astuple()) < 0:
            return False
        if self.ver >= test_frame.shape[0] or self.hor >= test_frame.shape[1]:
            return False
        return True

    def astuple(self) -> tuple[FrameIndex, FrameIndex]:
        """Return a tuple representation of the point

        :return: (hor, ver)

        >>> FramePoint(1, 2).astuple()
        (1, 2)
        """
        return self.hor, self.ver

    def asdict(self) -> dict[AttributeKey, FrameIndex]:
        """Return a dictionary of the mapping

        :return: Dictionary mapping

        >>> FramePoint(1, 2).asdict()
        {'hor': 1, 'ver': 2}
        """
        return dict((("hor", self.hor), ("ver", self.ver)))

    def as_ordereddict(self) -> OrderedDict[AttributeKey, FrameIndex]:
        """Return an OrderedDict

        >>> FramePoint(1, 2).as_ordereddict()
        OrderedDict([('hor', 1), ('ver', 2)])
        """
        return OrderedDict((("hor", self.hor), ("ver", self.ver)))

    def is_left(
        self, point_a: FramePoint, point_b: FramePoint
    ) -> FramePointLeftOfLineEnum:
        """Algorithm to determine if a Euclidean point is left, on, or right of an
        infinite line

        :param point_a: First point defining a line-segment
        :param point_b: Second point defining a line-segment
        :return: 0 if the point is on the line, +1 if the point is left of the line, or
            -1 if the point is right of the line

        >>> int(FramePoint(1, 10).is_left(FramePoint(1, 9), FramePoint(1, 11)))
        0
        >>> int(FramePoint(1, 10).is_left(FramePoint(2, 9), FramePoint(2, 11)))
        1
        >>> int(FramePoint(1, 10).is_left(FramePoint(0, 9), FramePoint(0, 11)))
        -1
        """
        # fmt: off
        result = (
            (point_b.hor - point_a.hor) * (self.ver - point_a.ver) -
            (self.hor - point_a.hor) * (point_b.ver - point_a.ver)
        )
        # fmt: on
        if result == FramePointLeftOfLineEnum.ON:
            return FramePointLeftOfLineEnum.ON
        if result > FramePointLeftOfLineEnum.ON:
            return FramePointLeftOfLineEnum.LEFT
        return FramePointLeftOfLineEnum.RIGHT


@attr.attrs(frozen=True)
class FrameBox:
    """A data container for detected object box in a frame.
    __Signature__: FramePoint(left, top, width, height)"""

    left: FrameIndex = attr.attrib(default=0, converter=FrameIndex, eq=True)
    """The left edge of the object box"""

    top: FrameIndex = attr.attrib(default=0, converter=FrameIndex, eq=True)
    """The top edge of the object box"""

    width: FrameIndex = attr.attrib(default=0, converter=FrameIndex, eq=True)
    """The width of the object box"""

    height: FrameIndex = attr.attrib(default=0, converter=FrameIndex, eq=True)
    """The height of the object box"""

    def astuple(self) -> tuple[FrameIndex, FrameIndex, FrameIndex, FrameIndex]:
        """Return a tuple repesentation of the box

        :return: (left, top, width, height)

        >>> f = FrameBox(1, 2, 3, 4)
        >>> f.astuple()
        (1, 2, 3, 4)
        """
        return self.left, self.top, self.width, self.height

    def asdict(self) -> dict[AttributeKey, FrameIndex]:
        """Return a dict repesentation of the box

        :return: A dictionary like so:
            {'left': l, 'top': t, 'width': w, 'height': h}
        :rtype: dict[str, int]

        >>> FrameBox(1, 2, 3, 4).asdict()
        {'left': 1, 'top': 2, 'width': 3, 'height': 4}
        """
        return attr.asdict(self)

    def as_partition(self) -> FramePartition:
        """Convert a box into a partition

        :return: Closed polygon/partition
        """
        bbox_points = [
            FramePoint(self.left, self.top),
            FramePoint(self.left, self.top + self.height),
            FramePoint(self.left + self.width, self.top + self.height),
            FramePoint(self.left + self.width, self.top),
            FramePoint(self.left, self.top),
        ]
        return FramePartition(bbox_points)

    def is_partially_inside_of(self, partition: FramePartition) -> bool:
        """Test if the box has at least one point inside a partition

        :param partition: Test partition
        :return: True if at least one point lies inside the partition, false otherwise

        >>> f_outer = FrameBox(1, 1, 5, 5)
        >>> f_inner = FrameBox(2, 2, 1, 1)
        >>> f_inner.is_partially_inside_of(f_outer.as_partition())
        True
        """
        return FramePartition.partition_partially_surrounds_partition(
            outer_partition=partition, test_partition=self.as_partition()
        )

    def is_partially_surrounding(self, partition: FramePartition) -> bool:
        """Test if the box partially surrounds a partition

        :param partition: Test partition
        :return: True if at least one point of a partition polygon point lies inside the
            box, false otherwise

        >>> f_outer = FrameBox(1, 1, 5, 5)
        >>> f_inner = FrameBox(2, 2, 1, 1)
        >>> f_outer.is_partially_surrounding(f_inner.as_partition())
        True
        """
        return FramePartition.partition_partially_surrounds_partition(
            outer_partition=self.as_partition(), test_partition=partition
        )


def build_frame_box(*args, **kwargs) -> FrameBox:
    """Generalized class method to build a FrameBox.

    :param args: Either a FrameBox instance itself, a kwarg mapping, or 4-tuple
    :param kwargs: Only a kwarg constructor
    :return: New frame box
    """
    try:
        if len(args) == 1:
            item = args[0]
            if isinstance(item, FrameBox):
                return item
            if isinstance(item, dict):
                return FrameBox(**item)
            if isinstance(args, Sequence):
                return FrameBox(*item)
        elif len(args) == 4:
            return FrameBox(*args)
        elif len(kwargs) == 4:
            return FrameBox(**kwargs)
        logger = logging.getLogger()
        logger.error(
            "Unable to build a FrameBox with the following inputs: Args: %s, and"
            " kwargs: %s",
            str(args),
            str(kwargs),
        )
    except (Exception,) as broad_exception:
        logger = logging.getLogger()
        logger.exception("Unable to build a FrameBox due to %s", str(broad_exception))
    logger.error("Building default frame box")
    return FrameBox()


class FramePartitionNotClosedException(Exception):
    """Exception raised when the constructor for the FramePartition is not closed
    meaning the first and last points are not equal"""


@attr.attrs(frozen=True)
class FramePartition:
    """A sequence of points defining a polygon which is contiguous portion of a frame.
    The first and last points are enforced to be equal. __Signature__:
    FramePartition([FramePoint(x0, y0), FramePoint(x1, y1), ..., FramePoint(x0, y0)])
    """

    points = attr.attrib(
        default=attr.Factory(list),
        validator=attr.validators.deep_iterable(
            iterable_validator=attr.validators.instance_of(Sequence),
            member_validator=attr.validators.instance_of(FramePoint),
        ),
        eq=True,
    )
    """The sequence of points that define a closed polygon"""

    @points.validator
    def check_is_closed(
        self, attribute: Optional[attr.Attribute] = None, value: list[FramePoint] = None
    ) -> None:
        """Validator method for the constructor

        :param attribute: The attr.Attribute
        :param value: The input point sequence
        :return: nothing
        :raises FramePartitionNotClosedException: If the first and last sequence of
            points are not equal or any point is the default FramePoint constructor
        """
        if value is None:
            value = self.points
        if len(value) > 2 and value[0] != value[-1]:
            logging.getLogger().exception(
                "The partition first and last points must be identical, got %s and %s",
                value[0],
                value[-1],
            )
            raise FramePartitionNotClosedException
        default_point = FramePoint()
        if any(point == default_point for point in value):
            logging.getLogger().exception(
                "The partition contains the invalid frame point %s",
                str(default_point.astuple()),
            )
            raise FramePartitionNotClosedException
        if not (attribute is None or isinstance(attribute, attr.Attribute)):
            logging.getLogger().exception(
                "The attribute is not None and not attr.Attribute class, but is a %s",
                attribute,
            )
            raise TypeError

    def fully_enclosed_by_frame(self, test_frame: FrameOrNone) -> bool:
        """Validator-like method to ensure if a partition exceeds the image frame
        shape/boundaries

        :param test_frame: Test frame
        :return: True if all points lie in or on the frame, false otherwise

        >>> import numpy as np
        >>> test_zero_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> partition = FrameBox(1, 1, 5, 5).as_partition()
        >>> partition.fully_enclosed_by_frame(test_zero_frame)
        True
        """
        if test_frame is None:
            return False
        return all(point.in_frame(test_frame) for point in self.points)

    def as_contour(
        self,
        member: type = list,
    ) -> list:
        """Convert the box to a closed contour partition. Useful as input with OpenCV
        polygon methods like cv2.fillPoly

        :param member: The type of the iterable member (default=list)
        :return: All points as a list of the specified member-type
        """
        # fmt: off
        if member in (list, tuple,):
            return list(member(point.astuple()) for point in self.points)
        if member in (dict,):
            return list(point.asdict() for point in self.points)
        if member in (OrderedDict,):
            return list(point.as_ordereddict() for point in self.points)
        # fmt: on
        logging.getLogger().exception(
            "The following type %s is not supported", str(member)
        )
        raise TypeError

    def _winding_number_test(self, test_point: FramePoint) -> int:
        """Algorithm to test if a point is inside a 2D polygon. Algorithm credit goes to
        Practical Geometry Algorithms by Daniel Sunday PhD
        <https://www.geomalgorithms.com/index.html>

        :param test_point: Test point
        :return: Non-zero if the test point is inside polygon, zero otherwise outside
            the polygon
        """
        winding_number: int = 0
        for point_index in range(0, len(self.points) - 1):
            point_a, point_b = self.points[point_index], self.points[point_index + 1]
            if point_a.ver <= test_point.ver:
                if point_b.ver > test_point.ver:
                    if (
                        test_point.is_left(point_a, point_b)
                        == FramePointLeftOfLineEnum.LEFT
                    ):
                        winding_number += 1
            else:
                if point_b.ver <= test_point.ver:
                    if (
                        test_point.is_left(point_a, point_b)
                        == FramePointLeftOfLineEnum.RIGHT
                    ):
                        winding_number -= 1
        return winding_number

    def is_enclosing_point(self, test_point: FramePoint) -> bool:
        """Test if a point is inside the partition as represented by a polygon.

        :param test_point: Test point
        :return: True if the point is inside the partition, false if outside the
            partition

        >>> partition = FrameBox(1, 1, 10, 10).as_partition()
        >>> partition.is_enclosing_point(FramePoint(5, 5))
        True
        """
        return (
            self._winding_number_test(test_point) != WindingNumberTestResultEnum.OUTSIDE
        )

    @staticmethod
    def partition_partially_surrounds_partition(
        outer_partition: FramePartition, test_partition: FramePartition
    ) -> bool:
        """Calculate if two polygons/partitions are partially overlapping.

        :param outer_partition: Candidate outer partition
        :param test_partition: Test partition for the outer partition candidate
        :return: True if any point of the test partition lies inside the outer
            partition, false otherwise
        """
        return any(
            outer_partition.is_enclosing_point(test_point)
            for test_point in test_partition.points
        )

    def is_partially_encolosing(
        self, inner_region: Union[FramePartition, FrameBox]
    ) -> bool:
        """Test if the given region has at least one point inside the partition. A
        convenient wrapper for the static method
        FramePartition.partition_partially_surrounds_partition

        :param inner_region: Candidate outer region partition, either a partition itself
            or a box
        :return: True if LHS partition has at least one point inside the candidate outer
            region. false otherwise

        >>> fp1, fp2 = FrameBox(5, 5, 10, 15), FrameBox(0, 0, 10, 10)
        >>> partition1, partition2 = fp1.as_partition(), fp2.as_partition()
        >>> partition2.is_partially_encolosing(fp1)
        True
        >>> partition2.is_partially_encolosing(partition1)
        True
        """
        inner_partition: FramePartition
        if isinstance(inner_region, FrameBox):
            inner_partition = inner_region.as_partition()
        else:
            inner_partition = inner_region
        return FramePartition.partition_partially_surrounds_partition(
            outer_partition=self, test_partition=inner_partition
        )

    def is_partially_inside_of(
        self, outer_region: Union[FramePartition, FrameBox]
    ) -> bool:
        """Test if the given outer region candidate encloses this partition instance. A
        convenient wrapper for the static method
        FramePartition.partition_partially_surrounds_partition

        :param outer_region: Candidate outer region partition, either a partition itself
            or a box
        :return: True if LHS partition has at least one point inside the candidate outer
            region. false otherwise

        >>> fp1, fp2 = FrameBox(5, 5, 10, 15), FrameBox(0, 0, 10, 10)
        >>> partition1, partition2 = fp1.as_partition(), fp2.as_partition()
        >>> partition1.is_partially_inside_of(fp2)
        True
        >>> partition1.is_partially_encolosing(partition2)
        True
        """
        outer_partition: FramePartition
        if isinstance(outer_region, FrameBox):
            outer_partition = outer_region.as_partition()
        else:
            outer_partition = outer_region
        return FramePartition.partition_partially_surrounds_partition(
            outer_partition=outer_partition, test_partition=self
        )

    def has_any_overlap_with(self, region: Union[FramePartition, FrameBox]) -> bool:
        """Test if the given region candidate has any overlapping points with this
        partition instance.

        :param region: Candidate region partition, either a partition itself or a box
        :return: True if either partition encloses/side the other, False otherwise

        >>> fp1, fp2 = FrameBox(5, 5, 10, 15), FrameBox(0, 0, 10, 10)
        >>> partition1, partition2 = fp1.as_partition(), fp2.as_partition()
        >>> partition1.has_any_overlap_with(partition2)
        True
        >>> partition2.is_partially_encolosing(partition1)
        True
        """
        return self.is_partially_encolosing(region) or self.is_partially_inside_of(
            region
        )


def build_partitions_mapping_from_json_file(
    json_filename: PathLike,
) -> dict[LocationID, FramePartition]:
    """Given a JSON file, construct a frame partition mapping. The file is a dictionary,
    where the keys are the partition name and the value is an array of dictionaries,
    which construct a sequence of `FramePoint` instances. Like a partition, the first
    and last points must be identical.

    :param json_filename: Path to a JSON file mapping
    :return: Dictionary mapping location ID --> frame partition
    """
    logging.getLogger().info("Building partition mapping from JSON file")
    json_path = Path(json_filename)
    partition_raw_map: dict[LocationID, list[dict[str, int]]]
    partition_map: dict[LocationID, FramePartition]
    with json_path.open(encoding="utf-8") as file_pointer:
        partition_raw_map = json.load(file_pointer)
    partition_map = {
        partition_name: build_partition_from_sequence_dict(partition_raw_mapping)
        for partition_name, partition_raw_mapping in partition_raw_map.items()
    }
    return partition_map


def build_partition_from_sequence_dict(sequence_dict: Sequence[dict]) -> FramePartition:
    """A class method to construct a partition using a sequence of dicts
    which construct a sequence of `FramePoint` instances. Like a partition, the first
    and last points must be identical.

    :param sequence_dict: A sequence of dicts with signature
    :return: A partition in the same order of the input sequence
    """
    input_points: list[Optional[FramePoint]] = [
        None,
    ] * len(sequence_dict)
    point_dict: dict[str, int]
    for index, point_dict in enumerate(sequence_dict):
        test_point = FramePoint(**point_dict)
        input_points[index] = test_point
    return FramePartition(input_points)
