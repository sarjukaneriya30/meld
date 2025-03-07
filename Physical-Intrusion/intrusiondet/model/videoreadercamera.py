"""A video reader camera is a model of how cameras, tethered to a video reader,
have a unique ID, location, and observe zones in the field of view"""
import logging
import os
from collections import UserDict
from functools import cache
from pathlib import Path
from typing import Final, Optional

import attr
from tomli import load

from intrusiondet.core.types import CameraID, LocationID, PathLike
from intrusiondet.model.framing import (
    FramePartition,
    build_partitions_mapping_from_json_file,
)


@attr.attrs(frozen=True)
class VideoReaderCamera:
    """Camera connected to a video reader"""

    camera_id: CameraID = attr.attrib(converter=CameraID, default=CameraID())
    """Unique ID of the camera for that particular video reader"""

    location: LocationID = attr.attrib(converter=LocationID, default=LocationID())
    """Location ID where the camera is placed"""

    zones: dict[LocationID, FramePartition] = attr.attrib(
        factory=dict,
        converter=build_partitions_mapping_from_json_file,
        validator=attr.validators.deep_mapping(
            key_validator=attr.validators.instance_of(LocationID),
            value_validator=attr.validators.instance_of(FramePartition),
            mapping_validator=attr.validators.instance_of(dict),
        ),
    )
    """Zones are areas in the field of view visible to the camera"""

    adjacencies: frozenset[LocationID] = attr.attrib(
        default=frozenset(),
        converter=frozenset,
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(LocationID),
            iterable_validator=attr.validators.instance_of(frozenset),
        ),
    )
    """Set of nearest camera adjacencies"""

    rois = attr.attrib(
        factory=list,
        converter=list,

    )
    """Regions of Interest"""


CAMERAID2VIDEOREADERMAPPING: Final = dict[CameraID, VideoReaderCamera]
"""A mapping of camera ID --> video reader camera"""


CAMERAIDVIDEOREADERTUPLE: Final = tuple[CameraID, VideoReaderCamera]
"""Pairing of camera ID and video reader camera"""


class VideoReaderCameraException(Exception):
    """Generalized exception for anything going wrong with video reader cameras"""


def load_video_reader_camera(
    camera_dir: PathLike, config_name: PathLike = "config.toml"
) -> VideoReaderCamera:
    """Load a camera configuration, located in the specified directory. The
    configuration for the camera is the LOCAL path to a TOML file appended to the camera
    directory.

    :param camera_dir: Directory for camera configuration
    :param config_name: Configuration file
    :return: video reader camera
    """
    logger = logging.getLogger()
    logger.info("Loading video reader camera at %s", os.fspath(camera_dir))
    video_reader_camera_path = Path(camera_dir)
    if not video_reader_camera_path.is_dir():
        logger.exception(
            "There is no video reader camera at the given path: %s",
            os.fspath(video_reader_camera_path),
        )
        raise VideoReaderCameraException
    camera_config: dict[str, str]
    with Path(video_reader_camera_path / config_name).open("rb") as file_pointer:
        camera_config = load(file_pointer)
    camera_config["zones"] = os.fspath(
        Path(video_reader_camera_path / camera_config["zones"])
    )
    ret = VideoReaderCamera(**camera_config)
    return ret


@cache
def get_adjacencies_cached(
    mapping: frozenset[CAMERAIDVIDEOREADERTUPLE], loc_id: LocationID
):
    """Get the adjacencies for a location ID

    :param mapping: Set of camera ID and video reader camera pairs
    :param loc_id: Location ID
    :return: Cameras immediately adjacent to location
    """
    camera: Optional[VideoReaderCamera] = next(
        (item[1] for item in mapping if item[1].location == loc_id), None
    )
    if camera is None:
        return None
    return camera.adjacencies


@cache
def find_cameras_in_path_cached(
    mapping: frozenset[CAMERAIDVIDEOREADERTUPLE],
    start_loc_id: LocationID,
    end_loc_id: LocationID,
) -> frozenset[LocationID]:
    """Find all the areas mapped between two locations

    :param mapping: Set of locations to WarehouseLocationWithCameraAdjacencyFinder class
    :param start_loc_id: Start location
    :param end_loc_id: End location
    :return: Set of locations with cameras in-between the two
    """
    start_loc_adj = get_adjacencies_cached(mapping, start_loc_id)
    start_found_adjacencies = set()
    if start_loc_adj is not None:
        for start_loc_adj_id in start_loc_adj:
            start_adjacencies = get_adjacencies_cached(mapping, start_loc_adj_id)
            if start_adjacencies is not None:
                start_found_adjacencies.update(start_adjacencies)
    end_loc_adj = get_adjacencies_cached(mapping, end_loc_id)
    end_found_adjacencies = set()
    if end_loc_adj is not None:
        for end_loc_adj_id in end_loc_adj:
            end_adjacencies = get_adjacencies_cached(mapping, end_loc_adj_id)
            if end_adjacencies is not None:
                end_found_adjacencies.update(end_adjacencies)
    return frozenset(start_found_adjacencies.intersection(end_found_adjacencies))


class WarehouseLocationCameraAdjacencyFinder(UserDict[CameraID, VideoReaderCamera]):
    """Finds camera adjacencies using location IDs."""

    def __init__(self, camera_id_mapping: CAMERAID2VIDEOREADERMAPPING) -> None:
        """Finds camera adjacencies using location IDs.

        :param camera_id_mapping: Mapping from camera ID --> video reader camera
        """
        self._logger = logging.getLogger()
        super().__init__(camera_id_mapping)
        self.data: CAMERAID2VIDEOREADERMAPPING

    @classmethod
    def build_from_directory_path(cls, directory: PathLike):
        """Given a directory where each child directory is a video reader camera ID and
        builder files (config.toml and zones.json), build a mapping of all the cameras

        :param directory: Full path to camera setup directory
        :return:
        """
        cameras_path = Path(directory)
        directories = [
            os.fspath(item) for item in cameras_path.iterdir() if item.is_dir()
        ]
        builder: CAMERAID2VIDEOREADERMAPPING = {}
        for direct in directories:
            video_reader_camera = load_video_reader_camera(direct)
            builder[video_reader_camera.camera_id] = video_reader_camera
        return cls(builder)

    def __hash__(self):
        return hash(frozenset(self.items()))

    def get_adjacencies(self, loc_id: LocationID) -> Optional[frozenset[LocationID]]:
        """Get the adjacencies for a location ID

        :param loc_id: Location ID
        :return: Cameras immediately adjacent to location
        """
        return get_adjacencies_cached(frozenset(self.data.items()), loc_id)

    def find_cameras_in_path(
        self, start_loc_id: LocationID, end_loc_id: LocationID
    ) -> frozenset[LocationID]:
        """Find all the areas mapped between two locations

        :param start_loc_id: Start location
        :param end_loc_id: End location
        :return: Set of locations with cameras in-between the two
        """
        return find_cameras_in_path_cached(
            frozenset(self.data.items()), start_loc_id, end_loc_id
        )
