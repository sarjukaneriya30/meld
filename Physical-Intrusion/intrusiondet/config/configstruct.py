"""Module to import the project configurations"""
import logging
import os
from pathlib import Path
from typing import Any

import tomli

from intrusiondet.core.types import (
    AttributeKey,
    PathLike,
    PathLikeOrNone,
    RecursiveStruct,
    Struct,
)


class ConfigStruct(RecursiveStruct):
    """Load a configuration file with nested Structs for each section.
    Currently, supports TOML only"""

    def __init__(self, config: PathLike):
        """Load a configuration file with nested Structs for each section.
         Currently, supports TOML only

        :param config: The file path to a configuration file
        """
        self._logger = logging.getLogger()
        config_path = Path(config).absolute()
        if config_path.suffix.upper() == ".TOML":
            self._logger.info("Loading configuration %s", os.fspath(config_path))
            with config_path.open("rb") as file_pointer:
                toml_config: dict = tomli.load(file_pointer)
            super().__init__(**toml_config)
        else:
            self._logger.exception(
                "Unable to parse file %s",
                os.fspath(config_path),
            )
            raise NotImplementedError


class IntrusionDetectionFileListWithOverride(Struct):
    """Configuration which indicates a file contains a list of strings separated by new
    line characters and can be overridden
    """

    def __init__(self, **kwargs):
        self.list_path: PathLikeOrNone = ""
        self.list: list[str] = []
        super().__init__(**kwargs)


class IntrusionDetectionDotEnvWithData(Struct):
    """Configuration which indicates a dotenv file contains data. The path points to the
    file and the data is the parsed data
    """

    def __init__(self, **kwargs):
        self.path: PathLikeOrNone = ""
        self.data: dict[AttributeKey, Any] = {}
        super().__init__(**kwargs)


class IntrusionDetectionOutput(Struct):
    """Output dump file for computer vision"""

    def __init__(self, **kwargs):
        self.json_dump_path: PathLike = ""
        super().__init__(**kwargs)


class IntrusionDetectionFrontend(Struct):
    """Frontend configuration"""

    def __init__(self, **kwargs):
        self.enable: bool = False
        self.window_name: str = ""
        super().__init__(**kwargs)


class IntrusionDetectionModel(Struct):
    """Computer vision object detection configuration, including which model to use
    and its parameters"""

    def __init__(self, **kwargs):
        self.type: str = ""
        self.path: PathLike = ""
        self.target: int = -1
        self.backend: int = -1
        self.confidence: float = 0.0
        self.nms: float = 0.0
        self.supernames_path: PathLike = ""
        super().__init__(**kwargs)


class IntrusionDetectionProcessing(Struct):
    """Computer vision frame processing configuration like number of frames to process
    simultaneously also called "batch size"
    """

    def __init__(self, **kwargs):
        self.batch_size: int = 1
        self.processes: int = 0
        self.full_playback: bool = False
        self.sleep_time_sec: float = 0
        self.skip_every_sec: float = 0.0
        self.start_time_sec: float = 0.0
        self.queue_maxsize: int = -1
        self.image_resize: float = 1.0
        super().__init__(**kwargs)


class IntrusionDetectionDetectedObjects(Struct):
    """Computer vision detected objects naming and conventions for this project"""

    class IntrusionDetectionDetectedObjectsIgnore(
        IntrusionDetectionFileListWithOverride
    ):
        """Override which prediction classes to ignore"""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class IntrusionDetectionDetectedObjectsAnomaly(
        IntrusionDetectionFileListWithOverride
    ):
        """Specify which predictions are globally anomalies, like weapons"""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class IntrusionDetectionDetectedObjectsColors(Struct):
        """Specify the frontend colors to use for frame boxes"""

        def __init__(self, **kwargs):
            self.path: PathLike = ""
            super().__init__(**kwargs)

    def __init__(self, **kwargs):
        self.ignore = (
            IntrusionDetectionDetectedObjects.IntrusionDetectionDetectedObjectsIgnore()
        )
        self.anomaly = (
            IntrusionDetectionDetectedObjects.IntrusionDetectionDetectedObjectsAnomaly()
        )
        self.colors = (
            IntrusionDetectionDetectedObjects.IntrusionDetectionDetectedObjectsColors()
        )
        super().__init__(**kwargs)


class IntrusionDetectionRemote(Struct):
    """Remote sources to query for data"""

    class IntrusionDetectionRemoteVideo(Struct):
        """Remote source for videos collected in the warehouse"""

        def __init__(self, **kwargs):
            self.cameras_path: PathLike = ""
            self.captures_path: PathLike = ""
            super().__init__(**kwargs)

    def __init__(self, **kwargs):
        self.video: IntrusionDetectionRemote.IntrusionDetectionRemoteVideo = (
            IntrusionDetectionRemote.IntrusionDetectionRemoteVideo()
        )
        super().__init__(**kwargs)


class IntrusionDetectionDatabase(Struct):
    """Record database connection information"""

    class IntrusionDetectionDatabasePublic(IntrusionDetectionDotEnvWithData):
        """Public connection data for the record database"""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class IntrusionDetectionDatabasePrivate(IntrusionDetectionDotEnvWithData):
        """Private connection data for the record database"""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class IntrusionDetectionDatabaseProperties(Struct):
        """Properties for record storage"""

        def __init__(self, **kwargs):
            self.detections_max_rows: int = 25
            super().__init__(**kwargs)

    def __init__(self, **kwargs):
        self.public: IntrusionDetectionDatabase.IntrusionDetectionDatabasePublic = (
            IntrusionDetectionDatabase.IntrusionDetectionDatabasePublic()
        )
        self.private: IntrusionDetectionDatabase.IntrusionDetectionDatabasePrivate = (
            IntrusionDetectionDatabase.IntrusionDetectionDatabasePrivate()
        )
        self.prop: IntrusionDetectionDatabase.IntrusionDetectionDatabaseProperties = (
            IntrusionDetectionDatabase.IntrusionDetectionDatabaseProperties()
        )
        super().__init__(**kwargs)


class IntrusionDetectionVideoMetadata(Struct):
    """Parse video filename for metadata model"""

    def __init__(self, **kwargs):
        self.schema: list[str] = []
        self.pattern: str | list[str] = []
        self.regex: bool = False
        super().__init__(**kwargs)


class IntrusionDetectionConfig(ConfigStruct):
    """Load a project configuration file with nested Structs for each section.
    Currently, supports TOML only."""

    def __init__(self, config: str):
        """Load a project configuration file with nested Structs for each section.
        Currently, supports TOML only.

        :param config: The file path to a configuration file
        """
        logging.getLogger().info("Building Intrusion Detection configuration class")
        self.output: IntrusionDetectionOutput = IntrusionDetectionOutput()
        self.frontend: IntrusionDetectionFrontend = IntrusionDetectionFrontend()
        self.model: IntrusionDetectionModel = IntrusionDetectionModel()
        self.proc: IntrusionDetectionProcessing = IntrusionDetectionProcessing()
        self.detobj: IntrusionDetectionDetectedObjects = (
            IntrusionDetectionDetectedObjects()
        )
        self.remote: IntrusionDetectionRemote = IntrusionDetectionRemote()
        self.database: IntrusionDetectionDatabase = IntrusionDetectionDatabase()
        self.video: IntrusionDetectionVideoMetadata = IntrusionDetectionVideoMetadata()
        super().__init__(config)
