"""Methods to initialize project configuration"""
import logging
import os
from pathlib import Path
from typing import Optional

import attrs
import dotenv

from intrusiondet.config import configstruct, postprocessing
from intrusiondet.core.logging import build_logger, listener_configurer
from intrusiondet.core.types import PathLike
from intrusiondet.dnn.yolo import YoloModel
from intrusiondet.model import videoreadercamera
from intrusiondet.orm.sqlalchemyrepository import SQLAlchemyRepository
from intrusiondet.orm.sqlalchemytables import METADATA, start_mappers

CONFIG: configstruct.IntrusionDetectionConfig


def get_home(dotenv_file: Optional[PathLike] = None):
    """Get the path where the 'intrusiondet' package path resides

    :return: A path string
    """
    if dotenv_file is not None:
        dotenv.load_dotenv(os.fspath(dotenv_file))
    home_path = Path(
        os.environ.get(
            "INTRUSIONDET_PATH", os.fspath(Path(Path(__file__).parent / "..").resolve())
        )
    )
    return os.fspath(home_path)


@attrs.define(kw_only=True)
class IntrusionDetectionBootstrapper:
    """A bootstrapper for initializing the project"""

    home: str = attrs.field(converter=os.fspath)
    """Path where the 'intrusiondet' package path resides"""

    config: configstruct.IntrusionDetectionConfig = attrs.field()
    """Configuration specification for the runtime"""

    logger: Optional[logging.Logger] = attrs.field(
        default=None,
        # validator=attrs.validators.instance_of(logging.Logger)
    )
    """Standard library Logging instance"""

    cameras: videoreadercamera.WarehouseLocationCameraAdjacencyFinder = attrs.field(
        validator=attrs.validators.instance_of(
            videoreadercamera.WarehouseLocationCameraAdjacencyFinder
        )
    )
    """Mapping of cameras IDs connected to a video reader to their location"""


def build_dnn_yolo_model(
    config: configstruct.IntrusionDetectionConfig, name: Optional[str] = None
) -> YoloModel:
    """Build a YOLO DNN model using the runtime specifications

    :param config: Configuration
    :param name: Unique name to give for the instance
    :return: A YOLO DNN model
    """
    if name is None:
        name = config.model.type
    yolo_model = YoloModel(
        name=name,
        yolo_dir=config.model.path,
        backend=config.model.backend,
        target=config.model.target,
    )
    return yolo_model


def load_project_config(path: str) -> configstruct.IntrusionDetectionConfig:
    """Load project runtime configuration

    :param path: Path to the configuration file
    :return: Populated configuration
    """
    global CONFIG
    CONFIG = configstruct.IntrusionDetectionConfig(path)
    logging.info("Loaded raw project configuration as %s", str(CONFIG))
    postprocessing.post_process_config(CONFIG)
    return CONFIG


def create_sql_repository_connection(
    config: configstruct.IntrusionDetectionConfig,
) -> SQLAlchemyRepository:
    """Create a SQL session connection using SQLAlchemy

    :param config: Project configuration
    :return: Data repository connection
    """
    database_config = config.database
    logger = logging.getLogger()
    logger.info(
        "Connecting to data repository with public data %s and private data %s",
        str(database_config.public),
        str(database_config.private),
    )
    repo_host = database_config.public.data["HOST"]
    data_repo = SQLAlchemyRepository.create_session(
        METADATA,
        start_mappers,
        driver=database_config.public.data["DRIVER"],
        host=os.fspath(Path(repo_host).absolute()),
        port=database_config.public.data["PORT"],
        uname=database_config.private.data["username"],
        passwd=database_config.private.data["password"],
        name=database_config.public.data["NAME"],
    )
    return data_repo


def bootstrap(
    home_path: Optional[str] = None,
    config_path: Optional[str] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    logger_name: Optional[str] = None,
    logger_to_console: Optional[bool] = None,
) -> IntrusionDetectionBootstrapper:
    """Bootstrap the runtime setup

    :param home_path: Path where the 'intrusiondet' package path resides
    :param config_path: Path where the config.toml configuration
    :param log_level: Logging level string
    :param log_format: Logging format string
    :param logger_name:
    :param logger_to_console:
    :return: A bootstrapper with runtime properties
    """
    root_logger = None
    if logger_to_console is True:
        root_logger = build_logger(log_level, log_format, logger_name)
    elif logger_name is not None:
        listener_configurer(logger_name)
        root_logger = logging.getLogger()
    if home_path is None:
        home_path = Path(get_home())
    print(f"Setting home path and CWD as {os.fspath(home_path)}")
    os.chdir(os.fspath(Path(home_path)))
    if config_path is None:
        # Try the default
        config_path = os.fspath(home_path / "configs/config.toml")
    print(f"Setting configuration file as {os.fspath(config_path)}")
    config = load_project_config(os.fspath(config_path))
    cameras = videoreadercamera.WarehouseLocationCameraAdjacencyFinder.build_from_directory_path(
        os.fspath(config.remote.video.cameras_path)
    )

    bootstrapper = IntrusionDetectionBootstrapper(
        home=home_path, config=config, logger=root_logger, cameras=cameras
    )
    return bootstrapper
