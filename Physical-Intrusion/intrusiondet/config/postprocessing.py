"""Perform post-processing actions on a raw project configuration"""
import functools
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import tomli
from dotenv import dotenv_values

from intrusiondet.config import configstruct
from intrusiondet.core.parser import parse_dotenv_secret_key_list
from intrusiondet.core.types import Struct


def pathify(in_struct: Struct, key: str) -> Optional[Path]:
    """Convert a struct member to a path

    :param in_struct: input struct
    :param key: The member name
    :return: A path if struct member is a populated string
    """
    test_path: Optional[str] = in_struct.get(key)
    if test_path is not None:
        if len(test_path) == 0:
            return None
        ret_path = Path(test_path).absolute()
        return ret_path
    return None


def setify(in_list: Sequence) -> list:
    """Remove duplicates in a list

    :param in_list: Input list or any collection
    :return: A list with duplicates removed
    """
    return list(set(in_list))


def log_info_section(_func: Optional[Callable] = None, **decorator_kwargs) -> Callable:
    """Provides header and footer logging for a configuration section

    :param _func: Function callable
    :param decorator_kwargs: Used key is only "name" which is the configuration section
     name
    """

    def log_decorator_info(func):
        @functools.wraps(func)
        def log_decorator_wrapper(*args, **kwargs):
            logging.info("Post processing the %s section", decorator_kwargs.get("name"))
            value = func(*args, **kwargs)
            logging.info(
                "Done post processing the %s section", decorator_kwargs.get("name")
            )
            return value

        return log_decorator_wrapper

    if _func is None:
        return log_decorator_info
    return log_decorator_info(_func)


def log_parameter(parameter_name: str, parameter_value: Any) -> None:
    """Log a post-processed log parameter with unique types like pathlib.Path

    :param parameter_name: Name of a configuration parameter
    :param parameter_value:  Value of the parameter
    """
    logging.info("Got parameter %s as %s", parameter_name, parameter_value)
    if isinstance(parameter_value, Path):
        logging.info("Parameter is a path: exists =? %s", parameter_value.exists())


@log_info_section(name="config.output")
def post_process_config_output_section(output_section: Struct) -> None:
    """Run post-processing on the config.output section

    :param output_section: config.output section
    """
    for path_key in ("json_dump_path",):
        output_section_a_path_parameter = pathify(output_section, path_key)
        log_parameter(f"output.{path_key}", output_section_a_path_parameter)
        output_section[path_key] = output_section_a_path_parameter


@log_info_section(name="config.frontend")
def post_process_config_frontend_section(frontend_section: Struct) -> None:
    """Run post-processing on the config.frontend section

    :param frontend_section: config.frontend section
    """
    assert frontend_section is not None


@log_info_section(name="config.model")
def post_process_config_model_section(model_section: Struct) -> None:
    """Run post-processing on the config.model section

    :param model_section: config.model section
    """
    for path_key in (
        "path",
        "supernames_path",
    ):
        model_section_a_path_parameter = pathify(model_section, path_key)
        log_parameter(f"model.{path_key}", model_section_a_path_parameter)
        model_section[path_key] = model_section_a_path_parameter


@log_info_section(name="config.proc")
def post_process_config_proc_section(proc_section: Struct) -> None:
    """Run post-processing on the DNN post-processing config.proc section

    :param proc_section: config.proc section
    """
    assert proc_section is not None


@log_info_section(name="config.detobj")
def post_process_config_detobj_section(detobj_section: Struct) -> None:
    """Run post-processing on the config.detobj section

    :param detobj_section: config.detobj section
    """
    sub_struct: Struct
    for list_sub_struct_name in (
        "ignore",
        "anomaly",
    ):
        sub_struct = detobj_section.get(list_sub_struct_name)
        assert sub_struct is not None
        for path_key in ("list_path",):
            detobj_subsection_a_path_parameter = pathify(sub_struct, path_key)
            log_parameter(
                f"detobj.{list_sub_struct_name}.{path_key}",
                detobj_subsection_a_path_parameter,
            )
            sub_struct[path_key] = detobj_subsection_a_path_parameter
            if detobj_subsection_a_path_parameter is not None:
                try:
                    with detobj_subsection_a_path_parameter.open("rb") as file_pointer:
                        list_config: dict = tomli.load(file_pointer)
                        sub_struct.list.extend(list_config.get("list"))
                        sub_struct.list = setify(sub_struct.list)
                except PermissionError:
                    print(f"Not loading extension for {list_sub_struct_name}")

    for dict_sub_struct_name in ("colors",):
        sub_struct = detobj_section.get(dict_sub_struct_name)
        assert sub_struct is not None
        for path_key in ("path",):
            dict_path = pathify(sub_struct, path_key)
            log_parameter(f"detobj.{dict_sub_struct_name}.{path_key}", dict_path)
            sub_struct[path_key] = dict_path
            if dict_path is not None:
                try:
                    with dict_path.open("rb") as file_pointer:
                        dict_config: dict = tomli.load(file_pointer)
                        sub_struct.update(dict_config)
                except PermissionError:
                    print(f"Not loading extension for {dict_sub_struct_name}")


@log_info_section(name="config.remote")
def post_process_remote_section(
    remote_section: configstruct.IntrusionDetectionRemote,
):
    """Run post-processing on the config.remote section

    :param remote_section: config.remote section
    """
    video_subsection: Struct = remote_section.video
    for path_key in (
        "cameras_path",
        "captures_path",
    ):
        video_subsection_a_path_parameter = pathify(video_subsection, path_key)
        log_parameter(f"remote.video.{path_key}", video_subsection_a_path_parameter)
        video_subsection[path_key] = video_subsection_a_path_parameter


@log_info_section(name="config.database")
def post_process_database_section(
    database_section: configstruct.IntrusionDetectionDatabase,
):
    """Run post-processing on the config.database section

    :param database_section: config.database section
    """
    public_subsection: Struct = database_section.public
    for path_key in ("path",):
        public_subsection_a_path_parameter = pathify(public_subsection, path_key)
        log_parameter(f"database.public.{path_key}", public_subsection_a_path_parameter)
        public_subsection[path_key] = public_subsection_a_path_parameter
        public_subsection.data = dotenv_values(public_subsection_a_path_parameter)
        logging.info("Parsed database.public data")

    private_subsection: Struct = database_section.private
    for path_key in ("path",):
        private_subsection_a_path_parameter = pathify(private_subsection, path_key)
        log_parameter(
            f"database.private.{path_key}", private_subsection_a_path_parameter
        )
        private_subsection[path_key] = private_subsection_a_path_parameter
        private_subsection.data = parse_dotenv_secret_key_list(
            private_subsection_a_path_parameter
        )
        logging.info("Parsed database.private data")


@log_info_section(name="config.video")
def post_process_video_section(
    video_section: configstruct.IntrusionDetectionVideoMetadata,
) -> None:
    """Run post-processing on the config.video metadata model section

    :param video_section: config.video section
    """
    assert video_section is not None


def post_process_config(
    config: configstruct.IntrusionDetectionConfig,
) -> configstruct.IntrusionDetectionConfig:
    """Post processes a configuration for special instances like file paths

    :param config: The configuration
    :return: Updated configuration
    """

    # output section
    output_section = config.output
    post_process_config_output_section(output_section)

    # frontend section
    frontend_section = config.frontend
    post_process_config_frontend_section(frontend_section)

    # DNN model section
    model_section = config.model
    post_process_config_model_section(model_section)

    # Model processing "proc" section
    proc_section = config.proc
    post_process_config_proc_section(proc_section)

    # detected objects' section
    detobj_section = config.detobj
    post_process_config_detobj_section(detobj_section)

    # remote section
    remote_section = config.remote
    post_process_remote_section(remote_section)

    # database section
    database_section = config.database
    post_process_database_section(database_section)

    # video metadata section
    video_section = config.video
    post_process_video_section(video_section)

    return config
