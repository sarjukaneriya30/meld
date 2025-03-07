"""Importing log files"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Union

from intrusiondet.core.types import AttributeKey, PathLike

SPREADSHEET: Final[tuple[str, ...]] = (
    "XLS",
    "XLSX",
    "XLSM",
    "XLSB",
    "ODF",
    "ODS",
    "ODT",
)
"""Spreadsheet/worksheet file formats"""


def get_log_format(json_file: PathLike) -> dict[AttributeKey, Union[str, dict, list]]:
    """Read a JSON file that expands upon the format of a log file

    :param json_file: JSON file path that describes the file schema
    :return: A dictionary mapping how a log is structured
    """
    raw_config: dict
    columns_config: dict[str, Any]

    cfg_file = Path(json_file)
    with cfg_file.open(encoding="utf-8") as file_pointer:
        raw_config = json.load(file_pointer)
    raw_columns_dict: dict = raw_config["COLUMNS"]
    columns_config = dict.fromkeys(raw_columns_dict.keys())
    key: str
    value: str
    for key, value in raw_columns_dict.items():
        key = key.upper()
        value = value.lower()
        if "int" in value:
            columns_config[key] = int
        elif "float" in value:
            columns_config[key] = float
        elif "str" in value:
            columns_config[key] = str
        else:
            raise ValueError(f"Unable to determine type for {key}: {value}")
    assert all(val is not None for val in columns_config.values())
    out_dict = raw_config.copy()
    out_dict.pop("COLUMNS")
    out_dict["COLUMNS"] = columns_config
    return out_dict
