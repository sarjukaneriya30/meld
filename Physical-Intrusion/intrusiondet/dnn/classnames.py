"""Module that converts between trained DNN class names and remapped classes"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import attrs

from intrusiondet.core.types import ClassNameID, PathLike


def get_class_names(
    class_names: Union[PathLike, list[str]], sep: str = "\n"
) -> list[ClassNameID]:
    r"""Get a list of class names enumerated in a file

    :param class_names: The path to the names file. If the input is a list already,
        return itself
    :param sep: The names separator in the file (default="\n")
    :return: The names given in the file
    """
    if isinstance(class_names, list):
        return class_names
    with Path(class_names).open(
        mode="r", encoding="utf-8"
    ) as fp:  # pylint: disable=invalid-name
        return fp.read().strip().split(sep)


def get_old_2_new_mapping(
    new_2_old_mapping: Union[Path, dict[str, list[str]]],
    old_names: Optional[list[str]] = None,
    new_names: Optional[list[str]] = None,
    checks: Optional[list[str]] = None,
) -> dict[str, str]:
    r"""Reads a JSON file mapping: new --> old (one-to-many) names and returns an
    inverse mapping for old --> new. The method is over-parameterized in order to
    validate between mappings.

    :param new_2_old_mapping: A JSON file mapping new-to-old names.
    :param old_names: Optional old names list for debugging and consistency checks.
    :param new_names: Optional new names list for debugging and consistency checks.
    :param checks: A list function attributes accepting "injective", "surjective" and
        "bijective"
    :return: A dictionary mapping an old name string to a new name string
    :raises ValueError: When a check fails an assertion as defined by the checks
        argument
    """
    if isinstance(new_2_old_mapping, Path):
        with new_2_old_mapping.open(mode="r") as fp:  # pylint: disable=invalid-name
            new_2_old_mapping = json.load(fp)
    if checks is None:
        checks = []
    old_2_new_mapping: dict[str, str]
    old_2_new_mapping = dict.fromkeys(old_names, "") if old_names else {}
    for new_name, old_names_list in new_2_old_mapping.items():
        new_name: str
        old_names_list: list[str]
        for old_name_str in old_names_list:
            if old_names:
                assert old_name_str in old_names
            old_2_new_mapping[old_name_str] = new_name
    if new_names:
        if set(new_names) != set(new_2_old_mapping.keys()):
            set_new_names = set(new_names)
            set_new_2_old_mapping_keys = set(new_2_old_mapping.keys())
            logging.getLogger().exception(
                "The mapping between old to new names is missing: %s",
                set_new_names.symmetric_difference(set_new_2_old_mapping_keys),
            )
            raise ValueError
    if ("injective" in checks or "bijective" in checks) and new_names:
        if len(set(old_2_new_mapping.keys())) != len(set(old_2_new_mapping.values())):
            logging.getLogger().exception(
                "The mapping failed the one-to-one/injective check"
            )
            raise ValueError
    if ("surjective" in checks or "bijective" in checks) and new_names:
        if set(old_2_new_mapping.values()) != set(new_names):
            logging.getLogger().exception(
                "The mapping failed the onto/surjective check"
            )
            raise ValueError
    return old_2_new_mapping


@attrs.define
class ClassNameConverter:  # pylint: disable=too-few-public-methods
    """A class that holds the indexed mapping between old <--> new names"""

    old: list[str] = attrs.field(
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(list),
            member_validator=attrs.validators.instance_of(str),
        )
    )
    """List of names to be converted (the domain)"""

    new: list[str] = attrs.field(
        validator=attrs.validators.deep_iterable(
            iterable_validator=attrs.validators.instance_of(list),
            member_validator=attrs.validators.instance_of(str),
        )
    )
    """List of names after conversion (the codomain)"""

    old_2_new: dict[str, str] = attrs.field(
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.instance_of(str),
            value_validator=attrs.validators.instance_of(str),
            mapping_validator=attrs.validators.instance_of(dict),
        )
    )
    """Mapping between old --> new names"""


def class_converter_from_supernames_directory(
    old_names_filename: str, new_names_directory: str
) -> ClassNameConverter:
    """A class method for `ClassNameConverter`. The name of the old names file and new
    names' directory are expected as identical

    :param old_names_filename: The path to the old news files
    :param new_names_directory: The path to the directory with the new names.
    :return: Class name converter
    """
    old_names = get_class_names(old_names_filename)
    supernames = Path(new_names_directory)
    new_names = get_class_names(supernames / "super.names")
    new_2_old_mapping = get_old_2_new_mapping(supernames / "mapping.json")
    class_converter = ClassNameConverter(
        old=old_names,
        new=new_names,
        old_2_new=new_2_old_mapping,
    )
    return class_converter


def convert_class_id(old_class_id: int, converter: ClassNameConverter) -> Optional[int]:
    r"""Convert a class name using the ID

    :param old_class_id: Input class ID
    :param converter: An instance of the ClassNameConverter class
    :return: Integer >= 0 if there is a mapping, None otherwise
    """
    old_class_name = converter.old[old_class_id]
    if old_class_name in converter.old_2_new.keys():
        new_class_name: Optional[str] = converter.old_2_new.get(old_class_name)
        if new_class_name:
            return converter.new.index(new_class_name)
    return None
