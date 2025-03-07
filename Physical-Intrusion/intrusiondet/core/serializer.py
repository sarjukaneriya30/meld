"""Enabling serialization"""

import datetime
from json import JSONEncoder
from typing import Any, Final

SERIALIZED_TYPES: Final[tuple[type, ...]] = (datetime.datetime, frozenset, set)
"""Not natively serializable types which has custom serializable support"""


class SerializeError(Exception):
    """Any object not listed by the SERIALIZED_TYPES variable
    which are also not natively serializable"""


def datetime_serialize(obj: datetime.datetime) -> float:
    """Serialize a datetime instance as a floating-point number POSIX timestamp"""
    return obj.timestamp()


def serialize(obj: Any) -> str | list | float:
    """Serialize predefined objects not natively serializable, else try to turn them
    into strings"""
    if isinstance(obj, datetime.datetime):
        return datetime_serialize(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def serialize_with_exception(obj: Any) -> str:
    """Serialize predefined objects not natively serializable,
    but raise an exception when the instance predefined"""
    if not isinstance(obj, SERIALIZED_TYPES):
        raise SerializeError
    return serialize(obj)


class DateTimeJSONEncoder(JSONEncoder):
    """Allow Datetime objects to be serialized"""

    def default(self, o):
        """Super method for JSONEncoder.default"""
        try:
            return serialize_with_exception(o)
        except SerializeError:
            return super().default(o)
