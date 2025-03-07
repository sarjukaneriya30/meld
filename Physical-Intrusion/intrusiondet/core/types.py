"""Store generic type aliases, objects, and new types"""
import logging
import os
from abc import ABC, abstractmethod
from collections import UserDict
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

FrameIndex: TypeAlias = int
"""Type alias for a frame index counter"""

AttributeKey: TypeAlias = str
"""Keyname for dataclass attributes"""

LocationID: TypeAlias = str
"""Location ID string"""

CameraID: TypeAlias = str
"""ID for a camera associated with a video reader"""

ClassID: TypeAlias = int
"""Class ID for computer vision prediction"""

ClassNameID: TypeAlias = str
"""Name for a class ID for computer vision prediction"""

WorkOrderID: TypeAlias = str
"""Work order ID"""

WorkOrderLineNumber: TypeAlias = int
"""Work order line number"""

PathLike = str | Path | os.PathLike
"""A string or Path instance"""

PathLikeOrNone = str | Path | os.PathLike | None
"""A string, Path instance, or None"""

DashPropertyID: TypeAlias = str
""""""

DashComponentID: TypeAlias = str
""""""

Integers = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
"""Numerical non-complex integer types"""

Floats = Union[float, np.float16, np.float32, np.float64]
"""Numerical non-complex floating-point types"""

IntOrFloat = Union[float, int]
"""Built-in integer or floating types"""

TimeStampInSeconds = IntOrFloat
"""Time stamp (POSIX) usually including the microseconds"""

TimeDifferentialInSeconds = IntOrFloat
"""Time differential in seconds including the microseconds"""

DateTimeLike = Union[pd.Timestamp, datetime, str]
"""A date/time-like or Pandas Timestamp object"""

IntTuple = tuple[int, ...]
"""Integer tuple of arbitrary length"""

NDArray = npt.NDArray
"""Numpy typed N-dimensional array"""

NDArrayOrNone = Union[NDArray, None]
"""An optional Numpy typed N-dimensional array"""

NDArrayFloat = NDArray[Floats]
"""Numpy typed N-dimensional floating-point type array"""

NDArrayInt = NDArray[Integers]
"""Numpy typed N-dimensional integer type array"""


Frame = NDArrayInt
"""Type alias for a definite image array"""

FrameOrNone = Union[Frame, None]
"""Type alias for a image array that may not exist"""


class Object(ABC):
    """A generic inheritable for sub-classing"""

    @abstractmethod
    def __str__(self):
        raise NotImplementedError(
            "You must implement the __str__ method for your class"
        )

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError(
            "You must implement the __str__ method for your class"
        )


def immutable_property(storage_name: str) -> property:
    """Immutable property factory. Since all instances have the __dict__ member,
     we can check against it to see if the instance has been assigned a value.
     If so, then raise an exception

    :param storage_name: The name of the property
    :return: An immutable property
    """

    def immut_prop_getter(instance: Any) -> Any:
        """Getter method for the immutable property"""
        return instance.__dict__.get(storage_name)

    def immut_prop_setter(instance: Any, value: Any) -> None:
        """Setter method for the immutable property"""
        if immut_prop_getter(instance) is not None:
            raise ValueError(f"Cannot reinitialize property {storage_name}")
        instance.__dict__[storage_name] = value

    return property(immut_prop_getter, immut_prop_setter)


class NamedObject(Object):
    """A generic inheritable class with a given name"""

    name: property = immutable_property("name")
    """The name of the object"""

    def __init__(self, *args, name: str = "", **kwargs):
        """A generic inheritable with a given name

        :param name: The object name
        :param args: No arguments are required nor checked
        :param kwargs: No keyword arguments are required nor checked
        """
        super().__init__()
        self.name: Final[str] = name

    def __str__(self) -> str:
        """A string of the class and its unique name"""
        return f"{self.__class__}: name: {self.name}"

    def __repr__(self) -> str:
        """A string of the class and its unique name"""
        return self.__str__()


class UInt8(Object):
    """An unsigned integer with 8-bits. The use of this class is to store the
    minimum, maximum value of 0, 255"""

    MIN: Final[int] = 0
    """Minimum acceptable value for 8-bit unsigned-integer"""

    MAX: Final[int] = 255
    """Maximum acceptable value for 8-bit unsigned-integer"""

    def __init__(self, value: Any = 0):
        """An unsigned integer with 8-bits. The use of this class is check inputs are bounded
         by minimum/maximum value of 0/255

        :param value: Scalar value
        :raises ValueError: When unable to convert an input to a numpy.uint8 type
        """
        super().__init__(value)
        if not UInt8.is_a(value):
            logging.getLogger().exception(
                "The input value %s cannot be converted to a unsigned 8-bit integer",
                value,
            )
            raise ValueError
        self.value = value

    def __repr__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def is_a(test_num: Any) -> bool:
        """Tests if an input is uint8-type

        :param test_num: The test number
        :returns: True if the instance is an integer-type between 0 and 255 (inclusive)
        """
        if not isinstance(test_num, Integers):
            return False
        if not UInt8.MIN <= test_num <= UInt8.MAX:
            return False
        return True


class Struct(UserDict):
    """Class to objectify keyword, value pairs to use its keys as first-class variables"""

    def __init__(self, **kwargs):
        """Class to objectify keyword, value pairs to use its keys as first-class variables

        :param kwargs: Any keyword, value pairs where the keyword is a string
        :raises KeyError: When any keyword is not a string

        >>> s = Struct(**{"key": "value", "sky": "blue", "ABC": 123})
        >>> s.key
        'value'
        >>> s.sky
        'blue'
        >>> s.ABC
        123
        """
        super().__init__()
        if not all(isinstance(key, str) for key in kwargs):
            raise KeyError("Only string keys are allowed!")
        self.__dict__.update(**kwargs)
        self.data = self.__dict__


class RecursiveStruct(Struct):
    """Created a nested Struct if the input kwargs have dictionary values also"""

    def __init__(self, **kwargs) -> None:
        """Created a nested Struct if the input kwargs have dictionary values also

        :param kwargs: Any keywords, value pairs as long as the keywords can be stringified

        >>> r = RecursiveStruct(**{"key": {"a": 1, "b": 2}})
        >>> r.key.a
        1
        >>> r.key.b
        2
        """
        super().__init__(**kwargs)
        self.data.update(RecursiveStruct.build_recursively(**kwargs))

    @staticmethod
    def build_recursively(**kwargs) -> Struct:
        """Recursive method to build nested Structs

        :param kwargs: Any keywords, value pairs as long as the keywords can be stringified
        :return: A Struct of Structs

        >>> r = RecursiveStruct.build_recursively(**{"key": {"a": 1, "b": 2}})
        >>> r.key.a
        1
        >>> r.key.b
        2
        """
        master = Struct(**kwargs)
        for key, value in kwargs.items():
            if isinstance(value, dict):
                master.__dict__.update(
                    **{key: RecursiveStruct.build_recursively(**value)}
                )
        return master
