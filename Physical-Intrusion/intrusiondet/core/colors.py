"""Enumerating colors for the application"""
import logging
from typing import Any, Final

import attr

from intrusiondet.core.types import UInt8


def color_validator(inst, attribute: attr.Attribute, value: Any):
    """Validator color attributes are unsigned 8-bit integers"""
    if not UInt8.is_a(value):
        logging.getLogger().exception(
            "The %s color with value %s is not a uint8 number", inst, value
        )
        raise TypeError
    if not (attribute is None or isinstance(attribute, attr.Attribute)):
        logging.getLogger().exception(
            "The %s color attribute %s is not supported",
            inst,
            attribute,
        )
        raise TypeError


@attr.attrs(frozen=True)
class Color:
    """A 3 channel color container to handle RGB normative input and BGR OpenCV input"""

    red: int = attr.attrib(eq=True, validator=color_validator)
    """Red color"""

    green: int = attr.attrib(eq=True, validator=color_validator)
    """Green color"""

    blue: int = attr.attrib(eq=True, validator=color_validator)
    """Blue color"""

    def as_rgb(self) -> tuple[int, int, int]:
        """Get the color as RGB 3-tuple

        :return: red, green, blue

        >>> Color(1, 2, 3).as_rgb()
        (1, 2, 3)
        """
        return self.red, self.green, self.blue

    def as_bgr(self) -> tuple[int, int, int]:
        """Get the color as BGR 3-tuple

        :return: blue, green, red

        >>> Color(1, 2, 3).as_bgr()
        (3, 2, 1)
        """
        return self.blue, self.green, self.red

    def rbgstr(self) -> str:
        """Get the colors as a RGB string

        :return: "rgb(red, green, blue)"
        """
        return f"rgb({self.red}, {self.green}, {self.blue})"


KPMG_BLUE: Final[Color] = Color(0, 51, 141)
LIGHT_BLUE: Final[Color] = Color(0, 145, 218)
PURPLE: Final[Color] = Color(109, 32, 119)
MEDIUM_BLUE: Final[Color] = Color(0, 94, 184)
GREEN: Final[Color] = Color(0, 163, 161)
YELLOW: Final[Color] = Color(234, 170, 0)
LIGHT_GREEN: Final[Color] = Color(67, 176, 42)
PINK: Final[Color] = Color(198, 0, 126)
DARK_BROWN: Final[Color] = Color(117, 63, 25)
LIGHT_BROWN: Final[Color] = Color(155, 100, 45)
OLIVE: Final[Color] = Color(157, 147, 117)
BEIGE: Final[Color] = Color(227, 188, 159)
LIGHT_PINK: Final[Color] = Color(227, 104, 119)
DARK_GREEN: Final[Color] = Color(69, 117, 37)
ORANGE: Final[Color] = Color(222, 134, 38)
BRIGHT_GREEN: Final[Color] = Color(162, 234, 66)
LIGHT_YELLOW: Final[Color] = Color(255, 218, 90)
ROSE: Final[Color] = Color(226, 160, 197)
RED: Final[Color] = Color(188, 32, 75)
TAN: Final[Color] = Color(178, 162, 68)
BLACK: Final[Color] = Color(0, 0, 0)
WHITE: Final[Color] = Color(255, 255, 255)
NEUTRAL_GREY: Final[Color] = Color(128, 128, 128)

KPMG_COLOR_PALETTE: Final[dict[str, Color]] = {
    "KPMG_BLUE": KPMG_BLUE,
    "LIGHT_BLUE": LIGHT_BLUE,
    "PURPLE": PURPLE,
    "MEDIUM_BLUE": MEDIUM_BLUE,
    "GREEN": GREEN,
    "YELLOW": YELLOW,
    "LIGHT_GREEN": LIGHT_GREEN,
    "PINK": PINK,
    "DARK_BROWN": DARK_BROWN,
    "LIGHT_BROWN": LIGHT_BROWN,
    "OLIVE": OLIVE,
    "BEIGE": BEIGE,
    "LIGHT_PINK": LIGHT_PINK,
    "DARK_GREEN": DARK_GREEN,
    "ORANGE": ORANGE,
    "BRIGHT_GREEN": BRIGHT_GREEN,
    "LIGHT_YELLOW": LIGHT_YELLOW,
    "ROSE": ROSE,
    "RED": RED,
    "TAN": TAN,
    "BLACK": BLACK,
    "WHITE": WHITE,
    "NEUTRAL_GREY": NEUTRAL_GREY,
}
"""Enumerated colors defined by KPMG corporate"""
