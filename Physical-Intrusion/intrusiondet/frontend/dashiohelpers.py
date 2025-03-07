"""Helper methods that eliminate the need to set strings in dash callbacks"""
from typing import Any, Dict, Literal, Optional

import matplotlib.font_manager

from intrusiondet.core.types import DashComponentID


def get_all_system_fonts() -> Dict[Literal["afm", "ttf"], list[str]]:
    """Get all system fonts"""
    font_extensions = ["afm", "ttf"]
    system_fonts = {
        ext: matplotlib.font_manager.findSystemFonts(fontext=ext)
        for ext in font_extensions
    }
    return system_fonts


SYSTEM_FONTS: Dict[Literal["afm", "ttf"], list[str]] = get_all_system_fonts()
"""All system fonts as a dictionary mapping format to systme path"""


def build_datatable_column(
    name: str, cid: Optional[DashComponentID] = None
) -> Dict[Literal["name", "id"], Any]:
    """Output a dictionary with the required keys

    :param name: Name to show
    :param cid: Component ID for the name label
    :return: Name, id dictionary
    """
    return {"name": name, "id": cid if cid is not None else name}
