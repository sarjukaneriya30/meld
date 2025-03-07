"""Make strings safe for databases and serialization"""
from typing import Any, Optional

import numpy as np


def sanitize_string(item: Any, allowed_chars: Optional[list[str]] = None) -> str:
    """Sanitize a string. Primarily to remove whitespace and replace in-phrase spaces
    with a dash

    :param allowed_chars:
    :param item: Any string
    :return: String without offending special characters
    """
    if item is None:
        return ""
    if not isinstance(item, str):
        if isinstance(item, float):
            if np.isnan(item) or np.isinf(item) or np.isneginf(item):
                return ""
            item = str(item)
    if allowed_chars is None:
        allowed_chars = ["-", "_"]
    item = item.strip()
    if not item.isalnum():
        item = "".join(
            [char for char in item if (char.isalnum() or char in allowed_chars)]
        )
    item = item.replace(" ", "-")
    return item
