"""Methods to convert inputs"""
import logging
from copy import deepcopy
from datetime import datetime as stdlib_datetime
from datetime import timezone as stdlib_timezone
from typing import Final, Optional

from intrusiondet.core.types import DateTimeLike, TimeStampInSeconds

DEFAULT_DATETIME_STR: Final[str] = "1970-01-01T00:00:00+00:00"
"""Default and earliest datetime input value for a datetime ISO string"""

EARLIEST_DATETIME: Final[stdlib_datetime] = stdlib_datetime.fromisoformat(
    DEFAULT_DATETIME_STR
)
"""Earliest datetime value for datetime evaluation"""


def datetimeify_no_earlier_than(
    test_datetime: stdlib_datetime,
    earliest_datetime: stdlib_datetime = EARLIEST_DATETIME,
) -> stdlib_datetime:
    """Ensure that a datetime instance is no earlier than a specific datetime.

    :param test_datetime: Datetime in question
    :param earliest_datetime: Set earliest datetime
    :return: Datetime no earlier than earliest
    """
    return max(test_datetime, earliest_datetime)
    # if test_datetime < earliest_datetime:
    #     return earliest_datetime
    # return test_datetime


def datetimeify(
    item: Optional[DateTimeLike | TimeStampInSeconds],
) -> Optional[stdlib_datetime]:
    """Convert a datetime-like object or None into a datetime instance

    :param item: Datetime-like instance
    :return: Datetime
    """
    if isinstance(item, stdlib_datetime):
        pass
    elif item is None or item == "":
        item = deepcopy(EARLIEST_DATETIME)
    elif isinstance(item, str):
        item = stdlib_datetime.fromisoformat(item)
    elif isinstance(item, TimeStampInSeconds):
        item = stdlib_datetime.fromtimestamp(item, tz=stdlib_timezone.utc)
    elif hasattr(item, "to_pydatetime"):
        item = item.to_pydatetime()
    else:
        logging.getLogger().error("Unable to datetimeify item %s", item)
        return None
    if item.tzinfo is None:
        item = item.replace(tzinfo=stdlib_timezone.utc)
    return datetimeify_no_earlier_than(item, deepcopy(EARLIEST_DATETIME))
