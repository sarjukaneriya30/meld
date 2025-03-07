"""Warehouse work orders and associated order lines are filtered using the classes and
methods in this module.

An OrderLine is a single work order line in a sequence of order lines for an aggregate
work order. This is the smallest unit that should be utilized for classification.
Aggregating work order lines into larger units are handled in separate classes like so.

 * WarehouseWorkLog -- list by ID and list by datetime --> WorkOrderList
 * WorkOrderList -- list of --> WorkOrder
 * WorkOrder -- containing --> OrderLineList
 * OrderLineList -- list of --> OrderLine

where the functional model piece OrderLine contains the work location ID, work type, and
creation datetime.
"""
from __future__ import annotations

import json
import logging
import os
from bisect import bisect_left, insort_left
from collections import UserList
from datetime import datetime as stdlib_datetime
from datetime import timedelta as stdlib_timedelta
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Optional, Sequence, Union

import attr
import pandas as pd
from dateutil import tz
from pyodbc import connect as odbc_connect

from intrusiondet.core import serializer
from intrusiondet.core.converters import DEFAULT_DATETIME_STR, datetimeify
from intrusiondet.core.sanitizer import sanitize_string
from intrusiondet.core.types import (
    AttributeKey,
    LocationID,
    PathLike,
    WorkOrderID,
    WorkOrderLineNumber,
)
from intrusiondet.model.logfeed import SPREADSHEET, get_log_format

WORK_ORDER_LINE_NUM_START: Final[WorkOrderLineNumber] = 1
"""First line number in all work orders"""

WORK_ORDER_LINE_STR_DEFAULT: Final[str] = "UNKNOWN"
"""Default string argument for any order line item not set"""

WORK_ORDER_LINE_INT_DEFAULT: Final[WorkOrderLineNumber] = -1
""""""


class OrderLineWorkStatus(str, Enum):
    """Enumerating the different kind of work status, particularly open"""

    OPEN = "Open"
    """Open means that the work is on-going or planned"""

    CLOSED = "Closed"
    """Closed means the work has been completed"""

    CANCELLED = "Cancelled"
    """Cancelled means that the work order is no longer relevant"""

    UNKNOWN = WORK_ORDER_LINE_STR_DEFAULT
    """Unknown (default) status"""


@attr.attributes(frozen=True)
class OrderLine:
    """A work order line relevant to physical intrusion detection. Each field's
    metadata specifies the pre-processed column name from the work order source"""

    work_id: WorkOrderID = attr.attrib(
        default=WORK_ORDER_LINE_STR_DEFAULT,
        converter=sanitize_string,
        metadata={"key": "WAREHOUSEWORKID"},
    )
    """Work ID encompassing the entire order, not just the order line number"""

    work_line_num: WorkOrderLineNumber = attr.attrib(
        default=WORK_ORDER_LINE_INT_DEFAULT,
        converter=WorkOrderLineNumber,
        metadata={"key": "WORKLINENUMBER", "start_num": WORK_ORDER_LINE_NUM_START},
    )
    """Work order line number in the sequence"""

    work_loc_id: LocationID = attr.attrib(
        default=WORK_ORDER_LINE_STR_DEFAULT,
        converter=sanitize_string,
        metadata={"key": "WAREHOUSELOCATIONID"},
    )
    """Work location ID"""

    work_type: str = attr.attrib(
        default=WORK_ORDER_LINE_STR_DEFAULT,
        converter=sanitize_string,
        metadata={"key": "WAREHOUSEWORKTYPE"},
    )
    """Specified work type"""

    work_creation_datetime_utc: stdlib_datetime = attr.attrib(
        converter=datetimeify,
        default=datetimeify(DEFAULT_DATETIME_STR),
        metadata={"key": "WAREHOUSEWORKPROCESSINGSTARTDATETIME", "isoformat_sep": " "},
        repr=lambda datetime: datetime.isoformat(),
    )
    """Work creation datetime in the UTC timezone"""

    work_status: str = attr.attrib(
        converter=sanitize_string,
        default=WORK_ORDER_LINE_STR_DEFAULT,
        metadata={"key": "WAREHOUSEWORKSTATUS"},
    )
    """The status of a work order: Open, closed, or cancelled"""

    def asdict(self) -> dict[AttributeKey, Any]:
        """Dictionary of the class"""
        return attr.asdict(self, value_serializer=order_line_serialize)

    def is_work_status_cancelled(self) -> bool:
        """Check if the work status is explicitly cancelled"""
        status = self.work_status.lower()
        return status == OrderLineWorkStatus.CANCELLED.value.lower()

    def is_work_status_open(self) -> bool:
        """Check if the work status is explicitly open"""
        status = self.work_status.lower()
        return status == OrderLineWorkStatus.OPEN.value.lower()

    def is_work_status_closed(self) -> bool:
        """Check if the work status is explicitly closed"""
        status = self.work_status.lower()
        return status == OrderLineWorkStatus.CLOSED.value.lower()

    def is_work_status_set_default(self) -> bool:
        """Check if the work status is explicitly set to the default"""
        status = self.work_status.lower()
        return status == OrderLineWorkStatus.UNKNOWN.value.lower()

    def is_work_status_anything_but_open(self) -> bool:
        """Check if the work status is set to anything but exclusively open"""
        if self.is_work_status_closed():
            return True
        if self.is_work_status_cancelled():
            return True
        if self.is_work_status_set_default():
            return True
        if not self.is_work_status_open():
            logging.getLogger().error(
                "Encountered weird work state %s for work order %s where order line is"
                " supposedly open, but all previous checks yielded otherwise. Check"
                " the human-generated input against hard-coded values",
                self.work_id,
                self.work_status,
            )
            return True
        return False


def key_order_lines_by_number(order_line: OrderLine) -> int:
    """Sorting key for work order lines"""
    if order_line.work_line_num < WORK_ORDER_LINE_NUM_START:
        logging.getLogger().error(
            "The work order line number is less than expected start value of %s",
            WORK_ORDER_LINE_NUM_START,
        )
    return order_line.work_line_num


class OrderLineList(UserList):
    """List of work order lines for a work order"""

    def __init__(self, initlist: Optional[list[OrderLine]] = None) -> None:
        """List of work order lines for a work order"""
        self._logger = logging.getLogger()
        self.data: list[OrderLine]
        super().__init__(initlist)
        if initlist:
            self.data.sort(key=key_order_lines_by_number)
        self.order_lines: list[OrderLine] = self.data

    def __repr__(self):
        return str([order_line.asdict() for order_line in self.data])

    def __str__(self):
        return self.__repr__()

    def get_by_line_num(self, line_num: WorkOrderLineNumber) -> Optional[OrderLine]:
        """Return the order line given a specific order number"""
        if not self.data or len(self.data) == 0:
            self._logger.debug(
                "Tried to get line number %d, but no order lines to access. Returning"
                " None",
                line_num,
            )
            return None
        if line_num >= 0:
            bisect_index = bisect_left(
                a=self.data, x=line_num, key=key_order_lines_by_number
            )
            if bisect_index != len(self.data):
                ret = self.data[bisect_index]
                if ret.work_line_num == line_num:
                    return ret
            self._logger.debug(
                "Tried to get line number %s, but no matching order line was found."
                " Returning None. This is normal if bootstrapping a work order",
                line_num,
            )
            return None
        if line_num == -1:
            return self.order_lines[-1]
        if line_num < 0 and abs(line_num) <= len(self.data):
            return self.get_by_line_num(len(self.data) + (line_num + 1))
        self._logger.error(
            "Unable to get work order line %d from work order %s",
            line_num,
            self.data[0].work_id,
        )
        return None

    def get_last_order_line(self) -> Optional[OrderLine]:
        """Return the last order line"""
        return self.get_by_line_num(-1)


class WorkOrder:
    """A collection of work order lines bootstrapped from empty"""

    def __init__(self, work_id: WorkOrderID) -> None:
        """A collection of work order lines bootstrapped from empty"""
        self._logger = logging.getLogger()
        self.work_id = work_id
        self.work_order_lines: OrderLineList = OrderLineList()

    def __hash__(self):
        return hash(frozenset(self.work_order_lines.order_lines))

    def __eq__(self, other: WorkOrder):
        if self.work_id != other.work_id:
            return False
        if self.work_order_lines != other.work_order_lines:
            return False
        return True

    def __len__(self):
        return self.num_order_lines

    def __repr__(self):
        return str(self.asdict())

    def asdict(
        self,
    ) -> dict[Union[Literal["work_id"], Literal["work_order_lines"]], Union[str, list]]:
        """Construct a dictionary of the instance

        :return: Dictionary
        """
        return {
            "work_id": self.work_id,
            "work_order_lines": [
                work_order_line.asdict() for work_order_line in self.work_order_lines
            ],
        }

    @property
    def num_order_lines(self) -> int:
        """Get the number of work order lines

        :return: Count of the work orders lines
        """
        return len(self.work_order_lines)

    def add_line_item(self, order_line: OrderLine) -> None:
        """Add work order lines to the order. Safety check is applied to ensure work
        order ID matches"""
        if self.can_add_line_item(order_line):
            self._logger.debug(
                "Adding order line %d to work order %s",
                order_line.work_line_num,
                self.work_id,
            )
            insort_left(
                a=self.work_order_lines, x=order_line, key=key_order_lines_by_number
            )

    def can_add_line_item(self, order_line: OrderLine) -> bool:
        """Determine if the order line is appropriate for the work order

        :return: True if the work order line can be added to the work order
        """
        if order_line.work_id != self.work_id:
            return False
        if self.work_order_lines.get_by_line_num(order_line.work_line_num) is not None:
            return False
        return True

    def get_order_line(self, work_line_num: WorkOrderLineNumber) -> Optional[OrderLine]:
        """Wrapper for the OrderLineList.get_by_line_num"""
        return self.work_order_lines.get_by_line_num(work_line_num)

    def get_last_order_line(self) -> Optional[OrderLine]:
        """Wrapper for the OrderLineList.get_last_order_line"""
        self._logger.debug(
            "Searching for the last order line to work order %s", self.work_id
        )
        return self.work_order_lines.get_last_order_line()

    def get_order_line_destination(
        self, work_line_num: WorkOrderLineNumber
    ) -> Optional[LocationID]:
        """For a work order line, get its destination if it exists"""
        self._logger.debug(
            "Searching for work order %s order line %d destination, if available.",
            self.work_id,
            work_line_num,
        )
        next_order_line_num = work_line_num + 1
        next_order_line = self.get_order_line(next_order_line_num)
        if next_order_line:
            self._logger.debug("Found work order line destination")
            return next_order_line.work_loc_id
        return None

    def is_associated_with_location(self, loc_id: LocationID) -> bool:
        """Determine if a location is associated with the work order

        :param loc_id: Location ID
        :return: True if the location ID is found in the work order, else false
        """
        return any(
            loc_id == order_line.work_loc_id
            for order_line in self.work_order_lines.order_lines
        )

    def is_work_status_open(self) -> bool:
        """Check if the work status is explicitly cancelled"""
        return self.get_last_order_line().is_work_status_open()

    def is_work_status_closed(self) -> bool:
        """Check if the work status is explicitly closed"""
        return self.get_last_order_line().is_work_status_closed()

    def is_work_status_cancelled(self) -> bool:
        """Check if the work status is explicitly cancelled"""
        return self.get_last_order_line().is_work_status_cancelled()

    def is_work_status_set_default(self) -> bool:
        """Check if the work status is explicitly set to the default"""
        return self.get_last_order_line().is_work_status_set_default()

    def is_work_status_anything_but_open(self) -> bool:
        """Check if the work status is set to anything but exclusively open"""
        return self.get_last_order_line().is_work_status_anything_but_open()


def key_work_order_by_work_id(work_order: WorkOrder) -> WorkOrderID:
    """Sorting key for work orders by the work order ID"""
    return work_order.work_id


def key_work_first_order_by_datetime(work_order: WorkOrder) -> stdlib_datetime:
    """Sorting key for work orders by the first work order line creation datetime"""
    order_line: Optional[OrderLine] = work_order.get_order_line(
        WORK_ORDER_LINE_NUM_START
    )
    if order_line:
        return order_line.work_creation_datetime_utc
    return stdlib_datetime.fromtimestamp(0, tz=tz.tzutc())


class WorkOrderList(UserList):
    """A collection of work orders. Orders can be appended by work ID and datetime."""

    def __init__(self, initlist: Optional[list[WorkOrder]] = None) -> None:
        self._logger = logging.getLogger()
        self.data: list[WorkOrder]
        super().__init__(initlist)
        self.work_orders: list[WorkOrder] = self.data
        # self._work_orders_work_id_index: dict[int, int] = {}
        # if initlist:
        #     self.update_work_order_indexing()

    def __repr__(self):
        return str(self.asdict())

    def asdict(self) -> dict[Literal["work_orders"], list[dict]]:
        """Construct a dictionary of the instance. The output is a dictionary
         with a single key

        :return:
        """
        return {"work_orders": [work_order.asdict() for work_order in self.data]}

    def append_by_datetime(self, work_order: WorkOrder) -> None:
        """Append to the work order list according to the first order line datetime"""
        # self._logger.debug(f"Adding work order {work_order.work_id} by datetime")
        insort_left(a=self.data, x=work_order, key=key_work_first_order_by_datetime)

    def append_by_work_id(self, work_order: WorkOrder) -> None:
        """Append to the work order list according to the work ID"""
        # self._logger.debug(f"Adding work order {work_order.work_id} by work order ID")
        insort_left(a=self.data, x=work_order, key=key_work_order_by_work_id)

    def find_by_work_id(self, work_id: WorkOrderID) -> Optional[WorkOrder]:
        """Find a work order by work ID if in the list"""
        bisect_index = bisect_left(
            a=self.data, x=work_id, key=key_work_order_by_work_id
        )
        if bisect_index != len(self.data):
            return self.data[bisect_index]
        return None


class BaseWorkLog:
    """A work log constructed from a table using Pandas or other sources"""

    def __init__(self) -> None:
        self._logger = logging.getLogger()


class WorkOrderSources(str, Enum):
    """Types of work order sources"""

    SQLDATABASE = "SQLDATABASE"
    """Remotely obtained work order log from a SQL database"""

    FILE = "FILE"
    """A locally, singly file"""

    FILES = "FILES"
    """Locally multiple files"""


class WarehouseWorkLog(BaseWorkLog):
    """A warehouse work log constructed from a log file or parsed log file using
    pandas"""

    def __init__(self, pandas_df: Optional[pd.DataFrame] = None) -> None:
        """A warehouse work log constructed from a log file, parsed log file using
        pandas, or constructed by work order.

        :param pandas_df: Parsed logfile in a DataFrame
        """
        super().__init__()
        self.dataframe: ClassVar[Optional[pd.DataFrame]] = None
        """DataFrame that contains parsed work log information"""
        self.work_orders_by_datetime: ClassVar[WorkOrderList] = WorkOrderList()
        """List of work orders sorted by the first line item creation datetime"""
        self.work_orders_by_work_id: WorkOrderList = WorkOrderList()
        """List of work orders sorted by the work ID string"""
        self._df_column_key_2_dataclass_attrib: dict[str, str] = {
            value.metadata["key"]: key
            for key, value in attr.fields_dict(OrderLine).items()
        }
        if isinstance(pandas_df, pd.DataFrame):
            self.dataframe = pandas_df.copy(deep=True)
            self._pop_unused_keys()
            self._build_work_orders()

    @classmethod
    def from_log_file(cls, work_log_filename: PathLike, format_filename: PathLike):
        """Load into a DataFrame a work log

        :param work_log_filename: Pre-processed work log
        :param format_filename: Column type specifiers and converters
        """
        pandas_df = load_raw_log_file(work_log_filename, format_filename)
        return cls(pandas_df)

    @classmethod
    def from_sql_database(
        cls, connstring: str, querystring: str, format_filename: PathLike
    ):
        """Load a work log from a SQL database

        :param connstring: Connection string
        :param querystring: Query string
        :param formatter: Log formatter file
        :return: Dataframe with the format specified by the log format parameter file
        """
        pandas_df = load_log_from_sql_database(
            connstring, querystring, os.fspath(format_filename)
        )
        return cls(pandas_df)

    def __len__(self):
        return len(self.work_orders_by_datetime)

    def _pop_unused_keys(self):
        """Remove all non-essential dataframe columns"""
        save_df_columns = self._df_column_key_2_dataclass_attrib.keys()
        all_keys = list(self.dataframe.keys())
        pop_keys = all_keys.copy()
        for save_key in save_df_columns:
            if save_key not in all_keys:
                self._logger.exception(
                    "The save key %s is NOT in the dataframe columns", save_key
                )
                raise KeyError
            pop_keys.remove(save_key)
        for pop_key in pop_keys:
            self.dataframe.pop(pop_key)

    def _build_work_orders(self):
        """Populate the list of work orders"""
        order_line_fields = attr.fields(OrderLine)
        order_line_pd_series: pd.Series
        work_id_key: str = order_line_fields.work_id.metadata.get("key")
        work_order_line_num_key: str = order_line_fields.work_line_num.metadata.get(
            "key"
        )
        work_id: WorkOrderID
        work_order_line_num: WorkOrderLineNumber
        new_order: Optional[WorkOrder] = None
        # Each row is a Series of work order line. This loop converts the Series into a
        # WorkOrder and catalogues it by both datetime and work order ID
        for _, order_line_pd_series in self.dataframe.iterrows():
            order_line_from_pd_series_dict: dict = dict(order_line_pd_series)
            work_order_line_num = order_line_from_pd_series_dict[
                work_order_line_num_key
            ]
            # If we arrive at the start of a work order line, then catalogue
            if work_order_line_num == WORK_ORDER_LINE_NUM_START:
                if new_order is not None:
                    self.work_orders_by_datetime.append_by_datetime(new_order)
                    self.work_orders_by_work_id.append_by_work_id(new_order)
                work_id = order_line_pd_series[work_id_key]
                new_order = WorkOrder(work_id)
            # By default, create a work order line
            order_line_kwargs = {
                self._df_column_key_2_dataclass_attrib[pd_series_key]: pd_series_value
                for (
                    pd_series_key,
                    pd_series_value,
                ) in order_line_from_pd_series_dict.items()
            }
            try:
                order_line = OrderLine(**order_line_kwargs)
                new_order.add_line_item(order_line)
            except AttributeError:
                self._logger.exception(
                    "Unable to get attribute for all kwargs constructor from %s",
                    str(order_line_kwargs),
                )
        if new_order is not None:
            self.work_orders_by_datetime.append_by_datetime(new_order)
            self.work_orders_by_work_id.append_by_work_id(new_order)

    def add_work_order(self, work_order: WorkOrder):
        """Add a work order to the work log"""
        self.work_orders_by_work_id.append_by_work_id(work_order)
        self.work_orders_by_datetime.append_by_datetime(work_order)

    def get_work_order(self, work_id: WorkOrderID) -> Optional[WorkOrder]:
        """Search for a work order by ID

        :param work_id: Word order ID
        :return: WorkOrder instance if the ID is found, None otherwise
        """
        self._logger.debug("Searching for work order %s.", work_id)
        bisect_index = bisect_left(
            a=self.work_orders_by_work_id, x=work_id, key=key_work_order_by_work_id
        )
        if bisect_index != len(self.work_orders_by_work_id):
            ret: WorkOrder = self.work_orders_by_work_id[bisect_index]
            if ret.work_id == work_id:
                self._logger.debug("Found work order %s", ret.work_id)
                return ret
        self._logger.debug("Searching for work order %s, but found none", work_id)
        return None

    def get_work_orders_between_timedelta(
        self,
        start_datetime: stdlib_datetime,
        stop_datetime: Union[stdlib_timedelta, stdlib_datetime],
    ) -> WorkOrderList:
        """Get all work orders in a time window

        :param start_datetime: Start datetime of the search
        :param stop_datetime: Stop datetime or timedelta from start of the search
        :return: Work order list, empty if no orders are found
        """
        final_datetime: stdlib_datetime
        if isinstance(stop_datetime, stdlib_timedelta):
            final_datetime = start_datetime + stop_datetime
        else:
            final_datetime = stop_datetime
        self._logger.debug(
            "Searching for work orders between %s and %s",
            start_datetime.isoformat(),
            stop_datetime.isoformat(),
        )
        left_bisect_index = bisect_left(
            a=self.work_orders_by_datetime,
            x=start_datetime,
            key=key_work_first_order_by_datetime,
        )
        right_bisect_index = bisect_left(
            a=self.work_orders_by_datetime,
            x=final_datetime,
            key=key_work_first_order_by_datetime,
        )
        self._logger.debug("Found %s record(s)", right_bisect_index - left_bisect_index)
        work_orders = WorkOrderList(
            self.work_orders_by_datetime[left_bisect_index:right_bisect_index]
        )
        return work_orders


def order_line_serialize(inst, field, value):
    """Serialize an OrderLine as used in the OrderLine.asdict method

    :param inst: Not used
    :param field: Not Used
    :param value: Value
    :return: Serializable
    """
    logging.getLogger().debug(
        "Order line serialize with instance %s, field %s, and value %s",
        str(inst),
        str(field),
        str(value),
    )
    try:
        return serializer.serialize_with_exception(value)
    except serializer.SerializeError:
        return value


class OrderLineEncoder(json.JSONEncoder):
    """Encoder for an OrderLine"""

    def default(self, o):
        """Super method for JSONEncoder.default"""
        if isinstance(o, OrderLine):
            return str(o.asdict())
        return super().default(o)


class OrderLineListEncoder(json.JSONEncoder):
    """Encoder for an OrderLineList"""

    def default(self, o):
        """Super method for JSONEncoder.default"""
        if isinstance(o, OrderLineList):
            out = [order_line.asdict() for order_line in o.order_lines]
            return str(out)
        return super().default(o)


class WorkOrderEncoder(json.JSONEncoder):
    """Encoder for a WorkOrder"""

    def default(self, o):
        """Super method for JSONEncoder.default"""
        if isinstance(o, WorkOrder):
            out = o.asdict()
            return str(out)
        return super().default(o)


class WorkOrderListEncoder(json.JSONEncoder):
    """Encoder for a WorkOrderList"""

    def default(self, o):
        """Super method for JSONEncoder.default"""
        if isinstance(o, WorkOrderList):
            return str(
                [WorkOrderEncoder().default(work_order) for work_order in o.work_orders]
            )
        return super().default(o)


class WarehouseWorkLogEncoder(json.JSONEncoder):
    """Encoder for a WorkOrderLog"""

    def default(self, o):
        """Super method for JSONEncoder.default"""
        if isinstance(o, WarehouseWorkLog):
            return WorkOrderListEncoder().default(o.work_orders_by_work_id)
        return super().default(o)


def load_log(
    source: WorkOrderSources,
    filename: Optional[PathLike] = None,
    connstring: Optional[str] = None,
    querystring: Optional[str] = None,
    formatter: Optional[PathLike] = None,
) -> pd.DataFrame:
    """

    :param source: Enumerated source
    :param filename: The path to warehouse management log file
    :param connstring: Connection string
    :param querystring: Query string
    :param formatter: Log formatter file
    :return: Dataframe with the format specified by the log format parameter file
    """
    logger = logging.getLogger()
    dataframe: Optional[pd.DataFrame] = None
    log_format = get_log_format(formatter)
    data_type_dict: dict[str, Any] = log_format["COLUMNS"]
    datetime_format: str = log_format["DATETIME_FORMAT"]
    names: list[str] = list(map(str.upper, list(data_type_dict.keys())))
    if source == WorkOrderSources.FILE:
        logger.info(
            "Loading raw log file %s with formatter file %s",
            os.fspath(filename),
            os.fspath(formatter),
        )
        filename_suffix = str(Path(filename).suffix).upper().strip(".")
        if filename_suffix in SPREADSHEET:
            dataframe = pd.read_excel(
                io=os.fspath(filename),
                sheet_name=log_format["SHEET_NAME"],
                names=names,
                dtype=data_type_dict,
                header=0,  # Header line is line 0 (first line)
            )
        elif filename_suffix in ("CSV",):
            dataframe = pd.read_csv(
                filepath_or_buffer=os.fspath(filename),
                names=names,
                sep=",",
                dtype=data_type_dict,
                header=0,  # Header line is line 0 (first line)
            )
        else:
            logger.exception(
                "Unable to determine the D365 log format for %s", os.fspath(formatter)
            )
            raise ValueError
    elif source == WorkOrderSources.SQLDATABASE:
        logger.info(
            "Loading log from database %s with formatter file %s",
            connstring,
            os.fspath(formatter),
        )
        try:
            with odbc_connect(connstring, timeout=60) as connection:
                dataframe = pd.read_sql(
                    sql=querystring,
                    con=connection,
                    columns=names,
                )
        except Exception as err:
            logger.exception("Unable to obtain SQL table due to err %s", str(err))
    else:
        logger.exception(
            "The input work order source %s is NOT implemented", source.value
        )
        raise NotImplementedError

    if dataframe is not None:
        for col_name in names:
            if "DATETIME" in col_name:
                dataframe[col_name] = pd.to_datetime(
                    dataframe[col_name], format=datetime_format
                )
                dataframe[col_name] = dataframe[col_name].dt.tz_localize(
                    tz.gettz(log_format["DATETIME_LOCALE"])
                )
    return dataframe


def load_log_from_sql_database(
    connstring: str, querystring: str, formatter: PathLike
) -> pd.DataFrame:
    """Retrieve a Pandas Dataframe from a SQL database query

    :param connstring: Connection string
    :param querystring: Query string
    :param formatter: Log formatter file
    :return: Dataframe with the format specified by the log format parameter file
    """
    return load_log(
        source=WorkOrderSources.SQLDATABASE,
        connstring=connstring,
        querystring=querystring,
        formatter=os.fspath(formatter),
    )


def load_raw_log_file(work_log_filename: PathLike, formatter: PathLike) -> pd.DataFrame:
    """Get a pandas Dataframe from a log file

    :param work_log_filename: The path to warehouse management log file
    :param formatter: The path to a JSON file describing the log file format
    :return: Dataframe with the format specified by the log format parameter file
    """
    return load_log(
        source=WorkOrderSources.FILE,
        filename=os.fspath(work_log_filename),
        formatter=os.fspath(formatter),
    )


def add_line_items(work_order: WorkOrder, work_order_lines: list[OrderLine]) -> None:
    """Add work order lines to a work order

    :param work_order: Work order instance
    :param work_order_lines: list of work order lines
    :return:
    """
    # logger = logging.getLogger()
    for order_line in work_order_lines:
        # logger.debug(f"Adding {order_line.work_line_num} to {work_order.work_id}")
        work_order.add_line_item(order_line)


def check_work_log_for_activity_at_locations(
    work_log: WarehouseWorkLog,
    location_ids: Sequence[LocationID],
) -> dict[LocationID, Optional[WorkOrder]]:
    """Given a work log, find the first work order involving specific location IDs

    :param work_log: Work log
    :param location_ids: Sequence of location IDs to cross-reference
    :return: Mapping between location IDs --> work order, if any, else none
    """
    out_dict = {
        loc_id: next(
            (
                work_order
                for work_order in work_log.work_orders_by_work_id.work_orders
                if work_order.is_associated_with_location(loc_id)
            ),
            None,
        )
        for loc_id in location_ids
    }
    return out_dict
