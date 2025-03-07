"""Stop watches for measure time differences between operations"""
from datetime import datetime, timezone

from intrusiondet.core.types import TimeDifferentialInSeconds, TimeStampInSeconds


class BasicStopWatch:
    """Stop watch operations to measure time differences on demand.
    For sensible time differentials, one must start and then stop the watch.
    A reset synchronizes the start and stop times"""

    def __init__(self) -> None:
        """Stop watch operations to measure time differences on demand.
        For sensible time differentials, one must start and then stop the watch.
        A reset synchronizes the start and stop times"""
        self._start: datetime = datetime.now(timezone.utc)
        self._stop: datetime = self._start
        self._has_started: bool = False

    def reset(self) -> None:
        """Reset the stop watch, both start and stop times.

        :return: Nothing
        """
        self._start = self._stop = datetime.now(timezone.utc)
        self._has_started = False

    def start(self) -> TimeStampInSeconds:
        """Start the stop watch timer

        :return: POSIX timestamp
        """
        self._start = datetime.now(timezone.utc)
        self._has_started = True
        return self._start.timestamp()

    def stop(self) -> TimeDifferentialInSeconds:
        """Measure the time differential from the most recent start.
        The watch is still running as the stop marker is updated only.

        :return: Time difference in seconds
        """
        self._stop = datetime.now(timezone.utc)
        out = self.delta_sec
        self._has_started = True
        return out

    @property
    def delta_sec(self) -> TimeDifferentialInSeconds:
        """Measure the current time differential at the last stop.

        :return: Time difference in seconds
        """
        return (self._stop - self._start).total_seconds()

    def is_running(self) -> bool:
        """Determine if the stopwatch is running"""
        return self._has_started
