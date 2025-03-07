"""Alerting tools for a frontend"""
from enum import Flag


class ObservedActivityFlag(Flag):
    """Flag for measuring if a location has an observation"""

    NOACTIVITY = 0
    ACTIVITY = 2


class WorkOrderFlag(Flag):
    """Flag for measuring if a location has any work orders"""

    NONE = 0
    FOUND = 1


class ObservedActivityWorkOrderFlag(Flag):
    """Flag for measuring if a location has observations and work orders simultaneously"""

    NOACTIVITYNONEWORKORDER = 1 << (
        ObservedActivityFlag.NOACTIVITY.value + WorkOrderFlag.NONE.value
    )  # 2**0 = 1
    NOACTIVITYFOUNDWORKORDER = 1 << (
        ObservedActivityFlag.NOACTIVITY.value + WorkOrderFlag.FOUND.value
    )  # 2**1 = 2
    ACTIVITYNONEWORKORDER = 1 << (
        ObservedActivityFlag.ACTIVITY.value + WorkOrderFlag.NONE.value
    )  # 2**2 = 4
    ACTIVITYFOUNDWORKORDER = 1 << (
        ObservedActivityFlag.ACTIVITY.value + WorkOrderFlag.FOUND.value
    )  # 2**3 = 8

    @classmethod
    def combine_flags(
        cls, activity_flag: ObservedActivityFlag, work_found_flag: WorkOrderFlag
    ):
        """Combine the ObservedActivityFlag and WorkOrderFlag flags

        :param activity_flag: Activity flag
        :param work_found_flag: Work found flag
        :return: ObservedActivityWorkOrderFlag flag
        """
        return cls(1 << (activity_flag.value + work_found_flag.value))


def is_no_activity_none_work(
    activity_flag: ObservedActivityFlag, work_found_flag: WorkOrderFlag
):
    """Test if there is no activity and no work logs for a location using flags

    :param activity_flag: Activity flag
    :param work_found_flag: Work found flag
    :return: True if no activity and no work logs for a location, false otherwise
    """
    return (
        ObservedActivityWorkOrderFlag.combine_flags(activity_flag, work_found_flag)
        == ObservedActivityWorkOrderFlag.NOACTIVITYNONEWORKORDER
    )


def is_activity_none_work(
    activity_flag: ObservedActivityFlag, work_found_flag: WorkOrderFlag
):
    """Test if there is activity and no work logs for a location using flags

    :param activity_flag: Activity flag
    :param work_found_flag: Work found flag
    :return: True if there is activity and no work logs for a location, false otherwise
    """
    return (
        ObservedActivityWorkOrderFlag.combine_flags(activity_flag, work_found_flag)
        == ObservedActivityWorkOrderFlag.ACTIVITYNONEWORKORDER
    )


def is_no_activity_found_work(
    activity_flag: ObservedActivityFlag, work_found_flag: WorkOrderFlag
):
    """Test if there is no activity, but found work logs for a location using flags

    :param activity_flag: Activity flag
    :param work_found_flag: Work found flag
    :return: True if there is no activity, but found work logs for a location, false otherwise
    """
    return (
        ObservedActivityWorkOrderFlag.combine_flags(activity_flag, work_found_flag)
        == ObservedActivityWorkOrderFlag.NOACTIVITYFOUNDWORKORDER
    )


def is_activity_found_work(
    activity_flag: ObservedActivityFlag, work_found_flag: WorkOrderFlag
):
    """Test if there is activity and found work logs for a location using flags

    :param activity_flag: Activity flag
    :param work_found_flag: Work found flag
    :return: True if there is activity and found work logs for a location, false otherwise
    """
    return (
        ObservedActivityWorkOrderFlag.combine_flags(activity_flag, work_found_flag)
        == ObservedActivityWorkOrderFlag.ACTIVITYFOUNDWORKORDER
    )
