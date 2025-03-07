"""Mapping camera observation locations to work order areas"""

from intrusiondet.core.types import LocationID, WorkOrderLineNumber
from intrusiondet.model.videoreadercamera import WarehouseLocationCameraAdjacencyFinder
from intrusiondet.model.warehousemanagement import WorkOrder


def get_locations_in_work_order(
    adj_finder: WarehouseLocationCameraAdjacencyFinder, work_order: WorkOrder
) -> dict[WorkOrderLineNumber, frozenset[LocationID]]:
    """Map the camera locations expected to observe agents working on a work order

    :param work_order: Work order
    :param adj_finder: Camera adjacencies finder
    :return: Mapping between work order line number and the expected cameras
    """
    work_order_camera_locs = dict.fromkeys(
        range(1, work_order.num_order_lines + 1), frozenset()
    )
    for work_order_line_num in range(1, work_order.num_order_lines):
        start_loc = work_order.get_order_line(work_order_line_num).work_loc_id
        end_loc = work_order.get_order_line_destination(work_order_line_num)
        work_order_camera_locs[work_order_line_num] = adj_finder.find_cameras_in_path(
            start_loc, end_loc
        )
    return work_order_camera_locs
