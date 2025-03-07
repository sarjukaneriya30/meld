"""Overhead to manage and run an OpenCV DNN model"""

import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from intrusiondet.core.types import NamedObject, NDArrayFloat
from intrusiondet.dnn.types import DNNOutput


class TargetProcessorIsNotSupportedOnHostException(Exception):
    """When attempting to use a target processor other than CPU, raise this error when
    that target processor is not supported"""


class BaseDNNModel(NamedObject):
    """The building blocks to implement an OpenCV DNN model"""

    def __init__(self, name: str, *args, **kwargs):
        """The building blocks to implement an OpenCV DNN model

        :param name: A unique name to specify this instance
        :param args: No arguments are required nor checked
        :param kwargs: No keyword arguments are required nor checked
        """
        self._logger = logging.getLogger()
        self._logger.info("Building DNN base model")
        super().__init__(name, *args, **kwargs)

    @abstractmethod
    def process_frame(self, *args, **kwargs) -> Any:
        """Process an image frame

        :param args: No args are accepted as this method is not implemented
        :param kwargs: No kwargs are accepted as this method is not implemented
        :raises NotImplementedError: This method is not implemented for BaseDNNModel
        """
        raise NotImplementedError

    @abstractmethod
    def process_frames(self, *args, **kwargs) -> Any:
        """Process a frame batch (multiple frames in a single blob)

        :param args: No args are accepted as this method is not implemented
        :param kwargs: No kwargs are accepted as this method is not implemented
        :raises NotImplementedError: This method is not implemented for BaseDNNModel
        """
        raise NotImplementedError

    @abstractmethod
    def process_image(self, *args, **kwargs) -> Any:
        """Wrapper for process_frame method for a single image

        :param args: No args are accepted as this method is not implemented
        :param kwargs: No kwargs are accepted as this method is not implemented
        :raises NotImplementedError: This method is not implemented for BaseDNNModel
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def classes(self):
        """Get the list of class names for the model

        :return: Class names list
        """
        raise NotImplementedError


def convert_multiple_frame_dnn_output_to_single_frame_dnn_output(
    dnn_multiple_frame_output: DNNOutput, dnn_output_index: int
) -> DNNOutput:
    """Take the output from multiple (batched) DNN input frames, obtain the results into
    a single digestible piece

    :param dnn_multiple_frame_output: Multiple (batched) frame output
    :param dnn_output_index: Index in the batch
    :return: Single DNN output blob
    """
    assert all(
        dnn_output_index < layer_blob.shape[0]
        for layer_blob in dnn_multiple_frame_output
    )
    dnn_output_single_frame: list[NDArrayFloat] = [
        np.array([], dtype=float),
    ] * len(dnn_multiple_frame_output)
    for dnn_layer_index, dnn_layer_blob in enumerate(dnn_multiple_frame_output):
        dnn_output_single_frame[dnn_layer_index] = dnn_layer_blob[dnn_output_index]
    assert all(len(dnn_output.shape) == 2 for dnn_output in dnn_output_single_frame)
    return tuple(dnn_output_single_frame)
