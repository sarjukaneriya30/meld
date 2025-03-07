"""Type aliases for OpenCV DNN output types"""
from intrusiondet.core.types import NDArrayFloat, NDArrayInt

DNNOutput = tuple[NDArrayFloat, ...]
"""Output from a DNN"""

DNNObjectConfidences = NDArrayFloat
"""Confidence values for a set of detected objects in a `DNNOutput` instance"""

DNNClassIDs = NDArrayInt
"""Class IDs for a set of detected objects in a `DNNOutput` instance"""
