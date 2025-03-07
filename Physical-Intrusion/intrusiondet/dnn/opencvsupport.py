"""Supported backand and targets using OpenCV"""
import os
from typing import Final

try:
    from cv2 import cv2
except ImportError:
    import cv2

# fmt: off
BACKENDS: Final[tuple[int, ...]] = (
    cv2.dnn.DNN_BACKEND_DEFAULT,                                   # 0
    cv2.dnn.DNN_BACKEND_HALIDE,  # Untested in this API            # 1
    cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,  # Untested in this API  # 2
    cv2.dnn.DNN_BACKEND_OPENCV,                                    # 3
    cv2.dnn.DNN_BACKEND_VKCOM,  # Untested in this API             # 4
    cv2.dnn.DNN_BACKEND_CUDA,  # Untested in this API              # 5
)
"""Enumerated OpenCV backends"""


TARGETS: Final[tuple[int, ...]] = (
    cv2.dnn.DNN_TARGET_CPU,                                     # 0
    cv2.dnn.DNN_TARGET_OPENCL,                                  # 1
    cv2.dnn.DNN_TARGET_OPENCL_FP16,  # Untested in this API     # 2
    cv2.dnn.DNN_TARGET_MYRIAD,  # Untested in this API          # 3
    cv2.dnn.DNN_TARGET_HDDL,  # Untested in this API            # 8
    cv2.dnn.DNN_TARGET_FPGA,  # Untested in this API            # 5
    cv2.dnn.DNN_TARGET_VULKAN,  # Untested in this API          # 4
    cv2.dnn.DNN_TARGET_CUDA,  # Untested in this API            # 6
    cv2.dnn.DNN_TARGET_CUDA_FP16,  # Untested in this API       # 7
)
"""Enumerated OpenCV target processors"""


SUPPORTED_BACKENDS_STR: Final[str] = os.linesep.join((
    "OpenCV computation backends: ",
    f" * {cv2.dnn.DNN_BACKEND_DEFAULT         }: automatically (by default),",       # 0
    f" * {cv2.dnn.DNN_BACKEND_HALIDE          }: Halide language -- UNTESTED,",      # 1
    f" * {cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE}: OPENVINO -- UNTESTED,",             # 2
    f" * {cv2.dnn.DNN_BACKEND_OPENCV          }: OpenCV implementation -- UNTESTED,",# 3
    f" * {cv2.dnn.DNN_BACKEND_VKCOM           }: VKCOM -- UNTESTED,",                # 4
    f" * {cv2.dnn.DNN_BACKEND_CUDA            }: CUDA -- UNTESTED",                  # 5
))
"""Human-friendly version of the OpenCV supported backends"""


SUPPORTED_TARGETS_STR: Final[str] = os.linesep.join((
    "The accepted backend inputs are: ",
    f" * {cv2.dnn.DNN_TARGET_CPU        }: CPU target (by default),",             # 0
    f" * {cv2.dnn.DNN_TARGET_OPENCL     }: OpenCL GPU,",                          # 1
    f" * {cv2.dnn.DNN_TARGET_OPENCL_FP16}: OpenCL GPU (half-float) -- UNTESTED",  # 2
    f" * {cv2.dnn.DNN_TARGET_MYRIAD     }: NCS2 VPU -- UNTESTED",                 # 3
    f" * {cv2.dnn.DNN_TARGET_HDDL       }: HDDL VPU -- UNTESTED",                 # 8
    f" * {cv2.dnn.DNN_TARGET_FPGA       }: FPGA -- UNTESTED",                     # 5
    f" * {cv2.dnn.DNN_TARGET_VULKAN     }: Vulkan -- UNTESTED",                   # 4
    f" * {cv2.dnn.DNN_TARGET_CUDA       }: CUDA -- UNTESTED",                     # 6
    f" * {cv2.dnn.DNN_TARGET_CUDA_FP16  }: CUDA (half-float) -- UNTESTED",        # 7
))
"""Human-friendly version of the OpenCV supported target processors"""
# fmt: on
