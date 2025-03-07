"""YOLO model as implemented in OpenCV"""
import logging
import os
from enum import Enum
from io import TextIOWrapper
from pathlib import Path
from typing import ClassVar, Final, Optional, Sequence, Union

import numpy as np

import intrusiondet.dnn.opencvsupport
from intrusiondet.core.image import get_frame
from intrusiondet.core.types import (
    ClassID,
    ClassNameID,
    Frame,
    FrameIndex,
    IntOrFloat,
    NDArrayFloat,
    NDArrayInt,
    PathLike,
)
from intrusiondet.dnn.basednnmodel import (
    BaseDNNModel,
    TargetProcessorIsNotSupportedOnHostException,
    convert_multiple_frame_dnn_output_to_single_frame_dnn_output,
)
from intrusiondet.dnn.blob import Blob, frame2blob, frames2blob
from intrusiondet.dnn.classnames import get_class_names
from intrusiondet.dnn.types import DNNClassIDs, DNNObjectConfidences, DNNOutput
from intrusiondet.model.detectedobject import DetectedObject

try:
    import cv2.cv2 as cv
except ImportError:
    import cv2 as cv


class YoloModel(BaseDNNModel):
    """Create a YOLO model DNN using OpenCV bindings given the directory path for the
    configuration"""

    class LayerTypes(str, Enum):
        """OpenCV DNN model layer types enumerated as strings"""

        DETECTION_OUTPUT: Final[str] = "DetectionOutput"
        """Output DNN layer is the "DetectionOutput" type"""

        REGION: Final[str] = "Region"
        """Output DNN layer is the "Region" type"""

    def __init__(
        self,
        name: str,
        yolo_dir: str,
        backend: Optional[int] = None,
        target: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Create a YOLO model DNN given the directory path for the configuration

        :param name: The name of this YOLO model instance
        :param yolo_dir: The path to the directory where YOLO model is saved
        :param target: OpenCV DNN target processing-unit integer enumeration
            (default=None)
        :param backend: OpenCV DNN backend integer enumeration (default=None)
        :param args: No arguments are required nor checked
        :param kwargs: No keyword arguments are required nor checked
        :raises ValueError: If either the target or backend are not in the supported
            OpenCV integer enumeration
        """
        super().__init__(name, *args, **kwargs)
        self.yolo_dir: Final[Path] = Path(yolo_dir)
        """Path to the directory containing the following files: coco.names, yolo.cfg,
        and yolo.weights"""
        self.backend: int = backend
        """OpenCV backend specification"""
        self.target: int = target
        """OpenCV DNN target processing-unit"""
        self.yolo_cfg: ClassVar[list[dict]]
        """Parsed YOLO configuration file"""
        self._network: cv.dnn_Net
        """OpenCV DNN"""
        self._layer_names: ClassVar[list[str]]
        """Names of output layers"""
        self._classes: ClassVar[list[ClassNameID]]
        """List of the YOLO prediction class names that establishes a mapping: class ID
        --> unique name"""
        self.class_indices: ClassVar[dict[ClassNameID, ClassID]]
        """Inverse mapping of classes: unique name --> class ID"""
        self.out_layer_names: ClassVar[list[str]]
        """The final output DNN layer names"""
        self.out_size: ClassVar[tuple[FrameIndex, FrameIndex]]
        """The output frame size as specified by the configuration file"""
        self.last_layer: ClassVar[cv.dnn_Layer]  # pylint: disable=no-member
        """The instance of the last DNN layer"""

        self._load_model()

    def _load_model(self) -> None:
        """Method to load the model parameters into memory

        :raises IOError: When the input YOLO model directory does not contain the
            following: yolo.cfg, yolo.weights, coco.names
        :raises ValueError: When the specified backend or target processor is invalid
        :raises TargetProcessorIsNotSupportedOnHostException: When target processor is
            not configured properly for the host
        """
        cfg_file = self.yolo_dir / "yolo.cfg"
        weights_file = self.yolo_dir / "yolo.weights"
        names_file = self.yolo_dir / "coco.names"
        if not cfg_file.is_file():
            self._logger.exception(
                "Unable to find YOLO configuration file %s. Perhaps you need to "
                "download the model?",
                os.fspath(cfg_file),
            )
            raise IOError
        self._logger.info("Reading configuration file %s", os.fspath(cfg_file))
        self.yolo_cfg = read_config(os.fspath(cfg_file))
        net_dict: dict[str, Union[float, str, int]]
        if len(self.yolo_cfg) == 0:
            self._logger.error(
                "Unable to correctly parse YOLO configuration file %s as there are no"
                "sections/layers",
                os.fspath(cfg_file),
            )
            raise IOError
        net_dict = self.yolo_cfg[0].get("net", None)
        if net_dict is None:
            self._logger.error(
                "Unable to correctly parse YOLO configuration file %s as there is no"
                '"net" section',
                os.fspath(cfg_file),
            )
            raise IOError
        self.out_size = FrameIndex(net_dict["width"]), FrameIndex(net_dict["height"])
        if not weights_file.is_file():
            self._logger.exception(
                "Unable to find YOLO weights file %s. Perhaps you need to "
                "download the model?",
                os.fspath(weights_file),
            )
            raise IOError
        if not names_file.is_file():
            self._logger.exception(
                "Unable to find COCO names file %s. Perhaps you need to "
                "download the model?",
                os.fspath(names_file),
            )
            raise IOError

        self._logger.info(
            "Loading YOLO model from config file %s and weights file %s into memory",
            os.fspath(cfg_file),
            os.fspath(weights_file),
        )
        self._network = cv.dnn.readNetFromDarknet(  # pylint: disable=no-member
            os.fspath(cfg_file), os.fspath(weights_file)
        )
        if self.target is None:
            logging.info(
                "YOLO DNN: No target set! Using CPU target %d by default",
                cv.dnn.DNN_TARGET_CPU,  # pylint: disable=no-member
            )
            self.target = cv.dnn.DNN_TARGET_CPU  # pylint: disable=no-member
        if self.target not in intrusiondet.dnn.opencvsupport.TARGETS:
            self._logger.exception(
                "Unable to use input DNN computation target %d. Please see below %s%s",
                self.target,
                os.linesep,
                intrusiondet.dnn.opencvsupport.SUPPORTED_TARGETS_STR,
            )
            raise ValueError
        if (
            self.target == cv.dnn.DNN_TARGET_OPENCL  # pylint: disable=no-member
        ):  # pylint: disable=no-member
            self._logger.info("Attempting to use target processor OpenCL")
            if cv.ocl.haveOpenCL():  # pylint: disable=no-member
                cv.ocl.useOpenCL()  # pylint: disable=no-member
            else:
                self._logger.exception(
                    "Unable to use target processor OpenCL since host does not support"
                    "it"
                )
                raise TargetProcessorIsNotSupportedOnHostException
        self._network.setPreferableTarget(self.target)
        if self.backend is None:
            self._logger.info("No backend specified. Using OpenCV backend as default")
            self.backend = cv.dnn.DNN_BACKEND_OPENCV  # pylint: disable=no-member
        if self.backend not in intrusiondet.dnn.opencvsupport.BACKENDS:
            self._logger.exception(
                "Unable to use input DNN backend %d. Please see below %s%s",
                self.backend,
                os.linesep,
                intrusiondet.dnn.opencvsupport.SUPPORTED_BACKENDS_STR,
            )
            raise ValueError
        self._logger.info("Setting backend to %d", self.backend)
        self._network.setPreferableBackend(self.backend)
        self._logger.info("Loaded core YOLO model components")

        self._logger.info("Finalizing YoloModel class configuration")
        self._layer_names = self._network.getLayerNames()
        self.out_layer_names = self._network.getUnconnectedOutLayersNames()
        self._classes = get_class_names(names_file)
        self.class_indices = {
            self._classes[index]: index for index in range(len(self._classes))
        }

        last_layer_id: int = self._network.getLayerId(self._layer_names[-1])
        self.last_layer = self._network.getLayer(last_layer_id)
        self._logger.info("Completed configuration")

    @property
    def classes(self) -> list[ClassNameID]:
        """Get the list of class names for the model

        :return: Class names list
        """
        return self._classes

    def _post_process_single_frame_dnn_output(
        self,
        frame: Frame,
        single_frame_dnn_output: DNNOutput,
        conf_thr: float,
        nms_thr: float,
    ) -> list[DetectedObject]:
        """Extract the detected objects from a single frame DNN output

        :param frame: The input frame for the DNN
        :param single_frame_dnn_output: DNN output for a single frame
        :param conf_thr: Confidence threshold
        :param nms_thr: Non-maximum suppression threshold
        :return: A list of the detected objects for the given single image frame DNN
            output
        :raises NotImplementedError: If the output DNN layer is not implemented
        """
        frame_height: int
        frame_width: int
        last_layer_type: str = self.last_layer.type
        class_ids: DNNClassIDs = np.array([], dtype=int)
        confidences: DNNObjectConfidences = np.array([], dtype=float)
        boxes = np.zeros((0, 4), dtype=int)
        detected_objects: list[DetectedObject]
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        if last_layer_type == self.LayerTypes.REGION:
            # Take from
            # https://github.com/opencv/opencv/blob/4.x/samples/dnn/object_detection.py
            # Network produces output blob with a shape NxC where N is a number of
            # detected objects and C is a number of classes + 4 where the first 4
            # numbers are [center_x, center_y, width, height]
            dnn_blob: Blob
            detection: np.ndarray
            scores: NDArrayFloat
            class_id: Union[ClassID, NDArrayInt]
            for dnn_blob in single_frame_dnn_output:
                for detection in dnn_blob:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence < conf_thr:
                        continue
                    center_x = FrameIndex(detection[0] * frame_width)
                    center_y = FrameIndex(detection[1] * frame_height)
                    width = FrameIndex(detection[2] * frame_width)
                    height = FrameIndex(detection[3] * frame_height)
                    left = FrameIndex(center_x - width // 2)
                    top = FrameIndex(center_y - height // 2)
                    class_ids = np.append(class_ids, ClassID(class_id))
                    confidences = np.append(confidences, float(confidence))
                    new_box = (left, top, width, height)
                    boxes = np.vstack((boxes, new_box))
        elif last_layer_type == self.LayerTypes.DETECTION_OUTPUT:
            # Implementation at
            # https://github.com/opencv/opencv/blob/4.x/samples/dnn/object_detection.py
            self._logger.exception(
                "Please implement the %s layer type", self.LayerTypes.DETECTION_OUTPUT
            )
            raise NotImplementedError
        else:
            self._logger.exception(
                'Unknown output layer type: "%s"', self.last_layer.type
            )
            raise NotImplementedError

        if (
            len(self.out_layer_names) > 1
            or last_layer_type == self.LayerTypes.REGION
            and self.backend != cv.dnn.DNN_BACKEND_OPENCV  # pylint: disable=no-member
        ):
            nms_suppressed_indices: NDArrayInt = np.array([], dtype=int)
            unique_classes = set(class_ids)
            class_indices: NDArrayInt
            nms_indices: NDArrayInt
            for uniq_cl in unique_classes:
                class_indices = np.where(class_ids == uniq_cl)[0]
                conf: Union[NDArrayFloat, float] = confidences[class_indices]
                box: list = boxes[class_indices].tolist()
                nms_indices = cv.dnn.NMSBoxes(  # pylint: disable=no-member
                    box, conf, conf_thr, nms_thr
                )
                if len(nms_indices) > 0 and len(nms_indices.shape) == 2:
                    nms_indices = nms_indices[:, 0]
                nms_suppressed_indices = np.append(
                    nms_suppressed_indices, class_indices[nms_indices]
                )
        else:
            nms_suppressed_indices = np.arange(0, len(class_ids))

        # Allocate a list of detected objects and update it below with the actual
        # attributes
        temp_detected_objects: list[Optional[DetectedObject]] = [
            None,
        ] * len(nms_suppressed_indices)
        detected_objects_index: int
        nms_index: int
        for detected_objects_index, nms_index in enumerate(nms_suppressed_indices):
            det_obj_class_id = class_ids[nms_index]
            det_obj_conf = confidences[nms_index]
            det_obj_box = boxes[nms_index]
            det_obj = DetectedObject(
                class_id=det_obj_class_id,
                class_name=self._classes[det_obj_class_id],
                conf=det_obj_conf,
                bbox_left=det_obj_box[0],
                bbox_top=det_obj_box[1],
                bbox_width=det_obj_box[2],
                bbox_height=det_obj_box[3],
            )
            temp_detected_objects[detected_objects_index] = det_obj
        detected_objects = temp_detected_objects
        return detected_objects

    def process_frames(  # pylint: disable=arguments-differ
        self,
        frames: Sequence[Frame],
        conf_thr: float,
        nms_thr: float,
    ) -> list[list[DetectedObject]]:
        """Process a sequence of image frames as a batch

        :param frames: The frames
        :param conf_thr: Confidence threshold
        :param nms_thr: Non-maximum suppression threshold
        :return: A list of lists where each member list is the detected objects in the
            same order as the given frames
        """
        blob = frames2blob(frames, out_size=self.out_size)
        self._network.setInput(blob)
        dnn_outputs = self._network.forward(self.out_layer_names)
        per_frame_detected_objects: list[list[DetectedObject]] = []

        post_processed_dnn_output: list[DetectedObject]
        single_frame_dnn_output: DNNOutput
        for index, frame in enumerate(frames):
            single_frame_dnn_output = (
                convert_multiple_frame_dnn_output_to_single_frame_dnn_output(
                    dnn_outputs, index
                )
            )
            post_processed_dnn_output = self._post_process_single_frame_dnn_output(
                frame, single_frame_dnn_output, conf_thr, nms_thr
            )
            per_frame_detected_objects.append(post_processed_dnn_output)

        return per_frame_detected_objects

    def process_frame(  # pylint: disable=arguments-differ
        self,
        frame: Frame,
        conf_thr: float,
        nms_thr: float,
    ) -> list[DetectedObject]:
        """
        Process an image frame in the YOLO model

        :param frame: An image frame
        :param conf_thr: Confidence threshold
        :param nms_thr: Non-maximum suppression threshold
        :return: A list of the detected objects for the given image frame
        """
        blob = frame2blob(frame, self.out_size)
        self._network.setInput(blob)
        single_frame_dnn_output: DNNOutput = self._network.forward(self.out_layer_names)
        return self._post_process_single_frame_dnn_output(
            frame, single_frame_dnn_output, conf_thr, nms_thr
        )

    def process_image(  # pylint: disable=arguments-differ
        self,
        image: PathLike,
        conf_thr: float,
        nms_thr: float,
    ) -> list[DetectedObject]:
        """
        Wrapper for `intrusiondet.dnn.yolo.YoloModel.process_frame` method for a single
        image

        :param image: Path to an image file
        :param conf_thr: Confidence threshold
        :param nms_thr: Non-maximum suppression threshold
        :return: list of DetectedObject classes
        :raises IOError: If the image frame is None
        :see: `YoloModel.process_frame`
        """
        frame = get_frame(image)
        if frame is None:
            self._logger.exception("Unable get frame from image %s", image)
            raise IOError
        return self.process_frame(frame, conf_thr, nms_thr)

    def get_class_index(self, class_name: ClassNameID) -> Optional[ClassID]:
        """Get the index for a class name

        :param class_name: The YOLO class name string
        :return: integer mapping for the string name
        """
        return self.class_indices.get(class_name)


def read_config(
    filename: str,
) -> list[dict[str, dict[str, Union[str, int, float, list[IntOrFloat, ...]]]]]:
    """Parse the YOLO configuration file and return a list of dictionaries

    :param filename: The path the file
    :return: list of dictionaries
    """
    file_path = Path(filename)
    if not file_path.is_file():
        logging.getLogger().exception("The input %s does NOT exist", filename)
        raise IOError
    yolo_config = []
    local_ref_dict: dict
    with file_path.open(encoding="utf-8") as file_ptr:
        file_ptr: TextIOWrapper
        config_lines: list[str] = file_ptr.readlines()
        for line in config_lines:
            line = line.strip()
            if not line:
                continue
            if line[0] == "#":
                continue
            if "[" in line and "]" in line:
                section_name = line.strip("[").strip("]")
                new_dict = {section_name: {}}
                yolo_config.append(new_dict)
                local_ref_dict = yolo_config[-1][section_name]
                continue
            keyword: str
            args: Union[str, int, float, list[IntOrFloat, ...]]
            keyword_args = line.split("=")
            keyword, args = map(str.strip, keyword_args)
            if "," in args:
                try:
                    arg_list = []
                    for num_str in map(str.strip, args.split(",")):
                        if "." in num_str:
                            arg_list.append(float(num_str))
                        else:
                            arg_list.append(int(num_str))
                except ValueError:
                    logging.getLogger().error(
                        "Improperly formatted list %s, trying to recover", str(args)
                    )
                    arg_list.clear()
                local_ref_dict[keyword] = arg_list
            elif any(str(num) in args for num in np.arange(0, 10)):
                if "." in args:
                    tmp_num = float(args)
                else:
                    tmp_num = int(args)
                local_ref_dict[keyword] = tmp_num
            else:
                local_ref_dict[keyword] = args

    return yolo_config
