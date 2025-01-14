from typing import Tuple

import numpy as np

import torch

import onnxruntime as ort

from typing import Union

import time

from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)
from inference.core.utils.onnx import run_session_via_iobinding


class YOLOv8ObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection model (Implements an object detection specific infer method).

    This class is responsible for performing object detection using the YOLOv8 model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs object detection on the given image using the ONNX session.
    """

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLOv8 model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

    def predict(self, img_in: Union[np.ndarray, torch.Tensor], **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class confidence scores.
        """
        # if isinstance(img_in, np.ndarray):
        #     img_in = torch.from_numpy(img_in).to(device="cuda").contiguous()
        # elif isinstance(img_in, torch.Tensor):
        #     img_in = img_in.to(device="cuda").contiguous()
        # else:
        #     raise ValueError(f"Unsupported input type: {type(img_in)}")

        # print(img_in.device)

        # device_type, device_id = img_in.device.type, img_in.device.index
        
        # binding = self.onnx_session.io_binding()
        # binding.bind_input(
        #     name=self.input_name,
        #     device_type=device_type,
        #     device_id=device_id,
        #     element_type=np.float16,
        #     shape=img_in.shape,
        #     buffer_ptr=img_in.data_ptr(),
        # )
        
        # # get the metadata for the first and only output
        # print(self.onnx_session.get_outputs())
        # output_metadata = self.onnx_session.get_outputs()[0]
        # print(output_metadata)
        # print(output_metadata.name)
        # predictions = torch.empty(output_metadata.shape, dtype=torch.float16, device=img_in.device).contiguous()
        # binding.bind_output(
        #     name=output_metadata.name,
        #     device_type=device_type,
        #     device_id=device_id,
        #     element_type=np.float16,
        #     shape=output_metadata.shape,
        #     buffer_ptr=predictions.data_ptr(),
        # )

        # # t0 = time.time()

        # self.onnx_session.run_with_iobinding(binding)

        # # t1 = time.time()
        # # print(f"Time taken: {t1 - t0} seconds")

        # predictions = predictions.cpu().numpy()

        predictions = run_session_via_iobinding(self.onnx_session, self.input_name, img_in)

        # predictions = self.onnx_session.run(None, {self.input_name: img_in})
        # print(predictions)
        # predictions = predictions[0]
        predictions = predictions.transpose(0, 2, 1)
        boxes = predictions[:, :, :4]
        class_confs = predictions[:, :, 4:]
        confs = np.expand_dims(np.max(class_confs, axis=2), axis=2)
        predictions = np.concatenate([boxes, confs, class_confs], axis=2)
        # print(predictions[:, :10])
        return (predictions,)
