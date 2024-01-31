from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
)
from inference.core.models.base import PreprocessReturnMetadata
from typing import Tuple, Any
from inference.core.env import FIX_BATCH_SIZE, MAX_BATCH_SIZE 
import numpy as np

class YOLONASObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    box_format = "xyxy"
    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLOv8 model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

    def preprocess(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        fix_batch_size: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        """Preprocesses an object detection inference request.

        Args:
            request (ObjectDetectionInferenceRequest): The request object containing images.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: Preprocessed image inputs and corresponding dimensions.
        """
        img_in, img_dims = self.load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
            cast_to_32=False
        )

        if self.batching_enabled:
            batch_padding = 0
            if FIX_BATCH_SIZE or fix_batch_size:
                if MAX_BATCH_SIZE == float("inf"):
                    self.logger.warn(
                        "Requested fix_batch_size but MAX_BATCH_SIZE is not set. Using dynamic batching."
                    )
                    batch_padding = 0
                else:
                    batch_padding = MAX_BATCH_SIZE - img_in.shape[0]
            if batch_padding < 0:
                raise ValueError(
                    f"Requested fix_batch_size but passed in {img_in.shape[0]} images "
                    f"when the model's batch size is {MAX_BATCH_SIZE}\n"
                    f"Consider turning off fix_batch_size, changing `MAX_BATCH_SIZE` in"
                    f"your inference server config, or passing at most {MAX_BATCH_SIZE} images at a time"
                )
            width_remainder = img_in.shape[2] % 32
            height_remainder = img_in.shape[3] % 32
            if width_remainder > 0:
                width_padding = 32 - (img_in.shape[2] % 32)
            else:
                width_padding = 0
            if height_remainder > 0:
                height_padding = 32 - (img_in.shape[3] % 32)
            else:
                height_padding = 0
            img_in = np.pad(
                img_in,
                ((0, batch_padding), (0, 0), (0, width_padding), (0, height_padding)),
                "constant",
            )
        
        assert img_in.dtype == np.uint8

        return img_in, PreprocessReturnMetadata(
            {
                "img_dims": img_dims,
                "disable_preproc_static_crop": disable_preproc_static_crop,
            }
        )
    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class confidence scores.
        """
        predictions = self.onnx_session.run(None, {self.input_name: img_in})
        boxes = predictions[0]
        class_confs = predictions[1]
        confs = np.expand_dims(np.max(class_confs, axis=2), axis=2)
        predictions = np.concatenate([boxes, confs, class_confs], axis=2)
        return (predictions,)
