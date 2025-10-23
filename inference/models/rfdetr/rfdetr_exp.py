from threading import Lock
from typing import Any, List, Union

import numpy as np
from time import perf_counter

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.models.base import Model
from inference.models.aliases import resolve_roboflow_model_alias

from inference.core.utils.image_utils import load_image_rgb
from inference.core.logger import logger


from inference.core.env import API_KEY

import numpy as np
from typing import Generic, List, Optional, Tuple, Union
from inference_exp.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)
from inference_exp.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
)


class InferenceExpObjectDetectionModelAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "object-detection"

        # Lazy import to avoid hard dependency if flag disabled
        from inference_exp import AutoModel  # type: ignore

        self._exp_model: ObjectDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id, api_key=self.api_key
        )
        self._exp_model.optimize_for_inference()
        self.class_names = list(self._exp_model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_rgb(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        return self._exp_model.pre_process(np_images, **kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._exp_model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[Detections]:
        detections_list = self._exp_model.post_process(
            predictions, preprocess_return_metadata, **kwargs
        )

        responses: List[ObjectDetectionInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()

            predictions: List[ObjectDetectionPrediction] = []

            for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                predictions.append(
                    ObjectDetectionPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                    )
                )

            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )

        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass


class RFDetrExperimentalModel(InferenceExpObjectDetectionModelAdapter):
    """Adapter for RF-DETR using inference_exp AutoModel backend.

    This class wraps an inference_exp AutoModel to present the same interface
    as legacy models in the inference server.
    """

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return {
            "threshold": kwargs.get("confidence"),
        }
