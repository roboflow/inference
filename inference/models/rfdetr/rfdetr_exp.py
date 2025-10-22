from threading import Lock
from typing import Any, List, Tuple, Union

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.models.roboflow import RoboflowInferenceModel
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.image_utils import load_image_rgb
from inference.core.logger import logger


class RFDetrExperimentalModel(RoboflowInferenceModel):
    """Adapter for RF-DETR using inference_exp AutoModel backend.

    This class wraps an inference_exp AutoModel to present the same interface
    as legacy models in the inference server.
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs) -> None:
        # Avoid legacy download/initialization path
        super().__init__(model_id, api_key=api_key, load_weights=False, **kwargs)
        self._state_lock = Lock()

        logger.info(f"Initialized RFDetrExperimentalModel for model_id: {model_id}")
        # Lazy import to avoid hard dependency if flag disabled
        from inference_exp import AutoModel  # type: ignore

        # Load experimental model; API key taken from self.api_key
        self._exp_model = AutoModel.from_pretrained(
            model_id_or_path=model_id, api_key=self.api_key
        )

        # Propagate class names for response formatting
        try:
            self.class_names = list(self._exp_model.class_names)
        except Exception:
            self.class_names = []

    def infer(
        self,
        image: Any,
        class_agnostic_nms: bool = False,
        confidence: float = 0.4,
        iou_threshold: float = 0.5,
        max_candidates: int = 3000,
        max_detections: int = 300,
        disable_preproc_auto_orient: bool = False,
        return_image_dims: bool = False,
        **kwargs,
    ) -> Union[
        ObjectDetectionInferenceResponse,
        List[ObjectDetectionInferenceResponse],
    ]:
        logger.info(
            f"RFDetrExperimentalModel Running inference for {len(image) if isinstance(image, list) else 1} images"
        )
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_rgb(v, disable_preproc_auto_orient=disable_preproc_auto_orient)
            for v in images
        ]

        detections_list = self._exp_model(np_images, threshold=float(confidence))

        responses: List[ObjectDetectionInferenceResponse] = []
        for np_img, det in zip(np_images, detections_list):
            H, W = np_img.shape[0], np_img.shape[1]

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

        return responses if is_batch else responses[0]

    def infer_from_request(
        self, request
    ) -> Union[
        List[ObjectDetectionInferenceResponse], ObjectDetectionInferenceResponse
    ]:
        with self._state_lock:
            # request may be a Pydantic model; prefer model_dump when available
            payload = (
                request.model_dump()
                if hasattr(request, "model_dump")
                else request.dict()
            )
            return self.infer(**payload)
