from typing import Any, List, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import torch

from inference.core.entities.responses import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
)
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models import AutoModel
from inference_models.models.depth_anything_v2.depth_anything_v2_hf import (
    DepthAnythingV2HF,
)
import cv2


class InferenceModelsDepthAnythingV2Adapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "depth-estimation"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: DepthAnythingV2HF = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def preprocess(self, image: Any, **kwargs):
        if isinstance(image, list):
            raise ValueError("DepthAnythingV2 does not support batched inference.")

        np_image = load_image_bgr(
            image,
            disable_preproc_auto_orient=kwargs.get(
                "disable_preproc_auto_orient", False
            ),
        )
        return np_image, PreprocessReturnMetadata(
            {"image_dims": (np_image.shape[1], np_image.shape[0])}
        )

    def predict(self, inputs: np.ndarray, **kwargs) -> Tuple[dict]:
        predictions = self._model(inputs)[0]
        depth_map = predictions.to(torch.float32).cpu().numpy()
        # Normalize depth values
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max == depth_min:
            raise ValueError("Depth map has no variation (min equals max)")
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)

        # Create visualization
        depth_for_viz = (normalized_depth * 255.0).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_for_viz, cv2.COLORMAP_VIRIDIS)
        colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)

        # Convert numpy array to WorkflowImageData
        parent_metadata = ImageParentMetadata(parent_id=f"{uuid4()}")
        colored_depth_image = WorkflowImageData(
            numpy_image=colored_depth, parent_metadata=parent_metadata
        )
        result = {
            "image": colored_depth_image,
            "normalized_depth": normalized_depth,
        }
        return (result,)

    def postprocess(
        self,
        predictions: torch.Tensor,
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[LMMInferenceResponse]:
        text = predictions[0]
        image_dims = preprocess_return_metadata["image_dims"]
        response = LMMInferenceResponse(
            response=text,
            image=InferenceResponseImage(width=image_dims[0], height=image_dims[1]),
        )
        return [response]

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass
