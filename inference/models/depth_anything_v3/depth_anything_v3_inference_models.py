# Copyright (c) 2025 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from inference_models.models.depth_anything_v3.depth_anything_v3_torch import (
    DepthAnythingV3Torch,
)


class InferenceModelsDepthAnythingV3Adapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "depth-estimation"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: DepthAnythingV3Torch = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            **kwargs,
        )

        # Precompute viridis colormap lookup table once during initialization
        cmap = plt.get_cmap("viridis")
        self._viridis_lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)

    def preprocess(self, image: Any, **kwargs):
        if isinstance(image, list):
            raise ValueError("DepthAnythingV3 does not support batched inference.")
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
        colored_depth = self._viridis_lut[depth_for_viz]

        # Convert numpy array to WorkflowImageData

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
        pass
