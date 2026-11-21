from time import perf_counter
from typing import List, Union

import numpy as np
import torch

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.embeddings import (
    EmbeddingInfo,
    ImageEmbeddingResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.exceptions import ModelDeploymentNotSupportedError
from inference.core.models.base import Model
from inference.core.models.embeddings import (
    FEATURE_VECTOR_OUTPUT_TYPE,
    load_embedding_request_images,
    make_embedding_info,
    pretrained_embedding_preprocessing,
    require_feature_vector_output,
)
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.models.aliases import resolve_roboflow_model_alias
from inference_models import AutoModel


class InferenceModelsPretrainedImageEmbeddingAdapter(Model):
    """Image embeddings from pretrained DINOv2, DINOv3 and SigLIP2 backbones.

    The backbones have no task head: they serve feature vectors through
    `POST /infer/embeddings` and the tensor-native embeddings port only.
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()
        require_feature_vector_output(
            kwargs.get("output_type", FEATURE_VECTOR_OUTPUT_TYPE)
        )

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)
        self._embedding_model_id = model_id

        self.task_type = "embedding"
        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = sorted(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )

    def infer_embeddings_from_request(
        self, request: ImageEmbeddingRequest
    ) -> ImageEmbeddingResponse:
        """Embed the request images and serialize the HTTP response.

        Args:
            request: One image or a list, and the requested output type.

        Returns:
            ImageEmbeddingResponse: Ordered vectors and compatibility metadata.

        Raises:
            ModelInputError: If the request asks for logits.
        """
        started = perf_counter()
        require_feature_vector_output(request.output_type)
        images = load_embedding_request_images(request)
        features = self._model.embed_images(
            images, output_type=request.output_type, input_color_format="bgr"
        )
        info = self._embedding_metadata(
            features=features,
            output_type=request.output_type,
            kwargs=request.model_dump(),
        )
        response = ImageEmbeddingResponse(
            embeddings=features.detach().cpu().float().tolist(),
            embedding_info=info,
            time=perf_counter() - started,
            inference_id=request.id,
        )

        return response

    def run_tensor_native_embeddings(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        *,
        input_color_format: str = "bgr",
        output_type: str = FEATURE_VECTOR_OUTPUT_TYPE,
        **kwargs,
    ) -> dict:
        """Embed images while keeping tensor inputs and outputs on device.

        Args:
            images: Materialized tensor or NumPy images.
            input_color_format: Color ordering of the supplied images.
            output_type: Must be the feature vector.
            **kwargs: Preprocessing overrides; only the auto-orient one joins
                the space identity, the classifier-only steps are ignored.

        Returns:
            Batched embedding tensor and compatibility metadata.

        Raises:
            ModelInputError: If logits are requested.
        """
        require_feature_vector_output(output_type)
        features = self._model.embed_images(
            images, output_type=output_type, input_color_format=input_color_format
        )
        info = self._embedding_metadata(
            features=features, output_type=output_type, kwargs=kwargs
        )
        result = {
            "embeddings": features.detach(),
            "embedding_info": info.model_dump(exclude_none=True),
        }

        return result

    def infer_from_request(self, request: InferenceRequest):
        raise ModelDeploymentNotSupportedError(
            f"Model {self._embedding_model_id} only produces image embeddings. "
            f"Use POST /infer/embeddings."
        )

    def _embedding_metadata(
        self, *, features: torch.Tensor, output_type: str, kwargs: dict
    ) -> EmbeddingInfo:
        info = make_embedding_info(
            model_id=self._embedding_model_id,
            feature_info=self._model.get_embedding_info(output_type),
            preprocessing=pretrained_embedding_preprocessing(
                self._model.embedding_preprocessing, kwargs
            ),
            backend=type(self._model).__name__,
            precision=str(features.dtype),
            dimension=features.shape[1],
        )
        return info
