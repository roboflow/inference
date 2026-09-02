from typing import Any, List

import torch

from inference.core.entities.responses import (
    InferenceResponseImage,
    LMMInferenceResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr
from inference_models import AutoModel, PreProcessingOverrides
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner


class InferenceModelsCosmos3ReasonerAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "lmm"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: Cosmos3EdgeReasoner = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, prompt: str = "", **kwargs):
        """One image, or a list of images that is one clip.

        The model does not batch independent images; a list is the consecutive frames of a
        video window, and ``video_fps`` (the rate they were sampled at) says how far apart
        they are in time. The response dims are those of the first frame.
        """
        disable_preproc_auto_orient = kwargs.get("disable_preproc_auto_orient", False)
        video_fps = kwargs.pop("video_fps", None)
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if isinstance(image, list):
            if video_fps is None:
                raise ValueError(
                    "A list of images is the frames of one clip for this model; pass "
                    "video_fps, the rate the frames were sampled at."
                )
            if len(image) == 0:
                raise ValueError("A clip needs at least one frame.")
            frames = [
                load_image_bgr(
                    frame, disable_preproc_auto_orient=disable_preproc_auto_orient
                )
                for frame in image
            ]
            input_shape = PreprocessReturnMetadata(
                {"image_dims": frames[0].shape[:2][::-1]}
            )
            return (
                self._model.pre_process_generation(
                    frames,
                    prompt,
                    as_video=True,
                    video_fps=video_fps,
                    **mapped_kwargs,
                ),
                input_shape,
            )
        np_image = load_image_bgr(
            image, disable_preproc_auto_orient=disable_preproc_auto_orient
        )
        input_shape = PreprocessReturnMetadata({"image_dims": np_image.shape[:2][::-1]})
        return (
            self._model.pre_process_generation(np_image, prompt, **mapped_kwargs),
            input_shape,
        )

    def predict(self, inputs, **kwargs) -> torch.Tensor:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.generate(inputs, **mapped_kwargs)

    def postprocess(
        self,
        predictions: torch.Tensor,
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[LMMInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        result = self._model.post_process_generation(predictions, **mapped_kwargs)[0]
        return [
            LMMInferenceResponse(
                response=result,
                image=InferenceResponseImage(
                    width=preprocess_return_metadata["image_dims"][0],
                    height=preprocess_return_metadata["image_dims"][1],
                ),
            )
        ]

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        pass
