from time import perf_counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderCompareRequest,
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderInferenceRequest,
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.entities.responses.perception_encoder import (
    PerceptionEncoderCompareResponse,
    PerceptionEncoderEmbeddingResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    CLIP_MAX_BATCH_SIZE,
    DEVICE,
    PERCEPTION_ENCODER_MODEL_ID,
)
from inference.core.models.base import Model
from inference.core.models.inference_models_adapters import (
    get_extra_weights_provider_headers,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image_bgr
from inference.core.utils.postprocess import cosine_similarity
from inference_models import AutoModel
from inference_models.models.perception_encoder.perception_encoder_pytorch import (
    PerceptionEncoderTorch,
)


class InferenceModelsPerceptionEncoderAdapter(Model):
    """Roboflow Perception Encoder model implementation.

    This class is responsible for handling the Percpetion Encoder model, including
    loading the model, preprocessing the input, and performing inference.
    """

    def __init__(
        self, model_id: str = PERCEPTION_ENCODER_MODEL_ID, api_key: str = None, **kwargs
    ):
        super().__init__()
        if model_id.startswith("perception_encoder/"):
            model_id = model_id.replace("perception_encoder/", "perception-encoder/")

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "embedding"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: PerceptionEncoderTorch = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            extra_weights_provider_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def preproc_image(self, image: InferenceRequestImage) -> np.ndarray:
        """Preprocesses an inference request image."""
        return load_image_bgr(image)

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[torch.Tensor, PreprocessReturnMetadata]:
        return self.preproc_image(image), PreprocessReturnMetadata({})

    def compare(
        self,
        subject: Any,
        prompt: Any,
        subject_type: str = "image",
        prompt_type: Union[str, List[str], Dict[str, Any]] = "text",
        **kwargs,
    ) -> Union[List[float], Dict[str, float]]:
        """
        Compares the subject with the prompt to calculate similarity scores.

        Args:
            subject (Any): The subject data to be compared. Can be either an image or text.
            prompt (Any): The prompt data to be compared against the subject. Can be a single value (image/text), list of values, or dictionary of values.
            subject_type (str, optional): Specifies the type of the subject data. Must be either "image" or "text". Defaults to "image".
            prompt_type (Union[str, List[str], Dict[str, Any]], optional): Specifies the type of the prompt data. Can be "image", "text", list of these types, or a dictionary containing these types. Defaults to "text".
            **kwargs: Additional keyword arguments.

        Returns:
            Union[List[float], Dict[str, float]]: A list or dictionary containing cosine similarity scores between the subject and prompt(s).
        """
        if subject_type == "image":
            subject_embeddings = self.embed_image(subject)
        elif subject_type == "text":
            subject_embeddings = self.embed_text(subject)
        else:
            raise ValueError(
                f"subject_type must be either 'image' or 'text', but got {subject_type}"
            )

        if isinstance(prompt, dict) and not ("type" in prompt and "value" in prompt):
            prompt_keys = prompt.keys()
            prompt = [prompt[k] for k in prompt_keys]
            prompt_obj = "dict"
        else:
            if not isinstance(prompt, list):
                prompt = [prompt]
            prompt_obj = "list"

        if len(prompt) > CLIP_MAX_BATCH_SIZE:
            raise ValueError(
                f"The maximum number of prompts that can be compared at once is {CLIP_MAX_BATCH_SIZE}"
            )

        if prompt_type == "image":
            prompt_embeddings = self.embed_image(prompt)
        elif prompt_type == "text":
            prompt_embeddings = self.embed_text(prompt)
        else:
            raise ValueError(
                f"prompt_type must be either 'image' or 'text', but got {prompt_type}"
            )

        similarities = [
            cosine_similarity(subject_embeddings, p) for p in prompt_embeddings
        ]

        if prompt_obj == "dict":
            similarities = dict(zip(prompt_keys, similarities))

        return similarities

    def make_compare_response(
        self, similarities: Union[List[float], Dict[str, float]]
    ) -> PerceptionEncoderCompareResponse:
        """Creates a PerceptionEncoderCompareResponse object from the provided similarity data."""
        response = PerceptionEncoderCompareResponse(similarity=similarities)
        return response

    def embed_image(
        self,
        image: Any,
        **kwargs,
    ) -> np.ndarray:
        """
        Embeds an image or a list of images using the PE-CLIP model.

        Args:
            image (Any): The image or list of images to be embedded.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The embeddings of the image(s) as a numpy array.
        """
        if isinstance(image, list):
            if len(image) > CLIP_MAX_BATCH_SIZE:
                raise ValueError(
                    f"The maximum number of images that can be embedded at once is {CLIP_MAX_BATCH_SIZE}"
                )
            img_in = [self.preproc_image(i) for i in image]
        else:
            img_in = [self.preproc_image(image)]

        return self._model.embed_images(img_in).cpu().numpy()

    def embed_text(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ) -> np.ndarray:
        """
        Embeds a text or a list of texts using the PE-CLIP model.

        Args:
            text (Union[str, List[str]]): The text string or list of text strings to be embedded.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The embeddings of the text or texts as a numpy array.
        """
        if isinstance(text, list):
            texts = text
        else:
            texts = [text]
        if len(texts) > CLIP_MAX_BATCH_SIZE:
            raise ValueError(
                f"The maximum number of texts that can be embedded at once is {CLIP_MAX_BATCH_SIZE}"
            )
        return self._model.embed_text(texts).cpu().numpy()

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Predict embeddings for an input tensor.

        Args:
            img_in (torch.Tensor): The input tensor to get embeddings for.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[np.ndarray]: A tuple containing the embeddings as a numpy array.
        """
        embeddings = self._model.embed_images(img_in).cpu().numpy()
        return (embeddings,)

    def make_embed_image_response(
        self, embeddings: np.ndarray
    ) -> PerceptionEncoderEmbeddingResponse:
        """Converts the given embeddings into a PerceptionEncoderEmbeddingResponse object."""
        response = PerceptionEncoderEmbeddingResponse(embeddings=embeddings.tolist())
        return response

    def make_embed_text_response(
        self, embeddings: np.ndarray
    ) -> PerceptionEncoderEmbeddingResponse:
        """Converts the given text embeddings into a PerceptionEncoderEmbeddingResponse object."""
        response = PerceptionEncoderEmbeddingResponse(embeddings=embeddings.tolist())
        return response

    def infer_from_request(
        self, request: PerceptionEncoderInferenceRequest
    ) -> PerceptionEncoderEmbeddingResponse:
        """Routes the request to the appropriate inference function."""
        t1 = perf_counter()
        if isinstance(request, PerceptionEncoderImageEmbeddingRequest):
            infer_func = self.embed_image
            make_response_func = self.make_embed_image_response
        elif isinstance(request, PerceptionEncoderTextEmbeddingRequest):
            infer_func = self.embed_text
            make_response_func = self.make_embed_text_response
        elif isinstance(request, PerceptionEncoderCompareRequest):
            infer_func = self.compare
            make_response_func = self.make_compare_response
        else:
            raise ValueError(
                f"Request type {type(request)} is not a valid PerceptionEncoderInferenceRequest"
            )
        data = infer_func(**request.dict())
        response = make_response_func(data)
        response.time = perf_counter() - t1
        return response

    def make_response(self, embeddings, *args, **kwargs) -> InferenceResponse:
        return [self.make_embed_image_response(embeddings)]

    def postprocess(
        self,
        predictions: Tuple[np.ndarray],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        return [self.make_embed_image_response(predictions[0])]

    def infer(self, image: Any, **kwargs) -> Any:
        """Embeds an image"""
        return super().infer(image, **kwargs)
