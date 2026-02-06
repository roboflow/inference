from time import perf_counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import onnxruntime

from inference.core.entities.requests.clip import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipInferenceRequest,
    ClipTextEmbeddingRequest,
)
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.responses.clip import (
    ClipCompareResponse,
    ClipEmbeddingResponse,
)
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    CLIP_MAX_BATCH_SIZE,
    CLIP_MODEL_ID,
)
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image_bgr
from inference.core.utils.postprocess import cosine_similarity
from inference_models import AutoModel
from inference_models.models.clip.clip_onnx import ClipOnnx
from inference_models.models.clip.clip_pytorch import ClipTorch


class InferenceModelsClipAdapter(Model):
    """Roboflow ONNX ClipModel model.

    This class is responsible for handling the ONNX ClipModel model, including
    loading the model, preprocessing the input, and performing inference.

    Attributes:
        visual_onnx_session (onnxruntime.InferenceSession): ONNX Runtime session for visual inference.
        textual_onnx_session (onnxruntime.InferenceSession): ONNX Runtime session for textual inference.
        resolution (int): The resolution of the input image.
        clip_preprocess (function): Function to preprocess the image.
    """

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        api_key: str = None,
        **kwargs,
    ):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        self.task_type = "embedding"
        self._model: Union[ClipOnnx, ClipTorch] = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            **kwargs,
        )

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
            Union[List[float], Dict[str, float]]: A list or dictionary containing cosine similarity scores between the subject and prompt(s). If prompt is a dictionary, returns a dictionary with keys corresponding to the original prompt dictionary's keys.

        Raises:
            ValueError: If subject_type or prompt_type is neither "image" nor "text".
            ValueError: If the number of prompts exceeds the maximum batch size.
        """

        if subject_type == "image":
            subject_embeddings = self.embed_image(subject)
        elif subject_type == "text":
            subject_embeddings = self.embed_text(subject)
        else:
            raise ValueError(
                "subject_type must be either 'image' or 'text', but got {request.subject_type}"
            )

        if isinstance(prompt, dict) and not ("type" in prompt and "value" in prompt):
            prompt_keys = prompt.keys()
            prompt = [prompt[k] for k in prompt_keys]
            prompt_obj = "dict"
        else:
            prompt = prompt
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
                "prompt_type must be either 'image' or 'text', but got {request.prompt_type}"
            )

        similarities = [
            cosine_similarity(subject_embeddings, p) for p in prompt_embeddings
        ]

        if prompt_obj == "dict":
            similarities = dict(zip(prompt_keys, similarities))

        return similarities

    def make_compare_response(
        self, similarities: Union[List[float], Dict[str, float]]
    ) -> ClipCompareResponse:
        """
        Creates a ClipCompareResponse object from the provided similarity data.

        Args:
            similarities (Union[List[float], Dict[str, float]]): A list or dictionary containing similarity scores.

        Returns:
            ClipCompareResponse: An instance of the ClipCompareResponse with the given similarity scores.

        Example:
            Assuming `ClipCompareResponse` expects a dictionary of string-float pairs:

            >>> make_compare_response({"image1": 0.98, "image2": 0.76})
            ClipCompareResponse(similarity={"image1": 0.98, "image2": 0.76})
        """
        response = ClipCompareResponse(similarity=similarities)
        return response

    def embed_image(
        self,
        image: Any,
        **kwargs,
    ) -> np.ndarray:
        """
        Embeds an image or a list of images using the Clip model.

        Args:
            image (Any): The image or list of images to be embedded. Image can be in any format that is acceptable by the preproc_image method.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The embeddings of the image(s) as a numpy array.

        Raises:
            ValueError: If the number of images in the list exceeds the maximum batch size.

        Notes:
            The function measures performance using perf_counter and also has support for ONNX session to get embeddings.
        """
        t1 = perf_counter()

        if isinstance(image, list):
            if len(image) > CLIP_MAX_BATCH_SIZE:
                raise ValueError(
                    f"The maximum number of images that can be embedded at once is {CLIP_MAX_BATCH_SIZE}"
                )
            imgs = [self.preproc_image(i) for i in image]
            img_in = np.concatenate(imgs, axis=0)
        else:
            img_in = self.preproc_image(image)
        embeddings = self._model.embed_images(images=img_in)
        return embeddings.cpu().numpy()

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        embeddings = self._model.embed_images(images=img_in)
        return (embeddings.cpu().numpy(),)

    def make_embed_image_response(
        self, embeddings: np.ndarray
    ) -> ClipEmbeddingResponse:
        """
        Converts the given embeddings into a ClipEmbeddingResponse object.

        Args:
            embeddings (np.ndarray): A numpy array containing the embeddings for an image or images.

        Returns:
            ClipEmbeddingResponse: An instance of the ClipEmbeddingResponse with the provided embeddings converted to a list.

        Example:
            >>> embeddings_array = np.array([[0.5, 0.3, 0.2], [0.1, 0.9, 0.0]])
            >>> make_embed_image_response(embeddings_array)
            ClipEmbeddingResponse(embeddings=[[0.5, 0.3, 0.2], [0.1, 0.9, 0.0]])
        """
        response = ClipEmbeddingResponse(embeddings=embeddings.tolist())

        return response

    def embed_text(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ) -> np.ndarray:
        """
        Embeds a text or a list of texts using the Clip model.

        Args:
            text (Union[str, List[str]]): The text string or list of text strings to be embedded.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The embeddings of the text or texts as a numpy array.

        Raises:
            ValueError: If the number of text strings in the list exceeds the maximum batch size.

        Notes:
            The function utilizes an ONNX session to compute embeddings and measures the embedding time with perf_counter.
        """
        if isinstance(text, list):
            texts = text
        else:
            texts = [text]
        embeddings = self._model.embed_text(texts=texts)
        return embeddings.cpu().numpy()

    def make_embed_text_response(self, embeddings: np.ndarray) -> ClipEmbeddingResponse:
        """
        Converts the given text embeddings into a ClipEmbeddingResponse object.

        Args:
            embeddings (np.ndarray): A numpy array containing the embeddings for a text or texts.

        Returns:
            ClipEmbeddingResponse: An instance of the ClipEmbeddingResponse with the provided embeddings converted to a list.

        Example:
            >>> embeddings_array = np.array([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
            >>> make_embed_text_response(embeddings_array)
            ClipEmbeddingResponse(embeddings=[[0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
        """
        response = ClipEmbeddingResponse(embeddings=embeddings.tolist())
        return response

    def infer_from_request(
        self, request: ClipInferenceRequest
    ) -> ClipEmbeddingResponse:
        """Routes the request to the appropriate inference function.

        Args:
            request (ClipInferenceRequest): The request object containing the inference details.

        Returns:
            ClipEmbeddingResponse: The response object containing the embeddings.
        """
        t1 = perf_counter()
        if isinstance(request, ClipImageEmbeddingRequest):
            infer_func = self.embed_image
            make_response_func = self.make_embed_image_response
        elif isinstance(request, ClipTextEmbeddingRequest):
            infer_func = self.embed_text
            make_response_func = self.make_embed_text_response
        elif isinstance(request, ClipCompareRequest):
            infer_func = self.compare
            make_response_func = self.make_compare_response
        else:
            raise ValueError(
                f"Request type {type(request)} is not a valid ClipInferenceRequest"
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
        """Embeds an image
        - image:
            can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
        """
        return super().infer(image, **kwargs)

    def preproc_image(self, image: InferenceRequestImage) -> np.ndarray:
        """Preprocesses an inference request image.

        Args:
            image (InferenceRequestImage): The object containing information necessary to load the image for inference.

        Returns:
            np.ndarray: A numpy array of the preprocessed image pixel data.
        """
        return load_image_bgr(image)

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        return self.preproc_image(image), PreprocessReturnMetadata({})
