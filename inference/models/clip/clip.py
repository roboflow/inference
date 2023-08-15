from time import perf_counter
from typing import List

import numpy as np
import onnxruntime
from clip import tokenize
from clip.clip import _transform

from inference.core.data_models import (
    ClipCompareRequest,
    ClipCompareResponse,
    ClipEmbeddingResponse,
    ClipImageEmbeddingRequest,
    ClipInferenceRequest,
    ClipTextEmbeddingRequest,
    InferenceRequestImage,
)
from inference.core.env import (
    CLIP_MAX_BATCH_SIZE,
    REQUIRED_ONNX_PROVIDERS,
    TENSORRT_CACHE_PATH,
)
from inference.core.exceptions import OnnxProviderNotAvailable
from inference.core.models.roboflow import OnnxRoboflowCoreModel
from inference.core.utils.postprocess import cosine_similarity


class ClipOnnxRoboflowCoreModel(OnnxRoboflowCoreModel):
    """Roboflow ONNX Clip model.

    This class is responsible for handling the ONNX Clip model, including
    loading the model, preprocessing the input, and performing inference.

    Attributes:
        visual_onnx_session (onnxruntime.InferenceSession): ONNX Runtime session for visual inference.
        textual_onnx_session (onnxruntime.InferenceSession): ONNX Runtime session for textual inference.
        resolution (int): The resolution of the input image.
        clip_preprocess (function): Function to preprocess the image.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the ClipOnnxRoboflowCoreModel with the given arguments and keyword arguments."""

        t1 = perf_counter()
        super().__init__(*args, **kwargs)
        # Create an ONNX Runtime Session with a list of execution providers in priority order. ORT attempts to load providers until one is successful. This keeps the code across devices identical.
        self.log("Creating inference sessions")
        self.visual_onnx_session = onnxruntime.InferenceSession(
            self.cache_file("visual.onnx"),
            providers=[
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": TENSORRT_CACHE_PATH,
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.textual_onnx_session = onnxruntime.InferenceSession(
            self.cache_file("textual.onnx"),
            providers=[
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": TENSORRT_CACHE_PATH,
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        if REQUIRED_ONNX_PROVIDERS:
            available_providers = onnxruntime.get_available_providers()
            for provider in REQUIRED_ONNX_PROVIDERS:
                if provider not in available_providers:
                    raise OnnxProviderNotAvailable(
                        f"Required ONNX Execution Provider {provider} is not availble. Check that you are using the correct docker image on a supported device."
                    )

        self.resolution = self.visual_onnx_session.get_inputs()[0].shape[2]

        self.clip_preprocess = _transform(self.resolution)
        self.log(f"CLIP model loaded in {perf_counter() - t1:.2f} seconds")

    def compare(self, request: ClipCompareRequest) -> ClipCompareResponse:
        """Compares the subject with the prompt using the Clip model.

        Args:
            request (ClipCompareRequest): The request object containing the subject and prompt.

        Returns:
            ClipCompareResponse: The response object containing the similarity score.
        """
        t1 = perf_counter()
        if request.subject_type == "image":
            subject_embed_request = ClipImageEmbeddingRequest(image=request.subject)
            subject_embeddings = self.embed_image(subject_embed_request).embeddings[0]
        elif request.subject_type == "text":
            subject_embed_request = ClipTextEmbeddingRequest(text=request.subject)
            subject_embeddings = self.embed_text(subject_embed_request).embeddings[0]
        else:
            raise ValueError(
                "subject_type must be either 'image' or 'text', but got {request.subject_type}"
            )

        if isinstance(request.prompt, dict):
            prompt_keys = request.prompt.keys()
            prompt = [request.prompt[k] for k in prompt_keys]
            prompt_obj = "dict"
        else:
            prompt = request.prompt
            if not isinstance(prompt, list):
                prompt = [prompt]
            prompt_obj = "list"

        if len(prompt) > CLIP_MAX_BATCH_SIZE:
            raise ValueError(
                f"The maximum number of prompts that can be compared at once is {CLIP_MAX_BATCH_SIZE}"
            )

        if request.prompt_type == "image":
            prompt_embed_request = ClipImageEmbeddingRequest(image=prompt)
            prompt_embeddings = self.embed_image(prompt_embed_request).embeddings
        elif request.prompt_type == "text":
            prompt_embed_request = ClipTextEmbeddingRequest(text=prompt)
            prompt_embeddings = self.embed_text(prompt_embed_request).embeddings
        else:
            raise ValueError(
                "prompt_type must be either 'image' or 'text', but got {request.prompt_type}"
            )

        similarities = [
            cosine_similarity(subject_embeddings, p) for p in prompt_embeddings
        ]

        if prompt_obj == "dict":
            similarities = dict(zip(prompt_keys, similarities))

        response = ClipCompareResponse(
            similarity=similarities, time=perf_counter() - t1
        )

        return response

    def embed_image(self, request: ClipImageEmbeddingRequest) -> ClipEmbeddingResponse:
        """Embeds an image using the Clip model.

        Args:
            request (ClipImageEmbeddingRequest): The request object containing the image.

        Returns:
            ClipEmbeddingResponse: The response object containing the embeddings.
        """
        t1 = perf_counter()

        if isinstance(request.image, list):
            if len(request.image) > CLIP_MAX_BATCH_SIZE:
                raise ValueError(
                    f"The maximum number of images that can be embedded at once is {CLIP_MAX_BATCH_SIZE}"
                )
            imgs = [self.preproc_image(i) for i in request.image]
            img_in = np.concatenate(imgs, axis=0)
        else:
            img_in = self.preproc_image(request.image)

        onnx_input_image = {self.visual_onnx_session.get_inputs()[0].name: img_in}
        embeddings = self.visual_onnx_session.run(None, onnx_input_image)[0]

        response = ClipEmbeddingResponse(
            embeddings=embeddings.tolist(), time=perf_counter() - t1
        )

        return response

    def embed_text(self, request: ClipTextEmbeddingRequest) -> ClipEmbeddingResponse:
        """Embeds a text using the Clip model.

        Args:
            request (ClipTextEmbeddingRequest): The request object containing the text.

        Returns:
            ClipEmbeddingResponse: The response object containing the embeddings.
        """
        t1 = perf_counter()

        if isinstance(request.text, list):
            if len(request.text) > CLIP_MAX_BATCH_SIZE:
                raise ValueError(
                    f"The maximum number of text strings that can be embedded at once is {CLIP_MAX_BATCH_SIZE}"
                )

            texts = request.text
        else:
            texts = [request.text]

        texts = tokenize(texts).numpy().astype(np.int32)

        onnx_input_text = {self.textual_onnx_session.get_inputs()[0].name: texts}
        embeddings = self.textual_onnx_session.run(None, onnx_input_text)[0]

        response = ClipEmbeddingResponse(
            embeddings=embeddings.tolist(), time=perf_counter() - t1
        )

        return response

    def get_infer_bucket_file_list(self) -> List[str]:
        """Gets the list of files required for inference.

        Returns:
            List[str]: The list of file names.
        """
        return ["textual.onnx", "visual.onnx"]

    def infer(self, request: ClipInferenceRequest) -> ClipEmbeddingResponse:
        """Routes the request to the appropriate inference function.

        Args:
            request (ClipInferenceRequest): The request object containing the inference details.

        Returns:
            ClipEmbeddingResponse: The response object containing the embeddings.
        """
        if isinstance(request, ClipImageEmbeddingRequest):
            infer_func = self.embed_image
        elif isinstance(request, ClipTextEmbeddingRequest):
            infer_func = self.embed_text
        elif isinstance(request, ClipCompareRequest):
            infer_func = self.compare
        else:
            raise ValueError(
                f"Request type {type(request)} is not a valid ClipInferenceRequest"
            )
        return infer_func(request)

    def preproc_image(self, image: InferenceRequestImage) -> np.ndarray:
        """Preprocesses an inference request image.

        Args:
            image (InferenceRequestImage): The object containing information necessary to load the image for inference.

        Returns:
            np.ndarray: A numpy array of the preprocessed image pixel data.
        """
        pil_image = self.load_image(image.type, image.value)
        preprocessed_image = self.clip_preprocess(pil_image)

        img_in = np.expand_dims(preprocessed_image, axis=0)

        return img_in.astype(np.float32)
