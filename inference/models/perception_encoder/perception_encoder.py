import os
from time import perf_counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

import inference.models.perception_encoder.vision_encoder.pe as pe
import inference.models.perception_encoder.vision_encoder.transforms as transforms
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
from inference.core.env import CLIP_MAX_BATCH_SIZE, DEVICE, PERCEPTION_ENCODER_MODEL_ID
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.batching import create_batches
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import cosine_similarity

if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"


class PerceptionEncoder(RoboflowCoreModel):
    """Roboflow Perception Encoder model implementation.

    This class is responsible for handling the Percpetion Encoder model, including
    loading the model, preprocessing the input, and performing inference.

    Attributes:
        model (pe.CLIP): The PE-CLIP model instance.
        preprocess (function): Function to preprocess the image.
        tokenizer (function): Function to tokenize text.
        device (str): The device to run inference on (cuda/cpu).
    """

    def __init__(
        self,
        model_id: str = PERCEPTION_ENCODER_MODEL_ID,
        device: str = DEVICE,
        *args,
        **kwargs,
    ):
        """Initializes the PerceptionEncoder with the given arguments and keyword arguments."""
        t1 = perf_counter()
        super().__init__(model_id=model_id.lower(), *args, **kwargs)
        self.device = device
        self.log("Creating PE-CLIP model")
        # Parse model config from model_id (format: perception-encoder/PE-Core-L14-336)
        model_config = model_id.split("/")[-1]
        checkpoint_path = os.path.join(self.cache_dir, "model.pt")
        self.model = pe.CLIP.from_config(
            model_config, pretrained=True, checkpoint_path=checkpoint_path
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.preprocessor = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

        self.task_type = "embedding"

    def get_infer_bucket_file_list(self) -> List[str]:
        """Gets the list of files required for inference."""
        return ["model.pt"]  # No files needed as model is downloaded from HuggingFace

    def initialize_model(self) -> None:
        """Initialize the model. Not needed for PE-CLIP as it's loaded in __init__."""
        pass

    def preproc_image(self, image: InferenceRequestImage) -> torch.Tensor:
        """Preprocesses an inference request image."""
        pil_image = Image.fromarray(load_image_rgb(image))
        preprocessed_image = self.preprocessor(pil_image)
        return preprocessed_image.unsqueeze(0)

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
        t1 = perf_counter()

        if isinstance(image, list):
            if len(image) > CLIP_MAX_BATCH_SIZE:
                raise ValueError(
                    f"The maximum number of images that can be embedded at once is {CLIP_MAX_BATCH_SIZE}"
                )
            imgs = [self.preproc_image(i) for i in image]
            img_in = torch.cat(imgs, dim=0).to(self.device)
        else:
            img_in = self.preproc_image(image).to(self.device)

        if self.device == "cpu" or self.device == "mps":
            with torch.inference_mode():
                image_features, _, _ = self.model(img_in, None)
                # Convert to float32 before converting to numpy
                embeddings = image_features.float().cpu().numpy()
        else:
            with torch.inference_mode(), torch.autocast(self.device):
                image_features, _, _ = self.model(img_in, None)
                # Convert to float32 before converting to numpy
                embeddings = image_features.float().cpu().numpy()

        return embeddings

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

        results = []
        for texts_batch in create_batches(
            sequence=texts, batch_size=CLIP_MAX_BATCH_SIZE
        ):
            tokenized = self.tokenizer(texts_batch).to(self.device)
            # Use float32 for CPU, bfloat16 for CUDA
            if self.device == "cpu" or self.device == "mps":
                with torch.no_grad():
                    _, text_features, _ = self.model(None, tokenized)
            else:
                with torch.inference_mode(), torch.autocast(self.device):
                    _, text_features, _ = self.model(None, tokenized)

            # Convert to float32 before converting to numpy
            embeddings = text_features.float().cpu().numpy()
            results.append(embeddings)

        return np.concatenate(results, axis=0)

    def predict(self, img_in: torch.Tensor, **kwargs) -> Tuple[np.ndarray]:
        """Predict embeddings for an input tensor.

        Args:
            img_in (torch.Tensor): The input tensor to get embeddings for.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[np.ndarray]: A tuple containing the embeddings as a numpy array.
        """
        img_in = img_in.to(self.device)
        if self.device == "cpu" or self.device == "mps":
            with torch.inference_mode():
                image_features, _, _ = self.model(img_in, None)
        else:
            with torch.inference_mode(), torch.autocast(self.device):
                image_features, _, _ = self.model(img_in, None)

        embeddings = image_features.float().cpu().numpy()
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
