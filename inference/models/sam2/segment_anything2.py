import base64
import hashlib
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio.features
import sam2.utils.misc
import torch
from torch.nn.attention import SDPBackend

sam2.utils.misc.get_sdp_backends = lambda z: [
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from shapely.geometry import Polygon as ShapelyPolygon

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2InferenceRequest,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.responses.sam2 import (
    Sam2EmbeddingResponse,
    Sam2SegmentationPrediction,
    Sam2SegmentationResponse,
)
from inference.core.env import DEVICE, SAM2_VERSION_ID, SAM_MAX_EMBEDDING_CACHE_SIZE
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2poly

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class SegmentAnything2(RoboflowCoreModel):
    """SegmentAnything class for handling segmentation tasks.

    Attributes:
        sam: The segmentation model.
        predictor: The predictor for the segmentation model.
        ort_session: ONNX runtime inference session.
        embedding_cache: Cache for embeddings.
        image_size_cache: Cache for image sizes.
        embedding_cache_keys: Keys for the embedding cache.

    """

    def __init__(self, *args, model_id: str = f"sam2/{SAM2_VERSION_ID}", **kwargs):
        """Initializes the SegmentAnything.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, model_id=model_id, **kwargs)
        checkpoint = self.cache_file("weights.pt")
        model_cfg = {
            "hiera_large": "sam2_hiera_l.yaml",
            "hiera_small": "sam2_hiera_s.yaml",
            "hiera_tiny": "sam2_hiera_t.yaml",
            "hiera_b_plus": "sam2_hiera_b+.yaml",
        }[self.version_id]

        self.sam = build_sam2(model_cfg, checkpoint, device=DEVICE)

        self.predictor = SAM2ImagePredictor(self.sam)

        self.embedding_cache = {}
        self.image_size_cache = {}
        self.embedding_cache_keys = []

        self.task_type = "unsupervised-segmentation"

    def get_infer_bucket_file_list(self) -> List[str]:
        """Gets the list of files required for inference.

        Returns:
            List[str]: List of file names.
        """
        return ["weights.pt"]

    def embed_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Embeds an image and caches the result if an image_id is provided. If the image has been embedded before and cached,
        the cached result will be returned.

        Args:
            image (Any): The image to be embedded. The format should be compatible with the preproc_image method.
            image_id (Optional[str]): An identifier for the image. If provided, the embedding result will be cached
                                      with this ID. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: A tuple where the first element is the embedding of the image
                                               and the second element is the shape (height, width) of the processed image.

        Notes:
            - Embeddings and image sizes are cached to improve performance on repeated requests for the same image.
            - The cache has a maximum size defined by SAM_MAX_EMBEDDING_CACHE_SIZE. When the cache exceeds this size,
              the oldest entries are removed.

        Example:
            >>> img_array = ... # some image array
            >>> embed_image(img_array, image_id="sample123")
            (array([...]), (224, 224))
        """
        if image_id and image_id in self.embedding_cache:
            return (
                self.embedding_cache[image_id],
                self.image_size_cache[image_id],
                image_id,
            )

        img_in = self.preproc_image(image)
        if image_id is None:
            image_id = hashlib.md5(img_in.tobytes()).hexdigest()[:12]

        if image_id and image_id in self.embedding_cache:
            return (
                self.embedding_cache[image_id],
                self.image_size_cache[image_id],
                image_id,
            )

        with torch.inference_mode():
            self.predictor.set_image(img_in)
            embedding_dict = self.predictor._features

        self.embedding_cache[image_id] = embedding_dict
        self.image_size_cache[image_id] = img_in.shape[:2]
        self.embedding_cache_keys.append(image_id)
        if len(self.embedding_cache_keys) > SAM_MAX_EMBEDDING_CACHE_SIZE:
            cache_key = self.embedding_cache_keys.pop(0)
            del self.embedding_cache[cache_key]
            del self.image_size_cache[cache_key]
        return (embedding_dict, img_in.shape[:2], image_id)

    def infer_from_request(self, request: Sam2InferenceRequest):
        """Performs inference based on the request type.

        Args:
            request (SamInferenceRequest): The inference request.

        Returns:
            Union[SamEmbeddingResponse, SamSegmentationResponse]: The inference response.
        """
        t1 = perf_counter()
        if isinstance(request, Sam2EmbeddingRequest):
            _, _, image_id = self.embed_image(**request.dict())
            inference_time = perf_counter() - t1
            return Sam2EmbeddingResponse(time=inference_time, image_id=image_id)
        elif isinstance(request, Sam2SegmentationRequest):
            masks, scores, low_resolution_logits = self.segment_image(**request.dict())
            if request.format == "json":
                return turn_segmentation_results_into_api_response(
                    masks=masks,
                    scores=scores,
                    mask_threshold=self.predictor.mask_threshold,
                    inference_start_timestamp=t1,
                )
            elif request.format == "binary":
                binary_vector = BytesIO()
                np.savez_compressed(
                    binary_vector, masks=masks, low_res_masks=low_resolution_logits
                )
                binary_vector.seek(0)
                binary_data = binary_vector.getvalue()
                return binary_data
            else:
                raise ValueError(f"Invalid format {request.format}")

        else:
            raise ValueError(f"Invalid request type {type(request)}")

    def preproc_image(self, image: InferenceRequestImage):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        np_image = load_image_rgb(image)
        return np_image

    def segment_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        prompts: Optional[Union[Sam2PromptSet, dict]] = None,
        mask_input: Optional[Union[np.ndarray, List[List[List[float]]]]] = None,
        multimask_output: Optional[bool] = True,
        **kwargs,
    ):
        """
        Segments an image based on provided embeddings, points, masks, or cached results.
        If embeddings are not directly provided, the function can derive them from the input image or cache.

        Args:
            image (Any): The image to be segmented.
            image_id (Optional[str]): A cached identifier for the image. Useful for accessing cached embeddings or masks.
            prompts (Optional[List[Sam2Prompt]]): List of prompts to use for segmentation. Defaults to None.
            multimask_output: (bool): Flag to decide if multiple masks proposal to be predicted (among which the most
                promising will be returned
            )
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of np.array, where:
                - first element is of size (prompt_set_size, h, w) and represent mask with the highest confidence
                    for each prompt element
                - second element is of size (prompt_set_size, ) and represents ths score for most confident mask
                    of each prompt element
                - third element is of size (prompt_set_size, 256, 256) and represents the low resolution logits
                    for most confident mask of each prompt element

        Raises:
            ValueError: If necessary inputs are missing or inconsistent.

        Notes:
            - Embeddings, segmentations, and low-resolution logits can be cached to improve performance
              on repeated requests for the same image.
            - The cache has a maximum size defined by SAM_MAX_EMBEDDING_CACHE_SIZE. When the cache exceeds this size,
              the oldest entries are removed.
        """
        with torch.inference_mode():
            if image is None and not image_id:
                raise ValueError("Must provide either image or  cached image_id")
            elif image_id and image is None and image_id not in self.embedding_cache:
                raise ValueError(
                    f"Image ID {image_id} not in embedding cache, must provide the image or embeddings"
                )
            embedding, original_image_size, image_id = self.embed_image(
                image=image, image_id=image_id
            )

            self.predictor._is_image_set = True
            self.predictor._features = embedding
            self.predictor._orig_hw = [original_image_size]
            self.predictor._is_batch = False
            args = dict()
            if prompts:
                if type(prompts) is dict:
                    args = Sam2PromptSet(**prompts).to_sam2_inputs()
                else:
                    args = prompts.to_sam2_inputs()

            masks, scores, low_resolution_logits = self.predictor.predict(
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=True,
                normalize_coords=True,
                **args,
            )
            masks, scores, low_resolution_logits = choose_most_confident_sam_prediction(
                masks=masks,
                scores=scores,
                low_resolution_logits=low_resolution_logits,
            )

            return masks, scores, low_resolution_logits


def choose_most_confident_sam_prediction(
    masks: np.ndarray,
    scores: np.ndarray,
    low_resolution_logits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is supposed to post-process SAM2 inference and choose most confident
    mask regardless of `multimask_output` parameter value
    Args:
        masks: np array with values 0.0 and 1.0 representing predicted mask of size
            (prompt_set_size, proposed_maks, h, w) or (proposed_maks, h, w) - depending on
            prompt set size - unfortunately, prompt_set_size=1 causes squeeze operation
            in SAM2 library, so to handle inference uniformly, we need to compensate with
            this function.
        scores: array of size (prompt_set_size, proposed_maks) or (proposed_maks, ) depending
            on prompt set size - this array gives confidence score for mask proposal
        low_resolution_logits: array of size (prompt_set_size, proposed_maks, 256, 256) or
            (proposed_maks, 256, 256) - depending on prompt set size. These low resolution logits
             can be passed to a subsequent iteration as mask input.
    Returns:
        Tuple of np.array, where:
            - first element is of size (prompt_set_size, h, w) and represent mask with the highest confidence
                for each prompt element
            - second element is of size (prompt_set_size, ) and represents ths score for most confident mask
                of each prompt element
            - third element is of size (prompt_set_size, 256, 256) and represents the low resolution logits
                for most confident mask of each prompt element
    """
    if len(masks.shape) == 3:
        masks = np.expand_dims(masks, axis=0)
        scores = np.expand_dims(scores, axis=0)
        low_resolution_logits = np.expand_dims(low_resolution_logits, axis=0)
    selected_masks, selected_scores, selected_low_resolution_logits = [], [], []
    for mask, score, low_resolution_logit in zip(masks, scores, low_resolution_logits):
        selected_mask, selected_score, selected_low_resolution_logit = (
            choose_most_confident_prompt_set_element_prediction(
                mask=mask,
                score=score,
                low_resolution_logit=low_resolution_logit,
            )
        )
        selected_masks.append(selected_mask)
        selected_scores.append(selected_score)
        selected_low_resolution_logits.append(selected_low_resolution_logit)
    return (
        np.asarray(selected_masks),
        np.asarray(selected_scores),
        np.asarray(selected_low_resolution_logits),
    )


def choose_most_confident_prompt_set_element_prediction(
    mask: np.ndarray, score: np.ndarray, low_resolution_logit: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    max_score_index = np.argsort(score)[-1]
    selected_mask = mask[max_score_index]
    selected_score = score[max_score_index].item()
    selected_low_resolution_logit = low_resolution_logit[max_score_index]
    return selected_mask, selected_score, selected_low_resolution_logit


def turn_segmentation_results_into_api_response(
    masks: np.ndarray,
    scores: np.ndarray,
    mask_threshold: float,
    inference_start_timestamp: float,
) -> Sam2SegmentationResponse:
    predictions = []
    masks_plygons = masks2poly(masks >= mask_threshold)
    for mask_polygon, score in zip(masks_plygons, scores):
        prediction = Sam2SegmentationPrediction(
            mask=mask_polygon.tolist(),
            confidence=score.item(),
        )
        predictions.append(prediction)
    return Sam2SegmentationResponse(
        time=perf_counter() - inference_start_timestamp,
        predictions=predictions,
    )
