import copy
import hashlib
from io import BytesIO
from threading import RLock
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar, Union

import numpy as np
import sam2.utils.misc
import torch
from pycocotools import mask as mask_utils
from torch.nn.attention import SDPBackend

from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.postprocess import masks2multipoly
from inference_models import AutoModel
from inference_models.models.sam2.cache import (
    Sam2ImageEmbeddingsInMemoryCache,
    Sam2LowResolutionMasksInMemoryCache,
)
from inference_models.models.sam2.sam2_torch import SAM2Torch

sam2.utils.misc.get_sdp_backends = lambda z: [
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2InferenceRequest,
    Sam2Prompt,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.responses.sam2 import (
    Sam2EmbeddingResponse,
    Sam2SegmentationPrediction,
    Sam2SegmentationResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DEVICE,
    DISABLE_SAM2_LOGITS_CACHE,
    SAM2_MAX_EMBEDDING_CACHE_SIZE,
    SAM2_MAX_LOGITS_CACHE_SIZE,
    SAM2_VERSION_ID,
)
from inference.core.models.base import Model
from inference.core.utils.image_utils import load_image_bgr
from inference.usage_tracking.collector import usage_collector

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MASK_THRESHOLD = 0.0


class InferenceModelsSAM2Adapter(Model):
    """SegmentAnything class for handling segmentation tasks.

    Attributes:
        sam: The segmentation model.
        embedding_cache: Cache for embeddings.
        image_size_cache: Cache for image sizes.
        embedding_cache_keys: Keys for the embedding cache.

    """

    def __init__(
        self,
        *args,
        model_id: str = f"sam2/{SAM2_VERSION_ID}",
        api_key: Optional[str] = None,
        low_res_logits_cache_size: int = SAM2_MAX_LOGITS_CACHE_SIZE,
        embedding_cache_size: int = SAM2_MAX_EMBEDDING_CACHE_SIZE,
        **kwargs,
    ):
        """Initializes the SegmentAnything.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "unsupervised-segmentation"

        sam2_image_embeddings_cache = Sam2ImageEmbeddingsInMemoryCache.init(
            size_limit=embedding_cache_size,
            send_to_cpu=True,
        )
        sam2_low_resolution_masks_cache = Sam2LowResolutionMasksInMemoryCache.init(
            size_limit=low_res_logits_cache_size,
            send_to_cpu=True,
        )
        extra_weights_provider_headers = get_extra_weights_provider_headers()
        self._model: SAM2Torch = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            sam2_image_embeddings_cache=sam2_image_embeddings_cache,
            sam2_low_resolution_masks_cache=sam2_low_resolution_masks_cache,
            sam2_allow_client_generated_hash_ids=True,
            weights_provider_extra_headers=extra_weights_provider_headers,
            **kwargs,
        )

    @usage_collector("model")
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
                    mask_threshold=MASK_THRESHOLD,
                    inference_start_timestamp=t1,
                )
            elif request.format == "rle":
                return turn_segmentation_results_into_rle_response(
                    masks=masks,
                    scores=scores,
                    mask_threshold=0.0,
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
            - The cache has a maximum size defined by SAM2_MAX_CACHE_SIZE. When the cache exceeds this size,
              the oldest entries are removed.

        Example:
            >>> img_array = ... # some image array
            >>> embed_image(img_array, image_id="sample123")
            (array([...]), (224, 224))
        """
        loaded_image = self.preproc_image(image)
        if loaded_image is None:
            raise ValueError("Image must be provided to handle this request.")
        embeddings = self._model.embed_images(
            images=loaded_image, image_hashes=image_id, **kwargs
        )[0]
        embedding_dict = {
            "image_embed": embeddings.embeddings.cpu().numpy(),
            "high_res_feats": [
                f.cpu().numpy() for f in embeddings.high_resolution_features
            ],
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embedding_dict, embeddings.image_size_hw, embeddings.image_hash

    def preproc_image(self, image: InferenceRequestImage):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        if image is not None:
            return load_image_bgr(image)
        return None

    def segment_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        prompts: Optional[Union[Sam2PromptSet, dict]] = None,
        multimask_output: Optional[bool] = True,
        mask_input: Optional[Union[np.ndarray, List[List[List[float]]]]] = None,
        save_logits_to_cache: bool = False,
        load_logits_from_cache: bool = False,
        **kwargs,
    ):
        """
        Segments an image based on provided embeddings, points, masks, or cached results.
        If embeddings are not directly provided, the function can derive them from the input image or cache.

        Args:
            image (Any): The image to be segmented.
            image_id (Optional[str]): A cached identifier for the image. Useful for accessing cached embeddings or masks.
            prompts (Optional[List[Sam2Prompt]]): List of prompts to use for segmentation. Defaults to None.
            mask_input (Optional[Union[np.ndarray, List[List[List[float]]]]]): Input low_res_logits for the image.
            multimask_output: (bool): Flag to decide if multiple masks proposal to be predicted (among which the most
                promising will be returned
            )
            use_logits_cache: (bool): Flag to decide to use cached logits from prior prompting
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
        load_logits_from_cache = (
            load_logits_from_cache and not DISABLE_SAM2_LOGITS_CACHE
        )
        save_logits_to_cache = save_logits_to_cache and not DISABLE_SAM2_LOGITS_CACHE
        loaded_image = self.preproc_image(image)
        if prompts is not None:
            if type(prompts) is dict:
                prompts = Sam2PromptSet(**prompts)
        else:
            prompts = Sam2PromptSet()
        args = prompts.to_sam2_inputs()
        args = pad_points(args)
        if not any(args.values()):
            args = {"point_coords": [[0, 0]], "point_labels": [-1], "box": None}
        if args["point_coords"] is not None:
            args["point_coords"] = np.array(args["point_coords"])
        if args["point_labels"] is not None:
            args["point_labels"] = np.array(args["point_labels"])
        if args["box"] is not None:
            args["box"] = np.array(args["box"])
        if mask_input is not None and isinstance(mask_input, list):
            mask_input = np.array(mask_input)
        prediction = self._model.segment_images(
            images=loaded_image,
            image_hashes=image_id,
            point_coordinates=args["point_coords"],
            point_labels=args["point_labels"],
            boxes=args["box"],
            mask_input=mask_input,
            multi_mask_output=multimask_output,
            threshold=MASK_THRESHOLD,
            load_from_mask_input_cache=load_logits_from_cache,
            save_to_mask_input_cache=save_logits_to_cache,
            use_embeddings_cache=True,
            return_logits=True,
        )[0]
        result = choose_most_confident_sam_prediction(
            masks=prediction.masks.cpu().numpy(),
            scores=prediction.scores.cpu().numpy(),
            low_resolution_logits=prediction.logits.cpu().numpy(),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pad_points(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pad arguments to be passed to sam2 model with not_a_point label (-1).
    This is necessary when there are multiple prompts per image so that a tensor can be created.


    Also pads empty point lists with a dummy non-point entry.
    """
    args = copy.deepcopy(args)
    if args["point_coords"] is not None:
        max_len = max(max(len(prompt) for prompt in args["point_coords"]), 1)
        for prompt in args["point_coords"]:
            for _ in range(max_len - len(prompt)):
                prompt.append([0, 0])
        for label in args["point_labels"]:
            for _ in range(max_len - len(label)):
                label.append(-1)
    else:
        if args["point_labels"] is not None:
            raise ValueError(
                "Can't have point labels without corresponding point coordinates"
            )
    return args


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
    masks_plygons = masks2multipoly(masks >= mask_threshold)
    for mask_polygon, score in zip(masks_plygons, scores):
        prediction = Sam2SegmentationPrediction(
            masks=[mask.tolist() for mask in mask_polygon],
            confidence=score.item(),
            format="polygon",
        )
        predictions.append(prediction)
    return Sam2SegmentationResponse(
        time=perf_counter() - inference_start_timestamp,
        predictions=predictions,
    )


def turn_segmentation_results_into_rle_response(
    masks: np.ndarray,
    scores: np.ndarray,
    mask_threshold: float,
    inference_start_timestamp: float,
) -> Sam2SegmentationResponse:
    predictions = []
    for mask, score in zip(masks, scores):
        # Apply same threshold as polygon format
        mask_binary = (mask >= mask_threshold).astype(np.uint8)

        # Encode mask to RLE format
        rle = mask_utils.encode(np.asfortranarray(mask_binary))
        rle["counts"] = rle["counts"].decode("utf-8")

        predictions.append(
            Sam2SegmentationPrediction(masks=rle, confidence=float(score), format="rle")
        )

    return Sam2SegmentationResponse(
        time=perf_counter() - inference_start_timestamp,
        predictions=predictions,
    )
