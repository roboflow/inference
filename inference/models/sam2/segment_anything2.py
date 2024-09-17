import copy
import hashlib
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import sam2.utils.misc
import torch
from torch.nn.attention import SDPBackend

sam2.utils.misc.get_sdp_backends = lambda z: [
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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
    DEVICE,
    DISABLE_SAM2_LOGITS_CACHE,
    SAM2_MAX_EMBEDDING_CACHE_SIZE,
    SAM2_MAX_LOGITS_CACHE_SIZE,
    SAM2_VERSION_ID,
)
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class LogitsCacheType(TypedDict):
    logits: np.ndarray
    prompt_set: Sam2PromptSet


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

    def __init__(
        self,
        *args,
        model_id: str = f"sam2/{SAM2_VERSION_ID}",
        low_res_logits_cache_size: int = SAM2_MAX_LOGITS_CACHE_SIZE,
        embedding_cache_size: int = SAM2_MAX_EMBEDDING_CACHE_SIZE,
        **kwargs,
    ):
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
        self.low_res_logits_cache_size = low_res_logits_cache_size
        self.embedding_cache_size = embedding_cache_size

        self.predictor = SAM2ImagePredictor(self.sam)

        self.embedding_cache = {}
        self.image_size_cache = {}
        self.embedding_cache_keys = []
        self.low_res_logits_cache: Dict[Tuple[str, str], LogitsCacheType] = {}
        self.low_res_logits_cache_keys = []

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
            - The cache has a maximum size defined by SAM2_MAX_CACHE_SIZE. When the cache exceeds this size,
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

        if image_id in self.embedding_cache:
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
        if image_id in self.embedding_cache_keys:
            self.embedding_cache_keys.remove(image_id)
        self.embedding_cache_keys.append(image_id)
        if len(self.embedding_cache_keys) > self.embedding_cache_size:
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
            prompt_set: Sam2PromptSet
            if prompts:
                if type(prompts) is dict:
                    prompt_set = Sam2PromptSet(**prompts)
                    args = prompt_set.to_sam2_inputs()
                else:
                    prompt_set = prompts
                    args = prompts.to_sam2_inputs()
            else:
                prompt_set = Sam2PromptSet()

            if mask_input is None and load_logits_from_cache:
                mask_input = maybe_load_low_res_logits_from_cache(
                    image_id, prompt_set, self.low_res_logits_cache
                )

            args = pad_points(args)
            if not any(args.values()):
                args = {"point_coords": [[0, 0]], "point_labels": [-1], "box": None}
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

            if save_logits_to_cache:
                self.add_low_res_logits_to_cache(
                    low_resolution_logits, image_id, prompt_set
                )

            return masks, scores, low_resolution_logits

    def add_low_res_logits_to_cache(
        self, logits: np.ndarray, image_id: str, prompt_set: Sam2PromptSet
    ) -> None:
        logits = logits[:, None, :, :]
        prompt_id = hash_prompt_set(image_id, prompt_set)
        self.low_res_logits_cache[prompt_id] = {
            "logits": logits,
            "prompt_set": prompt_set,
        }
        if prompt_id in self.low_res_logits_cache_keys:
            self.low_res_logits_cache_keys.remove(prompt_id)
        self.low_res_logits_cache_keys.append(prompt_id)
        if len(self.low_res_logits_cache_keys) > self.low_res_logits_cache_size:
            cache_key = self.low_res_logits_cache_keys.pop(0)
            del self.low_res_logits_cache[cache_key]


def hash_prompt_set(image_id: str, prompt_set: Sam2PromptSet) -> Tuple[str, str]:
    """Computes unique hash from a prompt set."""
    md5_hash = hashlib.md5()
    md5_hash.update(str(prompt_set).encode("utf-8"))
    return image_id, md5_hash.hexdigest()[:12]


def maybe_load_low_res_logits_from_cache(
    image_id: str,
    prompt_set: Sam2PromptSet,
    cache: Dict[Tuple[str, str], LogitsCacheType],
) -> Optional[np.ndarray]:
    "Loads prior masks from the cache by searching over possibel prior prompts."
    prompts = prompt_set.prompts
    if not prompts:
        return None

    return find_prior_prompt_in_cache(prompt_set, image_id, cache)


def find_prior_prompt_in_cache(
    initial_prompt_set: Sam2PromptSet,
    image_id: str,
    cache: Dict[Tuple[str, str], LogitsCacheType],
) -> Optional[np.ndarray]:
    """
    Performs search over the cache to see if prior used prompts are subset of this one.
    """

    logits_for_image = [cache[k] for k in cache if k[0] == image_id]
    maxed_size = 0
    best_match: Optional[np.ndarray] = None
    desired_size = initial_prompt_set.num_points() - 1
    for cached_dict in logits_for_image[::-1]:
        logits = cached_dict["logits"]
        prompt_set: Sam2PromptSet = cached_dict["prompt_set"]
        is_viable = is_prompt_strict_subset(prompt_set, initial_prompt_set)
        if not is_viable:
            continue

        size = prompt_set.num_points()
        # short circuit search if we find prompt with one less point (most recent possible mask)
        if size == desired_size:
            return logits
        if size >= maxed_size:
            maxed_size = size
            best_match = logits

    return best_match


def is_prompt_strict_subset(
    prompt_set_sub: Sam2PromptSet, prompt_set_super: Sam2PromptSet
) -> bool:
    if prompt_set_sub == prompt_set_super:
        return False

    super_copy = [p for p in prompt_set_super.prompts]
    for prompt_sub in prompt_set_sub.prompts:
        found_match = False
        for prompt_super in super_copy:
            is_sub = prompt_sub.box == prompt_super.box
            is_sub = is_sub and set(
                p.to_hashable() for p in prompt_sub.points or []
            ) <= set(p.to_hashable() for p in prompt_super.points or [])
            if is_sub:
                super_copy.remove(prompt_super)
                found_match = True
                break
        if not found_match:
            return False

    # every prompt in prompt_set_sub has a matching super prompt
    return True


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
        )
        predictions.append(prediction)
    return Sam2SegmentationResponse(
        time=perf_counter() - inference_start_timestamp,
        predictions=predictions,
    )


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
