import copy
import hashlib
from io import BytesIO
from threading import RLock
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar, Union

import numpy as np
import torch
from pycocotools import mask as mask_utils
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from inference.core.cache.model_artifacts import (
    are_all_files_cached,
    save_bytes_in_cache,
)
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
    CORE_MODEL_BUCKET,
    DEVICE,
    DISABLE_SAM3_LOGITS_CACHE,
    INFER_BUCKET,
    MODELS_CACHE_AUTH_ENABLED,
    SAM3_MAX_EMBEDDING_CACHE_SIZE,
    SAM3_MAX_LOGITS_CACHE_SIZE,
)
from inference.core.exceptions import ModelArtefactError, RoboflowAPINotAuthorizedError
from inference.core.models.roboflow import (
    RoboflowCoreModel,
    is_model_artefacts_bucket_available,
)
from inference.core.registries.roboflow import _check_if_api_key_has_access_to_model
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_from_url,
    get_roboflow_model_data,
)
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly
from inference.core.utils.torchscript_guard import _temporarily_disable_torch_jit_script

# from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
# from sam3.sam3_video_model_builder import build_sam3_tracking_predictor



if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


T = TypeVar("T")


class LogitsCacheType(TypedDict):
    logits: np.ndarray
    prompt_set: Sam2PromptSet


class Sam3ForInteractiveImageSegmentation(RoboflowCoreModel):
    """
    SegmentAnything3 class for handling segmentation tasks onm images with
    box prompting and point prompting, the way as SAM2 did.
    """

    def __init__(
        self,
        *args,
        model_id: str = "sam3/sam3_final",
        low_res_logits_cache_size: int = SAM3_MAX_LOGITS_CACHE_SIZE,
        embedding_cache_size: int = SAM3_MAX_EMBEDDING_CACHE_SIZE,
        **kwargs,
    ):
        """Initializes the SegmentAnything.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, model_id=model_id, **kwargs)
        checkpoint = self.cache_file("weights.pt")
        bpe_path = self.cache_file("bpe_simple_vocab_16e6.txt.gz")

        self.sam_model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_from_HF=False,
            compile=False,
            enable_inst_interactivity=True,
        )
        self.low_res_logits_cache_size = low_res_logits_cache_size
        self.embedding_cache_size = embedding_cache_size
        self.embedding_cache = {}
        self.image_size_cache = {}
        self.embedding_cache_keys = []
        self.low_res_logits_cache: Dict[Tuple[str, str], LogitsCacheType] = {}
        self.low_res_logits_cache_keys = []
        self._state_lock = RLock()
        self.task_type = "unsupervised-segmentation"

    def get_infer_bucket_file_list(self) -> List[str]:
        """Gets the list of files required for inference.

        Returns:
            List[str]: List of file names.
        """
        return ["weights.pt"]

    @torch.inference_mode()
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
        if image_id:
            embedding_cache_content = self.embedding_cache.get(image_id)
            image_size_content = self.image_size_cache.get(image_id)
            if embedding_cache_content is not None and image_size_content is not None:
                return embedding_cache_content, image_size_content, image_id

        img_in = self.preproc_image(image)
        if image_id is None:
            image_id = hashlib.md5(img_in.tobytes()).hexdigest()[:12]

        embedding_cache_content = self.embedding_cache.get(image_id)
        image_size_content = self.image_size_cache.get(image_id)
        if embedding_cache_content is not None and image_size_content is not None:
            return (
                embedding_cache_content,
                image_size_content,
                image_id,
            )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with _temporarily_disable_torch_jit_script():
                processor = Sam3Processor(self.sam_model)
            state = processor.set_image(torch.from_numpy(img_in).permute(2, 0, 1))
            embedding_dict = state

        with self._state_lock:
            self.embedding_cache[image_id] = embedding_dict
            self.image_size_cache[image_id] = img_in.shape[:2]
            safe_remove_from_list(values=self.embedding_cache_keys, element=image_id)
            self.embedding_cache_keys.append(image_id)
            if len(self.embedding_cache_keys) > self.embedding_cache_size:
                cache_key = safe_pop_from_list(values=self.embedding_cache_keys)
                if cache_key is not None:
                    safe_remove_from_dict(values=self.embedding_cache, key=cache_key)
                    safe_remove_from_dict(values=self.image_size_cache, key=cache_key)
            return embedding_dict, img_in.shape[:2], image_id

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
            predictions = _masks_to_predictions(masks, scores, request.format)
            return Sam2SegmentationResponse(
                time=perf_counter() - t1,
                predictions=predictions,
            )
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
            load_logits_from_cache and not DISABLE_SAM3_LOGITS_CACHE
        )
        save_logits_to_cache = save_logits_to_cache and not DISABLE_SAM3_LOGITS_CACHE
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

            # with _temporarily_disable_torch_jit_script():
            # processor = Sam3Processor(self.sam_model)

            # processor._is_image_set = True
            # processor._features = embedding
            # processor._orig_hw = [original_image_size]
            # processor._is_batch = False
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

            masks, scores, low_resolution_logits = self.sam_model.predict_inst(
                embedding,
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
        with self._state_lock:
            self.low_res_logits_cache[prompt_id] = {
                "logits": logits,
                "prompt_set": prompt_set,
            }
            safe_remove_from_list(
                values=self.low_res_logits_cache_keys, element=prompt_id
            )
            self.low_res_logits_cache_keys.append(prompt_id)
            if len(self.low_res_logits_cache_keys) > self.low_res_logits_cache_size:
                cache_key = safe_pop_from_list(values=self.low_res_logits_cache_keys)
                if cache_key is not None:
                    safe_remove_from_dict(
                        values=self.low_res_logits_cache, key=cache_key
                    )

    @property
    def model_artifact_bucket(self):
        # Use CORE bucket for base SAM3, standard INFER bucket for fine-tuned models
        return CORE_MODEL_BUCKET if self._is_core_sam3_endpoint() else INFER_BUCKET

    def _is_core_sam3_endpoint(self) -> bool:
        return isinstance(self.endpoint, str) and self.endpoint.startswith("sam3/")

    def download_weights(self) -> None:
        infer_bucket_files = self.get_infer_bucket_file_list()

        # Auth check aligned with chosen endpoint type
        if MODELS_CACHE_AUTH_ENABLED:
            endpoint_type = (
                ModelEndpointType.CORE_MODEL
                if self._is_core_sam3_endpoint()
                else ModelEndpointType.ORT
            )
            if not _check_if_api_key_has_access_to_model(
                api_key=self.api_key,
                model_id=self.endpoint,
                endpoint_type=endpoint_type,
            ):
                raise RoboflowAPINotAuthorizedError(
                    f"API key {self.api_key} does not have access to model {self.endpoint}"
                )
        # Already cached
        if are_all_files_cached(files=infer_bucket_files, model_id=self.endpoint):
            return None
        # S3 path works for both; keys are {endpoint}/<file>
        if is_model_artefacts_bucket_available():
            self.download_model_artefacts_from_s3()
            return None
            # API fallback
        if self._is_core_sam3_endpoint():
            # Base SAM3 from core_model endpoint; preserves filenames
            return super().download_model_from_roboflow_api()

        # Fine-tuned SAM3: use ORT endpoint to fetch weights map or model url
        api_data = get_roboflow_model_data(
            api_key=self.api_key,
            model_id=self.endpoint,
            endpoint_type=ModelEndpointType.ORT,
            device_id=self.device_id,
        )

        ort = api_data.get("ort") if isinstance(api_data, dict) else None
        if not isinstance(ort, dict):
            raise ModelArtefactError("ORT response malformed for fine-tuned SAM3")

        # Preferred: explicit weights map of filename -> URL
        weights_map = ort.get("weights")
        if isinstance(weights_map, dict) and len(weights_map) > 0:
            for filename, url in weights_map.items():
                resp = get_from_url(
                    url, json_response=False, verify_content_length=True
                )
                save_bytes_in_cache(
                    content=resp.content,
                    file=str(filename),
                    model_id=self.endpoint,
                )
            return None
        raise ModelArtefactError(
            "ORT response missing both 'weights' for fine-tuned SAM3"
        )


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


def _masks_to_predictions(
    masks_np: np.ndarray, scores: List[float], fmt: str
) -> List[Sam2SegmentationPrediction]:
    """Convert boolean masks (N,H,W) to API predictions in requested format.

    Assumes masks_np is already normalized to (N,H,W) by _to_numpy_masks.

    Args:
        masks_np: Boolean or uint8 masks array (N,H,W)
        scores: Confidence scores per mask
        fmt: Output format: 'polygon', 'json', or 'rle'

    Returns:
        List of Sam2SegmentationPrediction
    """
    preds = []

    if masks_np.ndim != 3 or 0 in masks_np.shape:
        return preds

    if fmt in ["polygon", "json"]:
        polygons = masks2multipoly((masks_np > 0).astype(np.uint8))
        for poly, score in zip(polygons, scores[: len(polygons)]):
            preds.append(
                Sam2SegmentationPrediction(
                    masks=[p.tolist() for p in poly],
                    confidence=float(score),
                    format="polygon",
                )
            )
    elif fmt == "rle":
        for m, score in zip(masks_np, scores[: masks_np.shape[0]]):
            mb = (m > 0).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(mb))
            rle["counts"] = rle["counts"].decode("utf-8")
            preds.append(
                Sam2SegmentationPrediction(
                    masks=rle, confidence=float(score), format="rle"
                )
            )
    return preds


def safe_remove_from_list(values: List[T], element: T) -> None:
    try:
        values.remove(element)
    except ValueError:
        pass


def safe_pop_from_list(values: List[T]) -> Optional[T]:
    try:
        return values.pop(0)
    except IndexError:
        return None


def safe_remove_from_dict(values: Dict[T, Any], key: T) -> None:
    try:
        del values[key]
    except ValueError:
        pass
