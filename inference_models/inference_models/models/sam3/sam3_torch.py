import hashlib
import json
from copy import copy
from typing import Dict, Generator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint as Sam3Datapoint
from sam3.train.data.sam3_image_dataset import FindQueryLoaded
from sam3.train.data.sam3_image_dataset import Image as Sam3ImageDP
from sam3.train.data.sam3_image_dataset import InferenceMetadata
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import (
    CorruptedModelPackageError,
    ModelInputError,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.sam3.cache import (
    Sam3ImageEmbeddingsCache,
    Sam3ImageEmbeddingsCacheNullObject,
    Sam3LowResolutionMasksCache,
    Sam3LowResolutionMasksCacheNullObject,
)
from inference_models.models.sam3.entities import (
    SAM3ImageEmbeddings,
    SAM3MaskCacheEntry,
    SAM3Prediction,
)
from inference_models.utils.file_system import read_json

ArrayOrTensor = Union[np.ndarray, torch.Tensor]
T = TypeVar("T")

MAX_SAM3_BATCH_SIZE = 8
DEFAULT_SAM3_IMAGE_SIZE = 1024

SUPPORTED_VERSIONS = {
    "sam3_final",
}


class SAM3Torch:
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = MAX_SAM3_BATCH_SIZE,
        image_size: int = DEFAULT_SAM3_IMAGE_SIZE,
        sam3_image_embeddings_cache: Optional[Sam3ImageEmbeddingsCache] = None,
        sam3_low_resolution_masks_cache: Optional[Sam3LowResolutionMasksCache] = None,
        compile_model: bool = False,
        enable_inst_interactivity: bool = True,
        **kwargs,
    ) -> "SAM3Torch":
        if sam3_image_embeddings_cache is None:
            sam3_image_embeddings_cache = Sam3ImageEmbeddingsCacheNullObject()
        if sam3_low_resolution_masks_cache is None:
            sam3_low_resolution_masks_cache = Sam3LowResolutionMasksCacheNullObject()

        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "weights.pt",
                "bpe_simple_vocab_16e6.txt.gz",
            ],
        )

        try:
            config_content = get_model_package_contents(
                model_package_dir=model_name_or_path,
                elements=["sam_configuration.json"],
            )
            version = decode_sam_version(
                config_path=config_content["sam_configuration.json"]
            )
            if version not in SUPPORTED_VERSIONS:
                raise CorruptedModelPackageError(
                    message=f"Detected unsupported version of SAM3 model: {version}. Supported versions: "
                    f"are {SUPPORTED_VERSIONS}. If you run inference locally, verify the correctness of "
                    f"SAM3 model package. If you see the error running on Roboflow platform - "
                    "contact us to get help.",
                    help_url="https://todo",
                )
        except KeyError:
            pass

        device_str = "cuda" if device.type == "cuda" else "cpu"
        sam3_model = build_sam3_image_model(
            bpe_path=model_package_content["bpe_simple_vocab_16e6.txt.gz"],
            checkpoint_path=model_package_content["weights.pt"],
            device=device_str,
            load_from_HF=False,
            compile=compile_model,
            enable_inst_interactivity=enable_inst_interactivity,
        )

        transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=image_size,
                    max_size=image_size,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        return cls(
            model=sam3_model,
            transform=transform,
            device=device,
            max_batch_size=max_batch_size,
            image_size=image_size,
            sam3_image_embeddings_cache=sam3_image_embeddings_cache,
            sam3_low_resolution_masks_cache=sam3_low_resolution_masks_cache,
            enable_inst_interactivity=enable_inst_interactivity,
        )

    def __init__(
        self,
        model,
        transform: ComposeAPI,
        device: torch.device,
        max_batch_size: int,
        image_size: int,
        sam3_image_embeddings_cache: Sam3ImageEmbeddingsCache,
        sam3_low_resolution_masks_cache: Sam3LowResolutionMasksCache,
        enable_inst_interactivity: bool = True,
    ):
        self._model = model
        self._transform = transform
        self._device = device
        self._max_batch_size = max_batch_size
        self._image_size = image_size
        self._sam3_image_embeddings_cache = sam3_image_embeddings_cache
        self._sam3_low_resolution_masks_cache = sam3_low_resolution_masks_cache
        self._enable_inst_interactivity = enable_inst_interactivity

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        use_embeddings_cache: bool = True,
        **kwargs,
    ) -> List[SAM3ImageEmbeddings]:
        images_list = maybe_wrap_in_list(images)
        if images_list is None:
            raise ModelInputError(
                message="No images provided to embed_images()",
                help_url="https://todo",
            )

        image_hashes = [compute_image_hash(img) for img in images_list]
        original_sizes = [get_image_size(img) for img in images_list]

        embeddings_from_cache: Dict[int, SAM3ImageEmbeddings] = {}
        images_to_compute, indices_to_compute = [], []

        for idx, (image, image_hash) in enumerate(zip(images_list, image_hashes)):
            cache_content = None
            if use_embeddings_cache:
                cache_content = self._sam3_image_embeddings_cache.retrieve_embeddings(
                    key=image_hash
                )
            if cache_content is not None:
                cache_content = cache_content.to(device=self._device)
                embeddings_from_cache[idx] = cache_content
            else:
                images_to_compute.append(image)
                indices_to_compute.append(idx)

        computed_embeddings = []
        if len(images_to_compute) > 0:
            for batch_start in range(0, len(images_to_compute), self._max_batch_size):
                batch_end = min(
                    batch_start + self._max_batch_size, len(images_to_compute)
                )
                batch_images = images_to_compute[batch_start:batch_end]
                batch_indices = indices_to_compute[batch_start:batch_end]

                batch_embeddings = self._forward_image_embeddings(
                    images=batch_images,
                    image_hashes=[image_hashes[i] for i in batch_indices],
                    original_sizes=[original_sizes[i] for i in batch_indices],
                )
                computed_embeddings.extend(batch_embeddings)

        result_embeddings = []
        computed_idx = 0
        for i in range(len(images_list)):
            if i in embeddings_from_cache:
                result_embeddings.append(embeddings_from_cache[i])
            else:
                result_embeddings.append(computed_embeddings[computed_idx])
                computed_idx += 1

        if use_embeddings_cache:
            for embeddings in result_embeddings:
                self._sam3_image_embeddings_cache.save_embeddings(
                    key=embeddings.image_hash, embeddings=embeddings
                )

        return result_embeddings

    @torch.inference_mode()
    def _forward_image_embeddings(
        self,
        images: List[Union[np.ndarray, torch.Tensor]],
        image_hashes: List[str],
        original_sizes: List[Tuple[int, int]],
    ) -> List[SAM3ImageEmbeddings]:
        result_embeddings = []

        for image, image_hash, size in zip(images, image_hashes, original_sizes):
            if isinstance(image, torch.Tensor):
                np_image = image.cpu().numpy()
                if np_image.shape[0] == 3:
                    np_image = np_image.transpose(1, 2, 0)
                np_image = (
                    (np_image * 255).astype(np.uint8)
                    if np_image.max() <= 1
                    else np_image
                )
            else:
                np_image = image

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                processor = Sam3Processor(self._model)
                state = processor.set_image(torch.from_numpy(np_image).permute(2, 0, 1))

            result_embeddings.append(
                SAM3ImageEmbeddings(
                    image_hash=image_hash,
                    image_size_hw=size,
                    embeddings=state,
                )
            )

        return result_embeddings

    def segment_images(
        self,
        images: Optional[
            Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
        ] = None,
        embeddings: Optional[
            Union[List[SAM3ImageEmbeddings], SAM3ImageEmbeddings]
        ] = None,
        point_coordinates: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        point_labels: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        boxes: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        mask_input: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        multi_mask_output: bool = True,
        return_logits: bool = False,
        load_from_mask_input_cache: bool = False,
        save_to_mask_input_cache: bool = False,
        use_embeddings_cache: bool = True,
        **kwargs,
    ) -> List[SAM3Prediction]:
        if images is None and embeddings is None:
            raise ModelInputError(
                message="Attempted to use SAM3 model segment_images(...) method not providing valid input - "
                "neither `images` nor `embeddings` parameter is given. If you run inference locally, "
                "verify your integration making sure that the model interface is used correctly. Running "
                "on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )

        if images is not None:
            embeddings_list = self.embed_images(
                images=images,
                use_embeddings_cache=use_embeddings_cache,
                **kwargs,
            )
        else:
            embeddings_list = maybe_wrap_in_list(embeddings)

        image_hashes = [e.image_hash for e in embeddings_list]
        original_image_sizes = [e.image_size_hw for e in embeddings_list]

        point_coordinates = maybe_wrap_in_list(point_coordinates)
        point_labels = maybe_wrap_in_list(point_labels)
        boxes = maybe_wrap_in_list(boxes)
        mask_input = maybe_wrap_in_list(mask_input)

        point_coordinates, point_labels, boxes, mask_input = equalize_batch_size(
            embeddings_batch_size=len(embeddings_list),
            point_coordinates=point_coordinates,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
        )

        predictions = []
        for idx, embedding in enumerate(embeddings_list):
            image_point_coords = point_coordinates[idx] if point_coordinates else None
            image_point_labels = point_labels[idx] if point_labels else None
            image_boxes = boxes[idx] if boxes else None
            image_mask_input = mask_input[idx] if mask_input else None
            image_hash = image_hashes[idx]
            original_size = original_image_sizes[idx]

            serialized_prompt, prompt_hash = None, None
            if save_to_mask_input_cache or load_from_mask_input_cache:
                serialized_prompt = serialize_prompt(
                    point_coordinates=image_point_coords,
                    point_labels=image_point_labels,
                    boxes=image_boxes,
                )
                prompt_hash = hash_serialized_prompt(serialized_prompt)

            if image_mask_input is None and load_from_mask_input_cache:
                image_mask_input = attempt_load_image_mask_from_cache(
                    image_hash=image_hash,
                    serialized_prompt_hash=prompt_hash,
                    serialized_prompt=serialized_prompt,
                    sam3_low_resolution_masks_cache=self._sam3_low_resolution_masks_cache,
                    device=self._device,
                )

            prediction = self._predict_for_single_image(
                embeddings=embedding,
                original_image_size=original_size,
                point_coordinates=image_point_coords,
                point_labels=image_point_labels,
                boxes=image_boxes,
                mask_input=image_mask_input,
                multi_mask_output=multi_mask_output,
                return_logits=return_logits,
            )

            if save_to_mask_input_cache and len(prediction.masks.shape) >= 2:
                max_score_id = torch.argmax(prediction.scores).item()
                mask_entry = SAM3MaskCacheEntry(
                    prompt_hash=prompt_hash,
                    serialized_prompt=serialized_prompt,
                    mask=prediction.logits[max_score_id].unsqueeze(dim=0),
                )
                self._sam3_low_resolution_masks_cache.save_mask(
                    key=image_hash,
                    mask=mask_entry,
                )

            predictions.append(prediction)

        return predictions

    @torch.inference_mode()
    def _predict_for_single_image(
        self,
        embeddings: SAM3ImageEmbeddings,
        original_image_size: Tuple[int, int],
        point_coordinates: Optional[ArrayOrTensor] = None,
        point_labels: Optional[ArrayOrTensor] = None,
        boxes: Optional[ArrayOrTensor] = None,
        mask_input: Optional[ArrayOrTensor] = None,
        multi_mask_output: bool = True,
        return_logits: bool = False,
    ) -> SAM3Prediction:
        args = {}

        if point_coordinates is not None and point_labels is not None:
            if isinstance(point_coordinates, np.ndarray):
                point_coordinates = point_coordinates.tolist()
            elif isinstance(point_coordinates, torch.Tensor):
                point_coordinates = point_coordinates.cpu().tolist()

            if isinstance(point_labels, np.ndarray):
                point_labels = point_labels.tolist()
            elif isinstance(point_labels, torch.Tensor):
                point_labels = point_labels.cpu().tolist()

            args["point_coords"] = point_coordinates
            args["point_labels"] = point_labels

        if boxes is not None:
            if isinstance(boxes, np.ndarray):
                boxes_list = boxes.tolist()
            elif isinstance(boxes, torch.Tensor):
                boxes_list = boxes.cpu().tolist()
            else:
                boxes_list = boxes
            if len(boxes_list) > 0 and isinstance(boxes_list[0], (int, float)):
                args["box"] = boxes_list
            else:
                args["box"] = boxes_list[0] if boxes_list else None

        args = pad_points(args)
        if not any(args.values()):
            args = {"point_coords": [[0, 0]], "point_labels": [-1], "box": None}

        mask_input_tensor = None
        if mask_input is not None:
            if isinstance(mask_input, np.ndarray):
                mask_input_tensor = torch.from_numpy(mask_input).to(self._device)
            elif isinstance(mask_input, torch.Tensor):
                mask_input_tensor = mask_input.to(self._device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, scores, low_res_logits = self._model.predict_inst(
                embeddings.embeddings,
                mask_input=mask_input_tensor,
                multimask_output=multi_mask_output,
                return_logits=True,
                normalize_coords=True,
                **args,
            )

        masks, scores, low_res_logits = choose_most_confident_prediction(
            masks=masks,
            scores=scores,
            low_resolution_logits=low_res_logits,
        )

        masks_tensor = (
            torch.from_numpy(masks) if isinstance(masks, np.ndarray) else masks
        )
        scores_tensor = (
            torch.from_numpy(scores) if isinstance(scores, np.ndarray) else scores
        )
        logits_tensor = (
            torch.from_numpy(low_res_logits)
            if isinstance(low_res_logits, np.ndarray)
            else low_res_logits
        )

        if not return_logits:
            masks_tensor = masks_tensor > 0

        return SAM3Prediction(
            masks=masks_tensor,
            scores=scores_tensor,
            logits=logits_tensor,
        )

    def segment_with_text(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        prompts: List[Dict],
        output_prob_thresh: float = 0.5,
        **kwargs,
    ) -> List[Dict]:
        images_list = maybe_wrap_in_list(images)
        if images_list is None:
            raise ModelInputError(
                message="No images provided to segment_with_text()",
                help_url="https://todo",
            )

        results = []
        for image in images_list:
            np_image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            if np_image.shape[0] == 3:
                np_image = np_image.transpose(1, 2, 0)

            h, w = np_image.shape[:2]
            pil_image = Image.fromarray(np_image.astype(np.uint8))

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    datapoint = Sam3Datapoint(
                        find_queries=[],
                        images=[Sam3ImageDP(data=pil_image, objects=[], size=(h, w))],
                    )

                    prompt_ids = []
                    for idx, p in enumerate(prompts):
                        if p.get("boxes"):
                            q = _build_visual_query(
                                coco_id=idx,
                                h=h,
                                w=w,
                                boxes=p["boxes"],
                                labels=p.get("box_labels", []),
                                text=p.get("text"),
                            )
                        else:
                            q = _build_text_query(
                                coco_id=idx,
                                h=h,
                                w=w,
                                text=p.get("text"),
                            )
                        datapoint.find_queries.append(q)
                        prompt_ids.append(idx)

                    datapoint = self._transform(datapoint)
                    batch = collate_fn_api(batch=[datapoint], dict_key="dummy")["dummy"]
                    batch = copy_data_to_device(
                        batch,
                        self._device,
                        non_blocking=True,
                    )

                    output = self._model(batch)

                    post = PostProcessImage(
                        max_dets_per_img=-1,
                        iou_type="segm",
                        use_original_sizes_box=True,
                        use_original_sizes_mask=True,
                        convert_mask_to_rle=False,
                        detection_threshold=float(output_prob_thresh),
                        to_cpu=True,
                    )
                    processed = post.process_results(output, batch.find_metadatas)

            image_results = []
            for idx, coco_id in enumerate(prompt_ids):
                masks = processed[coco_id].get("masks")
                scores = processed[coco_id].get("scores", [])

                if masks is not None:
                    if hasattr(masks, "detach"):
                        masks = masks.detach().cpu().numpy()
                    masks = np.array(masks)
                else:
                    masks = np.zeros((0, h, w), dtype=np.uint8)

                image_results.append(
                    {
                        "prompt_index": idx,
                        "masks": masks,
                        "scores": list(scores),
                    }
                )

            results.append(image_results)

        return results


def decode_sam_version(config_path: str) -> str:
    config = read_json(path=config_path)
    version = config["version"]
    if not isinstance(version, str):
        raise ValueError("Could not decode SAM model version")
    return version


def compute_image_hash(image: Union[torch.Tensor, np.ndarray]) -> str:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    return hashlib.sha1(image.tobytes()).hexdigest()


def get_image_size(image: Union[torch.Tensor, np.ndarray]) -> Tuple[int, int]:
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            if image.shape[0] == 3:
                return (image.shape[1], image.shape[2])
            else:
                return (image.shape[0], image.shape[1])
        return (image.shape[-2], image.shape[-1])
    return (image.shape[0], image.shape[1])


def maybe_wrap_in_list(value: Optional[Union[T, List[T]]]) -> Optional[List[T]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def equalize_batch_size(
    embeddings_batch_size: int,
    point_coordinates: Optional[List[ArrayOrTensor]],
    point_labels: Optional[List[ArrayOrTensor]],
    boxes: Optional[List[ArrayOrTensor]],
    mask_input: Optional[List[ArrayOrTensor]],
) -> Tuple[
    Optional[List[ArrayOrTensor]],
    Optional[List[ArrayOrTensor]],
    Optional[List[ArrayOrTensor]],
    Optional[List[ArrayOrTensor]],
]:
    if (
        point_coordinates is not None
        and len(point_coordinates) != embeddings_batch_size
    ):
        if len(point_coordinates) == 1:
            point_coordinates = point_coordinates * embeddings_batch_size
        else:
            raise ModelInputError(
                message=f"point_coordinates batch size ({len(point_coordinates)}) doesn't match "
                f"embeddings batch size ({embeddings_batch_size})",
                help_url="https://todo",
            )

    if point_labels is not None and len(point_labels) != embeddings_batch_size:
        if len(point_labels) == 1:
            point_labels = point_labels * embeddings_batch_size
        else:
            raise ModelInputError(
                message=f"point_labels batch size ({len(point_labels)}) doesn't match "
                f"embeddings batch size ({embeddings_batch_size})",
                help_url="https://todo",
            )

    if boxes is not None and len(boxes) != embeddings_batch_size:
        if len(boxes) == 1:
            boxes = boxes * embeddings_batch_size
        else:
            raise ModelInputError(
                message=f"boxes batch size ({len(boxes)}) doesn't match "
                f"embeddings batch size ({embeddings_batch_size})",
                help_url="https://todo",
            )

    if mask_input is not None and len(mask_input) != embeddings_batch_size:
        if len(mask_input) == 1:
            mask_input = mask_input * embeddings_batch_size
        else:
            raise ModelInputError(
                message=f"mask_input batch size ({len(mask_input)}) doesn't match "
                f"embeddings batch size ({embeddings_batch_size})",
                help_url="https://todo",
            )

    return point_coordinates, point_labels, boxes, mask_input


def pad_points(args: Dict) -> Dict:
    args = copy(args)
    if args.get("point_coords") is not None:
        point_labels = args.get("point_labels")
        if (
            not isinstance(point_labels, list)
            or len(point_labels) > 0
            and any(not isinstance(p, list) for p in point_labels)
        ):
            raise ModelInputError(
                message="point_labels must be a nested list (e.g., [[1, 0, 1]]). "
                "Each inner list should contain labels for points in a single prompt.",
                help_url="https://todo",
            )
        max_len = max(max(len(prompt) for prompt in args["point_coords"]), 1)
        for prompt in args["point_coords"]:
            for _ in range(max_len - len(prompt)):
                prompt.append([0, 0])
        for label in args["point_labels"]:
            for _ in range(max_len - len(label)):
                label.append(-1)
    return args


def choose_most_confident_prediction(
    masks: np.ndarray,
    scores: np.ndarray,
    low_resolution_logits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(masks.shape) == 3:
        masks = np.expand_dims(masks, axis=0)
        scores = np.expand_dims(scores, axis=0)
        low_resolution_logits = np.expand_dims(low_resolution_logits, axis=0)

    selected_masks, selected_scores, selected_logits = [], [], []
    for mask, score, logit in zip(masks, scores, low_resolution_logits):
        max_idx = np.argmax(score)
        selected_masks.append(mask[max_idx])
        selected_scores.append(score[max_idx])
        selected_logits.append(logit[max_idx])

    return (
        np.asarray(selected_masks),
        np.asarray(selected_scores),
        np.asarray(selected_logits),
    )


def serialize_prompt(
    point_coordinates: Optional[ArrayOrTensor],
    point_labels: Optional[ArrayOrTensor],
    boxes: Optional[ArrayOrTensor],
) -> List[dict]:
    if point_coordinates is None and point_labels is None and boxes is None:
        return []

    result = {"points": [], "box": None}

    if point_coordinates is not None and point_labels is not None:
        if isinstance(point_coordinates, torch.Tensor):
            coords_list = point_coordinates.cpu().tolist()
        elif isinstance(point_coordinates, np.ndarray):
            coords_list = point_coordinates.tolist()
        else:
            coords_list = point_coordinates

        if isinstance(point_labels, torch.Tensor):
            labels_list = point_labels.cpu().tolist()
        elif isinstance(point_labels, np.ndarray):
            labels_list = point_labels.tolist()
        else:
            labels_list = point_labels

        for coord, label in zip(coords_list, labels_list):
            result["points"].append(
                {
                    "x": coord[0] if isinstance(coord, (list, tuple)) else coord,
                    "y": coord[1] if isinstance(coord, (list, tuple)) else 0,
                    "positive": bool(label),
                }
            )

    if boxes is not None:
        if isinstance(boxes, torch.Tensor):
            result["box"] = boxes.cpu().tolist()
        elif isinstance(boxes, np.ndarray):
            result["box"] = boxes.tolist()
        else:
            result["box"] = boxes

    return [result]


def hash_serialized_prompt(serialized_prompt: List[dict]) -> str:
    serialized = json.dumps(serialized_prompt, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def attempt_load_image_mask_from_cache(
    image_hash: str,
    serialized_prompt_hash: str,
    serialized_prompt: List[dict],
    sam3_low_resolution_masks_cache: Sam3LowResolutionMasksCache,
    device: torch.device,
) -> Optional[torch.Tensor]:
    all_masks = sam3_low_resolution_masks_cache.retrieve_all_masks_for_image(
        key=image_hash
    )
    if not all_masks:
        return None
    if len(serialized_prompt) == 0:
        return None

    return find_prior_prompt_in_cache(
        serialized_prompt_hash=serialized_prompt_hash,
        serialized_prompt=serialized_prompt,
        matching_cache_entries=all_masks,
        device=device,
    )


def find_prior_prompt_in_cache(
    serialized_prompt_hash: str,
    serialized_prompt: List[dict],
    matching_cache_entries: List[SAM3MaskCacheEntry],
    device: torch.device,
) -> Optional[torch.Tensor]:
    maxed_size = 0
    best_match: Optional[SAM3MaskCacheEntry] = None
    num_points = (
        0 if not serialized_prompt else len(serialized_prompt[0].get("points", []))
    )
    if num_points <= 1:
        return None  # there is only 1 point, hence no prior prompt can be found
    desired_size = num_points - 1

    for cache_entry in matching_cache_entries[::-1]:
        is_viable = is_prompt_strict_subset(
            assumed_sub_set_prompt=(
                cache_entry.prompt_hash,
                cache_entry.serialized_prompt,
            ),
            assumed_super_set_prompt=(serialized_prompt_hash, serialized_prompt),
        )
        if not is_viable:
            continue

        cached_prompt = cache_entry.serialized_prompt
        current_size = (
            0 if not cached_prompt else len(cached_prompt[0].get("points", []))
        )
        if current_size == desired_size:
            return cache_entry.mask.to(device=device)
        if current_size >= maxed_size:
            maxed_size = current_size
            best_match = cache_entry

    if best_match is not None:
        return best_match.mask.to(device=device)
    return None


def is_prompt_strict_subset(
    assumed_sub_set_prompt: Tuple[str, List[dict]],
    assumed_super_set_prompt: Tuple[str, List[dict]],
) -> bool:
    if assumed_sub_set_prompt[0] == assumed_super_set_prompt[0]:
        return False

    super_set_copy = copy(assumed_super_set_prompt[1])
    for sub_element in assumed_sub_set_prompt[1]:
        found_match = False
        for super_element in super_set_copy:
            boxes_match = sub_element.get("box") == super_element.get("box")
            if not boxes_match:
                continue

            sub_points = {
                json.dumps(p, sort_keys=True) for p in sub_element.get("points", [])
            }
            super_points = {
                json.dumps(p, sort_keys=True) for p in super_element.get("points", [])
            }
            if sub_points <= super_points:
                super_set_copy.remove(super_element)
                found_match = True
                break

        if not found_match:
            return False

    return True


def _build_text_query(
    coco_id: int,
    h: int,
    w: int,
    text: Optional[str],
) -> FindQueryLoaded:
    return FindQueryLoaded(
        query_text=text if text is not None else "visual",
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        input_bbox=None,
        input_bbox_label=None,
        input_points=None,
        semantic_target=None,
        is_pixel_exhaustive=None,
        inference_metadata=InferenceMetadata(
            coco_image_id=coco_id,
            original_image_id=coco_id,
            original_category_id=1,
            original_size=(h, w),
            object_id=0,
            frame_index=0,
        ),
    )


def _build_visual_query(
    coco_id: int,
    h: int,
    w: int,
    boxes: Optional[List],
    labels: Optional[List],
    text: Optional[str],
) -> FindQueryLoaded:
    xyxy_pixels: List[List[float]] = []
    for b in boxes or []:
        if isinstance(b, dict):
            if "x" in b:
                x0 = float(b["x"])
                y0 = float(b["y"])
                x1 = x0 + float(b["width"])
                y1 = y0 + float(b["height"])
            else:
                x0 = float(b["x0"])
                y0 = float(b["y0"])
                x1 = float(b["x1"])
                y1 = float(b["y1"])
        elif hasattr(b, "x"):
            x0 = float(b.x)
            y0 = float(b.y)
            x1 = x0 + float(b.width)
            y1 = y0 + float(b.height)
        elif hasattr(b, "x0"):
            x0 = float(b.x0)
            y0 = float(b.y0)
            x1 = float(b.x1)
            y1 = float(b.y1)
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            x0, y0, x1, y1 = [float(v) for v in b]
        else:
            continue
        xyxy_pixels.append([x0, y0, x1, y1])

    labels_bool = [bool(int(v)) for v in (labels or [])]

    return FindQueryLoaded(
        query_text=text if text is not None else "visual",
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        input_bbox=(
            torch.tensor(xyxy_pixels, dtype=torch.float32) if xyxy_pixels else None
        ),
        input_bbox_label=(
            torch.tensor(labels_bool, dtype=torch.bool) if labels_bool else None
        ),
        input_points=None,
        semantic_target=None,
        is_pixel_exhaustive=None,
        inference_metadata=InferenceMetadata(
            coco_image_id=coco_id,
            original_image_id=coco_id,
            original_category_id=1,
            original_size=(h, w),
            object_id=0,
            frame_index=0,
        ),
    )
