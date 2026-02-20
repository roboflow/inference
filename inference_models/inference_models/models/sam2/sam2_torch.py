import hashlib
import json
from copy import copy
from threading import Lock
from typing import Dict, Generator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

from inference_models import ColorFormat
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import (
    AssumptionError,
    CorruptedModelPackageError,
    ModelInputError,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.sam2.cache import (
    Sam2ImageEmbeddingsCache,
    Sam2ImageEmbeddingsCacheNullObject,
    Sam2LowResolutionMasksCache,
    Sam2LowResolutionMasksCacheNullObject,
)
from inference_models.models.sam2.entities import (
    SAM2ImageEmbeddings,
    SAM2MaskCacheEntry,
    SAM2Prediction,
)
from inference_models.utils.file_system import read_json

ArrayOrTensor = Union[np.ndarray, torch.Tensor]
T = TypeVar("T")

MAX_SAM2_BATCH_SIZE = 8

SUPPORTED_VERSIONS = {
    "sam2_hiera_t",
    "sam2_hiera_s",
    "sam2_hiera_b+",
    "sam2_hiera_l",
    "sam2.1_hiera_t",
    "sam2.1_hiera_s",
    "sam2.1_hiera_b+",
    "sam2.1_hiera_l",
}


class SAM2Torch:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = MAX_SAM2_BATCH_SIZE,
        disable_sam2_torch_jit_transforms: bool = True,
        sam2_image_embeddings_cache: Optional[Sam2ImageEmbeddingsCache] = None,
        sam2_low_resolution_masks_cache: Optional[Sam2LowResolutionMasksCache] = None,
        sam2_allow_client_generated_hash_ids: bool = False,
        **kwargs,
    ) -> "SAM2Torch":
        if sam2_image_embeddings_cache is None:
            sam2_image_embeddings_cache = Sam2ImageEmbeddingsCacheNullObject()
        if sam2_low_resolution_masks_cache is None:
            sam2_low_resolution_masks_cache = Sam2LowResolutionMasksCacheNullObject()
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "model.pt",
                "sam_configuration.json",
            ],
        )
        try:
            version = decode_sam_version(
                config_path=model_package_content["sam_configuration.json"]
            )
        except Exception as error:
            raise CorruptedModelPackageError(
                message="Cold not decode SAM2 model version. If you see this error running inference locally, "
                "verify the contents of model package. If you see the error running on Roboflow platform - "
                "contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error
        if version not in SUPPORTED_VERSIONS:
            raise CorruptedModelPackageError(
                message=f"Detected unsupported version of SAM2 model: {version}. Supported versions: "
                f"are {SUPPORTED_VERSIONS}. If you run inference locally, verify the correctness of "
                f"SAM2 model package. If you see the error running on Roboflow platform - "
                "contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        model_config = f"{version}.yaml"
        sam2_model = build_sam2(
            model_config, model_package_content["model.pt"], device=device
        )
        transforms = SAM2Transforms(
            resolution=sam2_model.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
            disable_torch_jit=disable_sam2_torch_jit_transforms,
        )
        return cls(
            model=sam2_model,
            transform=transforms,
            device=device,
            max_batch_size=max_batch_size,
            sam2_image_embeddings_cache=sam2_image_embeddings_cache,
            sam2_low_resolution_masks_cache=sam2_low_resolution_masks_cache,
            sam2_allow_client_generated_hash_ids=sam2_allow_client_generated_hash_ids,
        )

    def __init__(
        self,
        model: SAM2Base,
        transform: SAM2Transforms,
        device: torch.device,
        max_batch_size: int,
        sam2_image_embeddings_cache: Sam2ImageEmbeddingsCache,
        sam2_low_resolution_masks_cache: Sam2LowResolutionMasksCache,
        sam2_allow_client_generated_hash_ids: bool,
    ):
        self._model = model
        self._transform = transform
        self._device = device
        self._max_batch_size = max_batch_size
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self._sam2_image_embeddings_cache = sam2_image_embeddings_cache
        self._sam2_low_resolution_masks_cache = sam2_low_resolution_masks_cache
        self._sam2_allow_client_generated_hash_ids = (
            sam2_allow_client_generated_hash_ids
        )
        self._lock = Lock()

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        use_embeddings_cache: bool = True,
        image_hashes: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[SAM2ImageEmbeddings]:
        if not self._sam2_allow_client_generated_hash_ids and image_hashes is not None:
            raise ModelInputError(
                message="When using SAM2 model, you are not allowed to provide image hashes, unless you explicitly "
                "allow it by setting `sam2_allow_client_generated_hash_ids` to `True` when loading the model. If you "
                "see this error running on Roboflow Platform - this is due to configuration of the service - contact "
                "us to get help. If you run inference locally, verify your integration making sure that the model "
                "interface is used correctly.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if image_hashes is None:
            model_input_images, image_hashes, original_image_sizes = (
                self.pre_process_images(
                    images=images,
                    **kwargs,
                )
            )
        else:
            if isinstance(image_hashes, str):
                image_hashes = [image_hashes]
            model_input_images, locally_computed_image_hashes, original_image_sizes = (
                self.pre_process_images(
                    images=images,
                    **kwargs,
                )
            )
            if len(image_hashes) != len(locally_computed_image_hashes):
                raise ModelInputError(
                    message="When using SAM2 model with client-generated `image_hashes`, the number of provided "
                    f"hashes ({len(image_hashes)}) must match the number of provided images "
                    f"({len(locally_computed_image_hashes)}). Please verify your integration",
                    help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
                )
        embeddings_from_cache: Dict[int, SAM2ImageEmbeddings] = {}
        images_to_compute, hashes_of_images_to_compute, sizes_of_images_to_compute = (
            [],
            [],
            [],
        )
        for idx, (image, image_hash, image_size) in enumerate(
            zip(model_input_images, image_hashes, original_image_sizes)
        ):
            cache_content = None
            if use_embeddings_cache:
                cache_content = self._sam2_image_embeddings_cache.retrieve_embeddings(
                    key=image_hash
                )
            if cache_content is not None:
                cache_content = cache_content.to(device=self._device)
                embeddings_from_cache[idx] = cache_content
            else:
                images_to_compute.append(image)
                hashes_of_images_to_compute.append(image_hash)
                sizes_of_images_to_compute.append(image_size)
        if len(images_to_compute) > 0:
            images_to_compute = torch.stack(images_to_compute, dim=0)
            computed_embeddings = self.forward_image_embeddings(
                model_input_images=images_to_compute,
                image_hashes=hashes_of_images_to_compute,
                original_image_sizes=sizes_of_images_to_compute,
            )
            computed_embeddings_idx = 0
            result_embeddings = []
            for i in range(len(model_input_images)):
                if i in embeddings_from_cache:
                    result_embeddings.append(embeddings_from_cache[i])
                else:
                    result_embeddings.append(
                        computed_embeddings[computed_embeddings_idx]
                    )
                    computed_embeddings_idx += 1
        else:
            result_embeddings = [
                embeddings_from_cache[i] for i in range(len(model_input_images))
            ]
        if use_embeddings_cache:
            for embeddings in result_embeddings:
                self._sam2_image_embeddings_cache.save_embeddings(
                    key=embeddings.image_hash, embeddings=embeddings
                )
        return result_embeddings

    def pre_process_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[torch.Tensor, List[str], List[Tuple[int, int]]]:
        if isinstance(images, torch.Tensor):
            images = images.to(device=self._device)
            if len(images.shape) == 4:
                image_hashes = [compute_image_hash(image=image) for image in images]
                original_image_sizes = [tuple(images.shape[2:4])] * images.shape[0]
                model_input_images = self._transform.transforms(images / 255.0)
            else:
                image_hashes = [compute_image_hash(image=images)]
                original_image_sizes = [tuple(images.shape[1:3])]
                model_input_images = self._transform.transforms(
                    (images / 255).unsqueeze(dim=0)
                )
        else:
            if isinstance(images, list):
                image_hashes = [compute_image_hash(image=image) for image in images]
                original_image_sizes = []
                model_input_images = []
                for image in images:
                    if isinstance(image, np.ndarray):
                        original_image_sizes.append(image.shape[:2])
                        input_image = self._transform(image).to(self._device)
                        model_input_images.append(input_image)
                    else:
                        original_image_sizes.append(tuple(image.shape[1:3]))
                        image = image.to(self._device)
                        input_image = self._transform.transforms(image / 255)
                        model_input_images.append(input_image)
                model_input_images = torch.stack(model_input_images, dim=0)
            else:
                image_hashes = [compute_image_hash(image=images)]
                original_image_sizes = [images.shape[:2]]
                model_input_images = (
                    self._transform(images).to(self._device).unsqueeze(dim=0)
                )
        return model_input_images, image_hashes, original_image_sizes

    @torch.inference_mode()
    def forward_image_embeddings(
        self,
        model_input_images: torch.Tensor,
        image_hashes: List[str],
        original_image_sizes: List[Tuple[int, int]],
        **kwargs,
    ) -> List[SAM2ImageEmbeddings]:
        result_embeddings = []
        for i in range(0, model_input_images.shape[0], self._max_batch_size):
            input_images_batch = model_input_images[
                i : i + self._max_batch_size
            ].contiguous()
            batch_size = input_images_batch.shape[0]
            with self._lock:
                backbone_out = self._model.forward_image(input_images_batch)
                _, vision_feats, _, _ = self._model._prepare_backbone_features(
                    backbone_out
                )
            if self._model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self._model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(
                    vision_feats[::-1], self._bb_feat_sizes[::-1]
                )
            ][::-1]
            for image_idx in range(batch_size):
                image_embeddings = feats[-1][image_idx].unsqueeze(dim=0)
                high_resolution_features = [
                    feature[image_idx].unsqueeze(dim=0) for feature in feats[:-1]
                ]
                result_embeddings.append(
                    SAM2ImageEmbeddings(
                        image_hash=image_hashes[i + image_idx],
                        image_size_hw=original_image_sizes[i + image_idx],
                        embeddings=image_embeddings,
                        high_resolution_features=high_resolution_features,
                    )
                )
        return result_embeddings

    def segment_images(
        self,
        images: Optional[
            Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
        ] = None,
        embeddings: Optional[
            Union[List[SAM2ImageEmbeddings], SAM2ImageEmbeddings]
        ] = None,
        image_hashes: Optional[Union[str, List[str]]] = None,
        point_coordinates: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        point_labels: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        boxes: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        mask_input: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        multi_mask_output: bool = True,
        return_logits: bool = False,
        input_color_format: Optional[ColorFormat] = None,
        mask_threshold: Optional[float] = None,
        load_from_mask_input_cache: bool = False,
        save_to_mask_input_cache: bool = False,
        use_embeddings_cache: bool = True,
        **kwargs,
    ) -> List[SAM2Prediction]:
        if images is None and embeddings is None and image_hashes is None:
            raise ModelInputError(
                message="Attempted to use SAM2 model segment_images(...) method not providing valid input - "
                "neither `images` nor `embeddings` nor `image_hashes` parameter is given. If you run inference locally, "
                "verify your integration making sure that the model interface is used correctly. Running "
                "on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if images is not None:
            embeddings = self.embed_images(
                images=images,
                input_color_format=input_color_format,
                use_embeddings_cache=use_embeddings_cache,
                image_hashes=image_hashes,
                **kwargs,
            )
        elif image_hashes is not None:
            if isinstance(image_hashes, str):
                image_hashes = [image_hashes]
            if (
                not use_embeddings_cache
                or not self._sam2_allow_client_generated_hash_ids
            ):
                raise ModelInputError(
                    message="Attempted to use SAM2 model segment_images(...) method providing `image_hashes` "
                    "without enabling `use_embeddings_cache` or `sam_allow_client_generated_hash_ids` which is not "
                    "allowed. If you run inference locally, verify your integration making sure that the model "
                    "interface is used correctly. Running on Roboflow platform - contact us to get help.",
                    help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
                )
            embeddings = []
            for image_hash in image_hashes:
                cache_content = self._sam2_image_embeddings_cache.retrieve_embeddings(
                    key=image_hash
                )
                if cache_content is None:
                    raise ModelInputError(
                        message=f"Attempted to use SAM model segment_images(...) method providing `image_hashes` "
                        f"for which no embeddings were found in the cache. This may be an effect of cache expiry or "
                        f"invalid integration.",
                        help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
                    )
                embeddings.append(cache_content)
        else:
            embeddings = maybe_wrap_in_list(value=embeddings)
        image_hashes = [e.image_hash for e in embeddings]
        original_image_sizes = [e.image_size_hw for e in embeddings]
        point_coordinates = maybe_wrap_in_list(value=point_coordinates)
        point_labels = maybe_wrap_in_list(value=point_labels)
        boxes = maybe_wrap_in_list(value=boxes)
        mask_input = maybe_wrap_in_list(value=mask_input)
        point_coordinates, point_labels, boxes, mask_input = equalize_batch_size(
            embeddings_batch_size=len(embeddings),
            point_coordinates=point_coordinates,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
        )
        point_coordinates, point_labels, boxes, mask_input = pre_process_prompts(
            point_coordinates=point_coordinates,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
            device=self._device,
            transform=self._transform,
            original_image_sizes=original_image_sizes,
        )
        predictions = []
        for (
            image_embedding,
            image_hash,
            image_size,
            image_point_coordinates,
            image_point_labels,
            image_boxes,
            image_mask_input,
        ) in generate_model_inputs(
            embeddings=embeddings,
            image_hashes=image_hashes,
            original_image_sizes=original_image_sizes,
            point_coordinates=point_coordinates,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
        ):
            serialized_prompt, prompt_hash = None, None
            if save_to_mask_input_cache or load_from_mask_input_cache:
                serialized_prompt = serialize_prompt(
                    point_coordinates=image_point_coordinates,
                    point_labels=image_point_labels,
                    boxes=image_boxes,
                )
                prompt_hash = hash_serialized_prompt(
                    serialized_prompt=serialized_prompt
                )
            if image_mask_input is None and load_from_mask_input_cache:
                image_mask_input = attempt_load_image_mask_from_cache(
                    image_hash=image_hash,
                    serialized_prompt_hash=prompt_hash,
                    serialized_prompt=serialized_prompt,
                    sam2_low_resolution_masks_cache=self._sam2_low_resolution_masks_cache,
                    device=self._device,
                )
            with self._lock:
                prediction = predict_for_single_image(
                    model=self._model,
                    transform=self._transform,
                    embeddings=image_embedding,
                    original_image_size=image_size,
                    point_coordinates=image_point_coordinates,
                    point_labels=image_point_labels,
                    boxes=image_boxes,
                    mask_input=image_mask_input,
                    multi_mask_output=multi_mask_output,
                    return_logits=return_logits,
                    mask_threshold=mask_threshold,
                )
            if save_to_mask_input_cache and len(prediction[0].shape) == 3:
                max_score_id = torch.argmax(prediction[1]).item()
                mask = SAM2MaskCacheEntry(
                    prompt_hash=prompt_hash,
                    serialized_prompt=serialized_prompt,
                    mask=prediction[2][max_score_id].unsqueeze(dim=0),
                )
                self._sam2_low_resolution_masks_cache.save_mask(
                    key=image_hash,
                    mask=mask,
                )
            parsed_prediction = SAM2Prediction(
                masks=prediction[0],
                scores=prediction[1],
                logits=prediction[2],
            )
            predictions.append(parsed_prediction)
        return predictions


def decode_sam_version(config_path: str) -> str:
    config = read_json(path=config_path)
    version = config["version"]
    if not isinstance(version, str):
        raise ValueError("Could not decode SAM model version")
    return version


def compute_image_hash(image: Union[torch.Tensor, np.ndarray]) -> str:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    return hash_function(value=image.tobytes())


def hash_function(value: Union[str, bytes]) -> str:
    return hashlib.sha1(value).hexdigest()


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
        if len(point_coordinates) != 1:
            raise ModelInputError(
                message="When using SAM2 model, parameter `point_coordinates` was provided with invalid "
                f"value indicating different input batch size ({len(point_coordinates)}) than provided "
                f"images / embeddings ({embeddings_batch_size}). If you run inference locally, verify your "
                "integration making sure that the model interface is used correctly. "
                "Running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        point_coordinates = point_coordinates * embeddings_batch_size
    if point_labels is not None and len(point_labels) != embeddings_batch_size:
        if len(point_labels) != 1:
            raise ModelInputError(
                message="When using SAM2 model, parameter `point_labels` was provided with invalid "
                f"value indicating different input batch size ({len(point_labels)}) than provided "
                f"images / embeddings ({embeddings_batch_size}). If you run inference locally, verify your "
                "integration making sure that the model interface is used correctly. "
                "Running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        point_labels = point_labels * embeddings_batch_size
    if boxes is not None and len(boxes) != embeddings_batch_size:
        if len(boxes) != 1:
            raise ModelInputError(
                message="When using SAM2 model, parameter `boxes` was provided with invalid "
                f"value indicating different input batch size ({len(boxes)}) than provided "
                f"images / embeddings ({embeddings_batch_size}). If you run inference locally, verify your "
                "integration making sure that the model interface is used correctly. "
                "Running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        boxes = boxes * embeddings_batch_size
    if mask_input is not None and len(mask_input) != embeddings_batch_size:
        if len(mask_input) != 1:
            raise ModelInputError(
                message="When using SAM2 model, parameter `mask_input` was provided with invalid "
                f"value indicating different input batch size ({len(mask_input)}) than provided "
                f"images / embeddings ({embeddings_batch_size}). If you run inference locally, verify your "
                "integration making sure that the model interface is used correctly. "
                "Running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        mask_input = mask_input * embeddings_batch_size
    prompts_first_dimension_characteristics = set()
    at_max_one_box_expected = False
    if point_coordinates is not None:
        point_coordinates_characteristic = "-".join(
            [str(p.shape[0]) for p in point_coordinates]
        )
        prompts_first_dimension_characteristics.add(point_coordinates_characteristic)
        points_dimensions = set(len(p.shape) for p in point_coordinates)
        if len(points_dimensions) != 1:
            raise ModelInputError(
                message="When using SAM2 model, in scenario when combination of `point_coordinates` provided with "
                "different shapes for different input images, which makes the input invalid. "
                "If you run inference locally, verify your integration making sure that the model interface is "
                "used correctly. Running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if points_dimensions.pop() == 2:
            at_max_one_box_expected = True
    if point_labels is not None:
        point_labels_characteristic = "-".join([str(l.shape[0]) for l in point_labels])
        prompts_first_dimension_characteristics.add(point_labels_characteristic)
    if len(prompts_first_dimension_characteristics) > 1:
        raise ModelInputError(
            message="When using SAM2 model, in scenario when combination of `point_coordinates` and `point_labels` "
            "provided, the model expect identical number of elements for each prompt component. "
            "If you run inference locally, verify your integration making sure that the model interface is "
            "used correctly. Running on Roboflow platform - contact us to get help.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if boxes is not None:
        boxes_characteristic = "-".join(
            [str(b.shape[0]) if len(b.shape) > 1 else "1" for b in boxes]
        )
        prompts_first_dimension_characteristics.add(boxes_characteristic)
        if at_max_one_box_expected:
            if not all(b.shape[0] == 1 if len(b.shape) > 1 else True for b in boxes):
                raise ModelInputError(
                    message="When using SAM2 model, with `point_coordinates` provided for single box, each box in "
                    "`boxes` parameter must only define single bounding box."
                    "If you run inference locally, verify your integration making sure that the model "
                    "interface is used correctly. Running on Roboflow platform - contact us to get help.",
                    help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
                )
        elif len(prompts_first_dimension_characteristics) > 1:
            raise ModelInputError(
                message="When using SAM2 model, in scenario when combination of `point_coordinates`, `point_labels`, "
                "`boxes` provided, the model expect identical number of elements for each prompt component. "
                "If you run inference locally, verify your integration making sure that the model interface is "
                "used correctly. Running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
    if mask_input is not None:
        mask_input = [i[None, :, :] if len(i.shape) == 2 else i for i in mask_input]
        if any(len(i.shape) != 3 or i.shape[0] != 1 for i in mask_input):
            raise ModelInputError(
                message="When using SAM2 model with `mask_input`, each mask must be 3D tensor of shape (1, H, W). "
                "If you run inference locally, verify your integration making sure that the model interface is "
                "used correctly. Running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
    return point_coordinates, point_labels, boxes, mask_input


def generate_model_inputs(
    embeddings: List[SAM2ImageEmbeddings],
    image_hashes: List[str],
    original_image_sizes: List[Tuple[int, int]],
    point_coordinates: Optional[List[torch.Tensor]],
    point_labels: Optional[List[torch.Tensor]],
    boxes: Optional[List[torch.Tensor]],
    mask_input: Optional[List[torch.Tensor]],
) -> Generator[
    Tuple[
        SAM2ImageEmbeddings,
        str,
        Tuple[int, int],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ],
    None,
    None,
]:
    if point_coordinates is None:
        point_coordinates = [None] * len(embeddings)
    if point_labels is None:
        point_labels = [None] * len(embeddings)
    if boxes is None:
        boxes = [None] * len(embeddings)
    if mask_input is None:
        mask_input = [None] * len(embeddings)
    for embedding, hash_value, image_size, coords, labels, box, mask in zip(
        embeddings,
        image_hashes,
        original_image_sizes,
        point_coordinates,
        point_labels,
        boxes,
        mask_input,
    ):
        yield embedding, hash_value, image_size, coords, labels, box, mask


def pre_process_prompts(
    point_coordinates: Optional[List[ArrayOrTensor]],
    point_labels: Optional[List[ArrayOrTensor]],
    boxes: Optional[List[ArrayOrTensor]],
    mask_input: Optional[List[ArrayOrTensor]],
    device: torch.device,
    transform: SAM2Transforms,
    original_image_sizes: List[Tuple[int, int]],
    normalize_coordinates: bool = True,
) -> Tuple[
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
]:
    (
        processed_point_coordinates,
        processed_point_labels,
        processed_boxes,
        processed_mask_input,
    ) = (None, None, None, None)
    if point_labels is not None and point_coordinates is None:
        raise ModelInputError(
            message="When using SAM2 model, provided `point_coordinates` without `point_labels` which makes "
            "invalid input. If you run inference locally, verify your integration making sure that the "
            "model interface is used correctly. Running on Roboflow platform - contact us to get help.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if point_coordinates is not None:
        if point_labels is None:
            raise ModelInputError(
                message="When using SAM2 model, provided `point_coordinates` without `point_labels` which makes "
                "invalid input. If you run inference locally, verify your integration making sure that the "
                "model interface is used correctly. Running on Roboflow platform - contact us to get help.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        processed_point_coordinates = []
        processed_point_labels = []
        for single_label, single_point_coordinates, image_size in zip(
            point_labels, point_coordinates, original_image_sizes
        ):
            if isinstance(single_point_coordinates, torch.Tensor):
                single_point_coordinates = single_point_coordinates.to(
                    dtype=torch.float, device=device
                )
            else:
                single_point_coordinates = torch.as_tensor(
                    single_point_coordinates, dtype=torch.float, device=device
                )
            single_point_coordinates = transform.transform_coords(
                single_point_coordinates,
                normalize=normalize_coordinates,
                orig_hw=image_size,
            )
            dimension_to_unsqueeze = len(single_point_coordinates.shape) == 2
            if dimension_to_unsqueeze:
                single_point_coordinates = single_point_coordinates[None, ...]
            processed_point_coordinates.append(single_point_coordinates)
            if isinstance(single_label, torch.Tensor):
                single_label = single_label.to(dtype=torch.int, device=device)
            else:
                single_label = torch.as_tensor(
                    single_label, dtype=torch.int, device=device
                )
            if dimension_to_unsqueeze:
                single_label = single_label[None, ...]
            processed_point_labels.append(single_label)
    if boxes is not None:
        processed_boxes = []
        for box, image_size in zip(boxes, original_image_sizes):
            if isinstance(box, torch.Tensor):
                box = box.to(dtype=torch.float, device=device)
            else:
                box = torch.as_tensor(box, dtype=torch.float, device=device)
            box = transform.transform_boxes(
                box,
                normalize=normalize_coordinates,
                orig_hw=image_size,
            )  # Bx2x2
            processed_boxes.append(box)
    if mask_input is not None:
        processed_mask_input = []
        for single_mask in mask_input:
            if isinstance(single_mask, torch.Tensor):
                single_mask = single_mask.to(dtype=torch.float, device=device)
            else:
                single_mask = torch.as_tensor(
                    single_mask, dtype=torch.float, device=device
                )
            if len(single_mask.shape) == 3:
                single_mask = single_mask[None, :, :, :]
            processed_mask_input.append(single_mask)
    return (
        processed_point_coordinates,
        processed_point_labels,
        processed_boxes,
        processed_mask_input,
    )


@torch.inference_mode()
def predict_for_single_image(
    model: SAM2Base,
    transform: SAM2Transforms,
    embeddings: SAM2ImageEmbeddings,
    original_image_size: Tuple[int, int],
    point_coordinates: Optional[torch.Tensor],
    point_labels: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor] = None,
    mask_input: Optional[torch.Tensor] = None,
    multi_mask_output: bool = True,
    return_logits: bool = False,
    mask_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if point_coordinates is not None:
        concat_points = (point_coordinates, point_labels)
    else:
        concat_points = None
    if boxes is not None:
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
        box_labels = box_labels.repeat(boxes.size(0), 1)
        # we merge "boxes" and "points" into a single "concat_points" input (where
        # boxes are added at the beginning) to sam_prompt_encoder
        if concat_points is not None:
            concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
            concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
            concat_points = (concat_coords, concat_labels)
        else:
            concat_points = (box_coords, box_labels)
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=mask_input,
    )
    batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
    low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
        image_embeddings=embeddings.embeddings,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multi_mask_output,
        repeat_image=batched_mode,
        high_res_features=embeddings.high_resolution_features,
    )
    masks = transform.postprocess_masks(low_res_masks, original_image_size)
    low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
    if not return_logits:
        masks = masks > (mask_threshold or 0.0)
    if masks.shape[0] == 1:
        return masks[0], iou_predictions[0], low_res_masks[0]
    else:
        return masks, iou_predictions, low_res_masks


def serialize_prompt(
    point_coordinates: Optional[torch.Tensor],
    point_labels: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor],
) -> List[dict]:
    if point_coordinates is None and point_labels is None and boxes is None:
        return []
    sizes = set()
    if point_coordinates is not None:
        sizes.add(point_coordinates.shape[0])
    if point_labels is not None:
        sizes.add(point_labels.shape[0])
    if boxes is not None:
        sizes.add(boxes.shape[0])
    if len(sizes) != 1:
        raise AssumptionError(
            message="In SAM2 implementation, after pre-processing, all prompt elements must have the same "
            "leading dimension. This assumption just got violated. This is most likely a bug. "
            "You can help us sorting out this problem by submitting an issue: "
            "https://github.com/roboflow/inference/issues",
            help_url="https://todo",
        )
    broadcast_size = sizes.pop()
    point_coordinates_list = (
        point_coordinates.tolist()
        if point_coordinates is not None
        else [None] * broadcast_size
    )
    point_labels_list = (
        point_labels.tolist() if point_labels is not None else [None] * broadcast_size
    )
    boxes_list = (
        boxes.reshape(-1).tolist() if boxes is not None else [None] * broadcast_size
    )
    results = []
    for points, labels, box in zip(
        point_coordinates_list, point_labels_list, boxes_list
    ):
        points_serialized = []
        if points is not None and labels is not None:
            for point, label in zip(points, labels):
                points_serialized.append(
                    {
                        "x": (
                            point[0].item()
                            if isinstance(point[0], torch.Tensor)
                            else point[0]
                        ),
                        "y": (
                            point[1].item()
                            if isinstance(point[1], torch.Tensor)
                            else point[1]
                        ),
                        "positive": (
                            label.item() if isinstance(labels, torch.Tensor) else label
                        ),
                    }
                )
        if box is not None:
            box_serialized = box
        else:
            box_serialized = None
        results.append({"points": points_serialized, "box": box_serialized})
    return results


def hash_serialized_prompt(serialized_prompt: List[dict]) -> str:
    serialized = json.dumps(serialized_prompt, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def attempt_load_image_mask_from_cache(
    image_hash: str,
    serialized_prompt_hash: str,
    serialized_prompt: List[dict],
    sam2_low_resolution_masks_cache: Sam2LowResolutionMasksCache,
    device: torch.device,
) -> Optional[torch.Tensor]:
    all_masks_for_image = sam2_low_resolution_masks_cache.retrieve_all_masks_for_image(
        key=image_hash
    )
    if not all_masks_for_image:
        return None
    if len(serialized_prompt) == 0:
        return None
    return find_prior_prompt_in_cache(
        serialized_prompt_hash=serialized_prompt_hash,
        serialized_prompt=serialized_prompt,
        matching_cache_entries=all_masks_for_image,
        device=device,
    )


def find_prior_prompt_in_cache(
    serialized_prompt_hash: str,
    serialized_prompt: List[dict],
    matching_cache_entries: List[SAM2MaskCacheEntry],
    device: torch.device,
) -> Optional[torch.Tensor]:
    maxed_size = 0
    best_match: Optional[SAM2MaskCacheEntry] = None
    desired_size = len(serialized_prompt) - 1
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

        # short circuit search if we find prompt with one less point (most recent possible mask)
        current_cache_entry_prompt_size = len(cache_entry.serialized_prompt)
        if current_cache_entry_prompt_size == desired_size:
            return cache_entry.mask.to(device=device)
        if current_cache_entry_prompt_size >= maxed_size:
            maxed_size = current_cache_entry_prompt_size
            best_match = cache_entry
    return best_match.mask.to(device=device)


def is_prompt_strict_subset(
    assumed_sub_set_prompt: Tuple[str, List[dict]],
    assumed_super_set_prompt: Tuple[str, List[dict]],
) -> bool:
    if assumed_sub_set_prompt[0] == assumed_super_set_prompt[0]:
        return False
    super_set_prompt_copy = copy(assumed_super_set_prompt[1])
    for sub_set_prompt_element in assumed_sub_set_prompt[1]:
        found_match = False
        for super_set_prompt_element in super_set_prompt_copy:
            boxes_matching = (
                sub_set_prompt_element["box"] == super_set_prompt_element["box"]
            )
            if not boxes_matching:
                continue
            sub_set_prompt_element_points = {
                get_hashable_point(point=point)
                for point in sub_set_prompt_element.get("points", [])
            }
            super_set_prompt_element_points = {
                get_hashable_point(point=point)
                for point in super_set_prompt_element.get("points", [])
            }
            if sub_set_prompt_element_points <= super_set_prompt_element_points:
                super_set_prompt_copy.remove(super_set_prompt_element)
                found_match = True
                break
        if not found_match:
            return False
    # every prompt in subset has a matching super prompt
    return True


def get_hashable_point(point: dict) -> str:
    return json.dumps(point, sort_keys=True, separators=(",", ":"))
