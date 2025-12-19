import hashlib
from typing import Dict, Generator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from inference_exp import ColorFormat
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.errors import CorruptedModelPackageError, ModelInputError
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.sam.cache import (
    SamImageEmbeddingsCache,
    SamImageEmbeddingsCacheNullObject,
    SamLowResolutionMasksCache,
    SamLowResolutionMasksCacheNullObject,
)
from inference_exp.models.sam.entities import SAMImageEmbeddings, SAMPrediction
from inference_exp.utils.file_system import read_json
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

T = TypeVar("T")

MAX_SAM_BATCH_SIZE = 8

ArrayOrTensor = Union[np.ndarray, torch.Tensor]


class SAMTorch:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = MAX_SAM_BATCH_SIZE,
        sam_image_embeddings_cache: Optional[SamImageEmbeddingsCache] = None,
        sam_low_resolution_masks_cache: Optional[SamLowResolutionMasksCache] = None,
        **kwargs,
    ) -> "SAMTorch":
        if sam_image_embeddings_cache is None:
            sam_image_embeddings_cache = SamImageEmbeddingsCacheNullObject()
        if sam_low_resolution_masks_cache is None:
            sam_low_resolution_masks_cache = SamLowResolutionMasksCacheNullObject()
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "model.pth",
                "sam_configuration.json",
            ],
        )
        try:
            version = decode_sam_version(
                config_path=model_package_content["sam_configuration.json"]
            )
        except Exception as error:
            raise CorruptedModelPackageError(
                message="Cold not decode SAM model version. If you see this error running inference locally, "
                "verify the contents of model package. If you see the error running on Roboflow platform - "
                "contact us to get help.",
                help_url="https://todo",
            ) from error
        try:
            sam_model = sam_model_registry[version](
                checkpoint=model_package_content["model.pth"]
            ).to(device)
        except Exception as error:
            raise CorruptedModelPackageError(
                message=f"Cold not decode initialize SAM model - cause: {error} If you see this error running "
                f"locally - verify installation of inference and contents of model package. If you use "
                f"Roboflow platform, contact us to get help.",
                help_url="https://todo",
            ) from error
        transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        return cls(
            model=sam_model,
            transform=transform,
            device=device,
            max_batch_size=max_batch_size,
            sam_image_embeddings_cache=sam_image_embeddings_cache,
            sam_low_resolution_masks_cache=sam_low_resolution_masks_cache,
        )

    def __init__(
        self,
        model: Sam,
        transform: ResizeLongestSide,
        device: torch.device,
        max_batch_size: int,
        sam_image_embeddings_cache: SamImageEmbeddingsCache,
        sam_low_resolution_masks_cache: SamLowResolutionMasksCache,
    ):
        self._model = model
        self._transform = transform
        self._device = device
        self._max_batch_size = max_batch_size
        self._sam_image_embeddings_cache = sam_image_embeddings_cache
        self._sam_low_resolution_masks_cache = sam_low_resolution_masks_cache

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        use_embeddings_cache: bool = True,
        **kwargs,
    ) -> List[SAMImageEmbeddings]:
        model_input_images, image_hashes, original_image_sizes = (
            self.pre_process_images(
                images=images,
                input_color_format=input_color_format,
                **kwargs,
            )
        )
        embeddings_from_cache: Dict[int, SAMImageEmbeddings] = {}
        images_to_compute = []
        for idx, (image, image_hash) in enumerate(
            zip(model_input_images, image_hashes)
        ):
            cache_content = None
            if use_embeddings_cache:
                cache_content = self._sam_image_embeddings_cache.retrieve_embeddings(
                    key=image_hash
                )
            if cache_content is not None:
                cache_content = cache_content.to(device=self._device)
                embeddings_from_cache[idx] = cache_content
            else:
                images_to_compute.append(image)
        if len(images_to_compute) > 0:
            images_to_compute = torch.stack(images_to_compute, dim=0)
            computed_embeddings = self.forward_image_embeddings(
                model_input_images=images_to_compute,
            )
            computed_embeddings_idx = 0
            result_embeddings = []
            for i in range(len(model_input_images)):
                if i in embeddings_from_cache:
                    result_embeddings.append(embeddings_from_cache[i].embeddings)
                else:
                    result_embeddings.append(
                        computed_embeddings[computed_embeddings_idx]
                    )
                    computed_embeddings_idx += 1
        else:
            result_embeddings = [
                embeddings_from_cache[i].embeddings
                for i in range(len(model_input_images))
            ]
        results = []
        for image_hash, image_size, image_embeddings in zip(
            image_hashes, original_image_sizes, result_embeddings
        ):
            result = SAMImageEmbeddings(
                image_hash=image_hash,
                image_size_hw=image_size,
                embeddings=image_embeddings,
            )
            results.append(result)
            if use_embeddings_cache:
                self._sam_image_embeddings_cache.save_embeddings(
                    key=image_hash, embeddings=result
                )
        return results

    def pre_process_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[str], List[Tuple[int, int]]]:
        if isinstance(images, torch.Tensor):
            images = images.to(device=self._device)
            if len(images.shape) == 4:
                image_hashes = [compute_image_hash(image=image) for image in images]
                if input_color_format == "bgr":
                    images = images[:, :-1, :, :].contiguous()
                original_image_sizes = [tuple(images.shape[2:4])] * images.shape[0]
                model_input_images = self._transform.apply_image_torch(image=images)
            else:
                image_hashes = [compute_image_hash(image=images)]
                if input_color_format == "bgr":
                    images = images[::-1, :, :].contiguous()
                original_image_sizes = [tuple(images.shape[1:3])]
                model_input_images = self._transform.apply_image_torch(
                    image=images.unsqueeze(dim=0)
                )
        else:
            if isinstance(images, list):
                image_hashes = [compute_image_hash(image=image) for image in images]
                original_image_sizes = []
                model_input_images = []
                for image in images:
                    if isinstance(image, np.ndarray):
                        original_image_sizes.append(image.shape[:2])
                        if input_color_format in {None, "bgr"}:
                            image = np.ascontiguousarray(image[:, :, ::-1])
                        input_image = self._transform.apply_image(image=image)
                        input_image = (
                            torch.as_tensor(input_image, device=self._device)
                            .permute(2, 0, 1)
                            .contiguous()
                        )
                        model_input_images.append(input_image)
                    else:
                        original_image_sizes.append(tuple(image.shape[1:3]))
                        if input_color_format == "bgr":
                            image = image[::-1, :, :].contiguous()
                        input_image = self._transform.apply_image_torch(
                            image=image.unsqueeze(dim=0)
                        )[0]
                        model_input_images.append(input_image)
                model_input_images = torch.stack(model_input_images, dim=0)
            else:
                image_hashes = [compute_image_hash(image=images)]
                original_image_sizes = [images.shape[:2]]
                if input_color_format in {None, "bgr"}:
                    images = np.ascontiguousarray(images[:, :, ::-1])
                model_input_images = self._transform.apply_image(image=images)
                model_input_images = (
                    torch.as_tensor(model_input_images, device=self._device)
                    .permute(2, 0, 1)
                    .contiguous()[None, :, :, :]
                )
        return model_input_images, image_hashes, original_image_sizes

    @torch.inference_mode()
    def forward_image_embeddings(
        self, model_input_images: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        result_embeddings = []
        for i in range(0, model_input_images.shape[0], self._max_batch_size):
            input_images_batch = model_input_images[
                i : i + self._max_batch_size
            ].contiguous()
            pre_processed_images_batch = self._model.preprocess(input_images_batch)
            batch_embeddings = self._model.image_encoder(pre_processed_images_batch).to(
                device=self._device
            )
            result_embeddings.append(batch_embeddings)
        return torch.cat(result_embeddings, dim=0)

    def segment_images(
        self,
        images: Optional[
            Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
        ] = None,
        embeddings: Optional[
            Union[List[SAMImageEmbeddings], SAMImageEmbeddings]
        ] = None,
        point_coordinates: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        point_labels: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        boxes: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        mask_input: Optional[Union[List[ArrayOrTensor], ArrayOrTensor]] = None,
        multi_mask_output: bool = True,
        return_logits: bool = False,
        input_color_format: Optional[ColorFormat] = None,
        mask_threshold: Optional[float] = None,
        enforce_mask_input: bool = False,
        use_mask_input_cache: bool = True,
        use_embeddings_cache: bool = True,
        **kwargs,
    ) -> List[SAMPrediction]:
        if images is None and embeddings is None:
            raise ModelInputError(
                message="Attempted to use SAM model segment_images(...) method not providing valid input - "
                "neither `images` nor `embeddings` parameter is given. If you run inference locally, "
                "verify your integration making sure that the model interface is used correctly. Running "
                "on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        if images is not None:
            embeddings = self.embed_images(
                images=images,
                input_color_format=input_color_format,
                use_embeddings_cache=use_embeddings_cache,
                **kwargs,
            )
        else:
            embeddings = maybe_wrap_in_list(value=embeddings)
        embeddings_tensors = [e.embeddings.to(self._device) for e in embeddings]
        image_hashes = [e.image_hash for e in embeddings]
        original_image_sizes = [e.image_size_hw for e in embeddings]
        point_coordinates = maybe_wrap_in_list(value=point_coordinates)
        point_labels = maybe_wrap_in_list(value=point_labels)
        boxes = maybe_wrap_in_list(value=boxes)
        mask_input = maybe_wrap_in_list(value=mask_input)
        masks_from_the_cache = [
            (
                self._sam_low_resolution_masks_cache.retrieve_mask(key=image_hash)
                if use_mask_input_cache
                else None
            )
            for image_hash in image_hashes
        ]
        if enforce_mask_input and mask_input is None:
            if not all(e is not None for e in masks_from_the_cache):
                raise ModelInputError(
                    message="Attempted to use SAM model segment_images(...) method enforcing the presence of "
                    "low-resolution mask input and not providing the mask explicitly (causing fallback to "
                    "SAM cache lookup which  failed for at least one image) - this problem may be temporary, "
                    "but may also be a result of bug or invalid integration. If you run inference locally, "
                    "verify your integration making sure that the model interface is used correctly. Running "
                    "on Roboflow platform - contact us to get help.",
                    help_url="https://todo",
                )
            mask_input = [mask.to(self._device) for mask in masks_from_the_cache]
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
            embeddings=embeddings_tensors,
            image_hashes=image_hashes,
            original_image_sizes=original_image_sizes,
            point_coordinates=point_coordinates,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
        ):
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
            if use_mask_input_cache and len(prediction[0].shape) == 3:
                max_score_id = torch.argmax(prediction[1]).item()
                self._sam_low_resolution_masks_cache.save_mask(
                    key=image_hash, mask=prediction[2][max_score_id].unsqueeze(dim=0)
                )
            parsed_prediction = SAMPrediction(
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
                message="When using SAM model, parameter `point_coordinates` was provided with invalid "
                f"value indicating different input batch size ({len(point_coordinates)}) than provided "
                f"images / embeddings ({embeddings_batch_size}). If you run inference locally, verify your "
                "integration making sure that the model interface is used correctly. "
                "Running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        point_coordinates = point_coordinates * embeddings_batch_size
    if point_labels is not None and len(point_labels) != embeddings_batch_size:
        if len(point_labels) != 1:
            raise ModelInputError(
                message="When using SAM model, parameter `point_labels` was provided with invalid "
                f"value indicating different input batch size ({len(point_labels)}) than provided "
                f"images / embeddings ({embeddings_batch_size}). If you run inference locally, verify your "
                "integration making sure that the model interface is used correctly. "
                "Running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        point_labels = point_labels * embeddings_batch_size
    if boxes is not None and len(boxes) != embeddings_batch_size:
        if len(boxes) != 1:
            raise ModelInputError(
                message="When using SAM model, parameter `boxes` was provided with invalid "
                f"value indicating different input batch size ({len(boxes)}) than provided "
                f"images / embeddings ({embeddings_batch_size}). If you run inference locally, verify your "
                "integration making sure that the model interface is used correctly. "
                "Running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        boxes = boxes * embeddings_batch_size
    if mask_input is not None and len(mask_input) != embeddings_batch_size:
        if len(mask_input) != 1:
            raise ModelInputError(
                message="When using SAM model, parameter `mask_input` was provided with invalid "
                f"value indicating different input batch size ({len(mask_input)}) than provided "
                f"images / embeddings ({embeddings_batch_size}). If you run inference locally, verify your "
                "integration making sure that the model interface is used correctly. "
                "Running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        mask_input = mask_input * embeddings_batch_size
    prompts_first_dimension_characteristics = set()
    if point_coordinates is not None:
        point_coordinates_characteristic = "-".join(
            [str(p.shape[0]) for p in point_coordinates]
        )
        prompts_first_dimension_characteristics.add(point_coordinates_characteristic)
    if point_labels is not None:
        point_labels_characteristic = "-".join([str(l.shape[0]) for l in point_labels])
        prompts_first_dimension_characteristics.add(point_labels_characteristic)
    if boxes is not None:
        boxes_characteristic = "-".join(
            [str(b.shape[0]) if len(b.shape) > 1 else "1" for b in boxes]
        )
        prompts_first_dimension_characteristics.add(boxes_characteristic)
    if len(prompts_first_dimension_characteristics) > 1:
        raise ModelInputError(
            message="When using SAM model, in scenario when combination of `point_coordinates` and `point_labels` and "
            "`boxes` provided, the model expect identical number of elements for each prompt component. "
            "If you run inference locally, verify your integration making sure that the model interface is "
            "used correctly. Running on Roboflow platform - contact us to get help.",
            help_url="https://todo",
        )
    if mask_input is not None and any(
        len(i.shape) != 3 or i.shape[0] != 1 for i in mask_input
    ):
        raise ModelInputError(
            message="When using SAM model with `mask_input`, each mask must be 3D tensor of shape (1, H, W). "
            "If you run inference locally, verify your integration making sure that the model interface is "
            "used correctly. Running on Roboflow platform - contact us to get help.",
            help_url="https://todo",
        )
    if boxes is not None:
        batched_boxes_provided = False
        for box in boxes:
            if len(box.shape) > 1 and box.shape[0] > 1:
                batched_boxes_provided = True
        if batched_boxes_provided and any(
            e is not None for e in [point_coordinates, point_labels, mask_input]
        ):
            raise ModelInputError(
                message="When using SAM, providing batched boxes (multiple RoIs for single image) makes it impossible "
                "to use other components of the prompt - like `point_coordinates`, `point_labels` "
                "or `mask_input` - and such situation was detected. "
                "If you run inference locally, verify your integration making sure that the model interface is "
                "used correctly. Running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
    return point_coordinates, point_labels, boxes, mask_input


def maybe_broadcast_list(value: Optional[List[T]], n: int) -> Optional[List[T]]:
    if value is None:
        return None
    return value * n


def pre_process_prompts(
    point_coordinates: Optional[List[ArrayOrTensor]],
    point_labels: Optional[List[ArrayOrTensor]],
    boxes: Optional[List[ArrayOrTensor]],
    mask_input: Optional[List[ArrayOrTensor]],
    device: torch.device,
    transform: ResizeLongestSide,
    original_image_sizes: List[Tuple[int, int]],
) -> Tuple[
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
]:
    if point_coordinates is not None:
        if point_labels is None:
            raise ModelInputError(
                message="When using SAM model, provided `point_coordinates` without `point_labels` which makes invalid "
                "input. If you run inference locally, verify your integration making sure that the model "
                "interface is used correctly. Running on Roboflow platform - contact us to get help.",
                help_url="https://todo",
            )
        point_coordinates = [
            (
                c.to(device)[None, :, :]
                if isinstance(c, torch.Tensor)
                else torch.from_numpy(c).to(device)[None, :, :]
            )
            for c in point_coordinates
        ]
        point_labels = [
            (
                l.to(device)[None, :]
                if isinstance(l, torch.Tensor)
                else torch.from_numpy(l).to(device)[None, :]
            )
            for l in point_labels
        ]
        point_coordinates = [
            transform.apply_coords_torch(point_coords, image_shape)
            for point_coords, image_shape in zip(
                point_coordinates, original_image_sizes
            )
        ]
    if boxes is not None:
        boxes = [
            (
                box.to(device)[None, :]
                if isinstance(box, torch.Tensor)
                else torch.from_numpy(box).to(device)[None, :]
            )
            for box in boxes
        ]
        boxes = [
            transform.apply_boxes_torch(box, image_shape)
            for box, image_shape in zip(boxes, original_image_sizes)
        ]
    if mask_input is not None:
        mask_input = [
            (
                mask.to(device)[None, :, :]
                if isinstance(mask, torch.Tensor)
                else torch.from_numpy(mask).to(device)[None, :, :]
            )
            for mask in mask_input
        ]
    return point_coordinates, point_labels, boxes, mask_input


def generate_model_inputs(
    embeddings: List[torch.Tensor],
    image_hashes: List[str],
    original_image_sizes: List[Tuple[int, int]],
    point_coordinates: Optional[List[torch.Tensor]],
    point_labels: Optional[List[torch.Tensor]],
    boxes: Optional[List[torch.Tensor]],
    mask_input: Optional[List[torch.Tensor]],
) -> Generator[
    Tuple[
        torch.Tensor,
        str,
        Tuple[int, int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
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


@torch.inference_mode()
def predict_for_single_image(
    model: Sam,
    transform: ResizeLongestSide,
    embeddings: torch.Tensor,
    original_image_size: Tuple[int, int],
    point_coordinates: Optional[torch.Tensor],
    point_labels: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor] = None,
    mask_input: Optional[torch.Tensor] = None,
    multi_mask_output: bool = True,
    return_logits: bool = False,
    mask_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = embeddings.unsqueeze(dim=0)
    if point_coordinates is not None:
        points = (point_coordinates, point_labels)
    else:
        points = None
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=mask_input,
    )
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multi_mask_output,
    )
    model_input_size = transform.get_preprocess_shape(
        original_image_size[0], original_image_size[1], transform.target_length
    )
    masks = model.postprocess_masks(
        low_res_masks, model_input_size, original_image_size
    )
    if not return_logits:
        threshold = mask_threshold or model.mask_threshold
        masks = masks > threshold
    if masks.shape[0] == 1:
        return masks[0], iou_predictions[0], low_res_masks[0]
    else:
        return masks, iou_predictions, low_res_masks
