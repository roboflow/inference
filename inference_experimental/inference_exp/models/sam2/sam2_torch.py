import hashlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import ColorFormat
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.sam2.cache import (
    Sam2ImageEmbeddingsCache,
    Sam2ImageEmbeddingsCacheNullObject,
    Sam2LowResolutionMasksCache,
    Sam2LowResolutionMasksCacheNullObject,
)
from inference_exp.models.sam2.entities import SAM2ImageEmbeddings
from inference_exp.utils.file_system import read_json
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

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
                help_url="https://todo",
            ) from error
        if version not in SUPPORTED_VERSIONS:
            raise CorruptedModelPackageError(
                message=f"Detected unsupported version of SAM2 model: {version}. Supported versions: "
                f"are {SUPPORTED_VERSIONS}. If you run inference locally, verify the correctness of "
                f"SAM2 model package. If you see the error running on Roboflow platform - "
                "contact us to get help.",
                help_url="https://todo",
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
        )

    def __init__(
        self,
        model: SAM2Base,
        transform: SAM2Transforms,
        device: torch.device,
        max_batch_size: int,
        sam2_image_embeddings_cache: Sam2ImageEmbeddingsCache,
        sam2_low_resolution_masks_cache: Sam2LowResolutionMasksCache,
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

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        use_embeddings_cache: bool = True,
        **kwargs,
    ) -> List[SAM2ImageEmbeddings]:
        model_input_images, image_hashes, original_image_sizes = (
            self.pre_process_images(
                images=images,
                input_color_format=input_color_format,
                **kwargs,
            )
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
                model_input_images = self._transform.transforms(images / 255.0)
            else:
                image_hashes = [compute_image_hash(image=images)]
                if input_color_format == "bgr":
                    images = images[::-1, :, :].contiguous()
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
                        if input_color_format in {None, "bgr"}:
                            image = np.ascontiguousarray(image[:, :, ::-1])
                        input_image = self._transform(image).to(self._device)
                        model_input_images.append(input_image)
                    else:
                        original_image_sizes.append(tuple(image.shape[1:3]))
                        image = image.to(self._device)
                        if input_color_format == "bgr":
                            image = image[::-1, :, :].contiguous()
                        input_image = self._transform.transforms(image / 255)
                        model_input_images.append(input_image)
                model_input_images = torch.stack(model_input_images, dim=0)
            else:
                image_hashes = [compute_image_hash(image=images)]
                original_image_sizes = [images.shape[:2]]
                if input_color_format in {None, "bgr"}:
                    images = np.ascontiguousarray(images[:, :, ::-1])
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
            backbone_out = self._model.forward_image(input_images_batch)
            _, vision_feats, _, _ = self._model._prepare_backbone_features(backbone_out)
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
