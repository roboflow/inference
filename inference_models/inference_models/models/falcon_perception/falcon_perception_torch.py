# Copyright 2026 Technology Innovation Institute (TII), Abu Dhabi.
# Licensed under the Apache License, Version 2.0.
# Adapted from https://github.com/tiiuae/Falcon-Perception for integration
# with the inference-models package.
#
# Falcon Perception integration with inference-models framework.
# Implements OpenVocabularyObjectDetectionModel for detection mode
# and provides segmentation mode via the `task` parameter.

import os
from dataclasses import dataclass
from threading import RLock
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import (
    Detections,
    OpenVocabularyObjectDetectionModel,
)
from inference_models.models.falcon_perception.config import (
    DEFAULT_CONFIG,
    FalconPerceptionConfig,
)
from inference_models.models.falcon_perception.engine import BatchEngine
from inference_models.models.falcon_perception.model import FalconPerceptionModel
from inference_models.models.falcon_perception.postprocessing import (
    result_to_detections,
    result_to_instance_detections,
)
from inference_models.models.falcon_perception.preprocessing import (
    ImageMetadata,
    get_special_token_ids,
    load_tokenizer,
    preprocess_image,
    tokenize_prompts,
)


@dataclass
class FalconPerceptionPreProcessingMeta:
    """Preprocessing metadata for a batch of images."""

    image_metadatas: List[ImageMetadata]
    original_images_rgb: List[np.ndarray]


class FalconPerceptionTorch(
    OpenVocabularyObjectDetectionModel[
        torch.Tensor,
        FalconPerceptionPreProcessingMeta,
        Dict,
    ]
):
    """Falcon Perception model for open-vocabulary object detection
    and instance segmentation.

    Supports two inference modes:
    - Detection only (task="detection"): Returns bounding boxes and confidence.
    - Segmentation (task="segmentation"): Returns bounding boxes, confidence, and masks.

    The model uses a unified dense Transformer that processes image patches
    and text tokens jointly, then autoregressively generates structured
    predictions (coordinates, sizes, and optionally segmentation masks).
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "FalconPerceptionTorch":
        """Load Falcon Perception from a local directory.

        The directory should contain:
        - model.safetensors (or model-00001-of-*.safetensors for sharded weights)
        - tokenizer.json (BPE tokenizer configuration)
        - config.json (optional, model configuration overrides)

        Args:
            model_name_or_path: Path to directory with model files.
            device: Target device (cpu, cuda, etc.).

        Returns:
            Initialized FalconPerceptionTorch instance.
        """
        model_dir = model_name_or_path
        config = _load_config(model_dir)
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        # Load tokenizer and update config with special token IDs
        if os.path.exists(tokenizer_path):
            special_ids = get_special_token_ids(tokenizer_path)
            # Create new config with special token IDs from tokenizer
            config_dict = {
                k: v
                for k, v in config.__dict__.items()
                if k not in special_ids
            }
            config_dict.update(special_ids)
            config = FalconPerceptionConfig(**config_dict)
            tokenizer = load_tokenizer(tokenizer_path, config)
        else:
            tokenizer = None

        # Build model
        model = FalconPerceptionModel(config)

        # Load weights from safetensors
        _load_weights(model, model_dir, device)

        model = model.to(device)
        model.eval()

        return cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            device=device,
        )

    def __init__(
        self,
        model: FalconPerceptionModel,
        config: FalconPerceptionConfig,
        tokenizer,
        device: torch.device,
    ):
        self._model = model
        self._config = config
        self._tokenizer = tokenizer
        self._device = device
        self._engine = BatchEngine(model=model, config=config, device=device)
        self._lock = RLock()

    def infer(
        self,
        images: Union[
            torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]
        ],
        classes: Union[str, List[str]],
        task: str = "segmentation",
        **kwargs,
    ) -> List[Union[Detections, InstanceDetections]]:
        """Run inference on images with text prompts.

        Args:
            images: Input image(s). Can be:
                - np.ndarray: Single BGR image (H, W, 3)
                - torch.Tensor: Single RGB image (3, H, W) or batch (B, 3, H, W)
                - List of above
            classes: Text prompt(s). Can be a single string or list of strings.
                Also accepted as 'prompts' kwarg for convenience.
            task: "detection" for boxes only, "segmentation" for boxes + masks.
            **kwargs: Additional arguments (e.g., input_color_format).

        Returns:
            List of Detections (detection mode) or InstanceDetections (segmentation mode),
            one per input image.
        """
        if isinstance(classes, str):
            classes = [classes]

        # Support 'prompts' as alias for 'classes'
        prompts = kwargs.pop("prompts", None)
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            classes = prompts

        pre_processed_images, pre_processing_meta = self.pre_process(images, **kwargs)
        model_results = self.forward(
            pre_processed_images, classes, task=task, **kwargs
        )
        return self.post_process(
            model_results, pre_processing_meta, task=task, **kwargs
        )

    def pre_process(
        self,
        images: Union[
            torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]
        ],
        input_color_format: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, FalconPerceptionPreProcessingMeta]:
        """Preprocess images for Falcon Perception.

        Converts to RGB, resizes preserving aspect ratio, pads to patch multiples,
        normalizes to ImageNet stats.

        Returns:
            (batched_pixel_values, preprocessing_metadata)
        """
        # Normalize input to list of RGB numpy arrays
        images_rgb = _to_rgb_numpy_list(images, input_color_format)

        pixel_values_list = []
        metadata_list = []
        for img in images_rgb:
            pv, meta = preprocess_image(img, self._config)
            pixel_values_list.append(pv)
            metadata_list.append(meta)

        # Pad to same size within batch (needed for batched processing)
        max_h = max(pv.shape[1] for pv in pixel_values_list)
        max_w = max(pv.shape[2] for pv in pixel_values_list)
        padded = []
        for pv in pixel_values_list:
            pad_h = max_h - pv.shape[1]
            pad_w = max_w - pv.shape[2]
            if pad_h > 0 or pad_w > 0:
                pv = torch.nn.functional.pad(pv, (0, pad_w, 0, pad_h))
            padded.append(pv)

        batched = torch.stack(padded, dim=0).to(self._device)
        return batched, FalconPerceptionPreProcessingMeta(
            image_metadatas=metadata_list,
            original_images_rgb=images_rgb,
        )

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        classes: List[str],
        task: str = "segmentation",
        **kwargs,
    ) -> Dict:
        """Run model forward pass.

        Tokenizes prompts, runs batch engine for each image independently,
        collects results.

        Returns:
            Dict with 'results' list and metadata for post-processing.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Make sure tokenizer.json exists in the model directory."
            )

        # Tokenize prompts
        text_token_ids = tokenize_prompts(classes, self._tokenizer, self._config)

        all_results = []
        with self._lock:
            for i in range(pre_processed_images.shape[0]):
                result = self._engine.run(
                    pixel_values=pre_processed_images[i],
                    text_token_ids=text_token_ids,
                    image_metadata=None,  # Will use metadata from pre_processing_meta
                    prompts=classes,
                    task=task,
                )
                all_results.append(result)

        return {"results": all_results, "classes": classes, "task": task}

    def post_process(
        self,
        model_results: Dict,
        pre_processing_meta: FalconPerceptionPreProcessingMeta,
        task: str = "segmentation",
        **kwargs,
    ) -> List[Union[Detections, InstanceDetections]]:
        """Convert raw model results to Detections or InstanceDetections.

        Rescales coordinates from model space to original image dimensions.
        """
        results = model_results["results"]
        classes = model_results["classes"]
        task = model_results.get("task", task)
        outputs = []

        for result, img_meta in zip(
            results, pre_processing_meta.image_metadatas
        ):
            if task == "segmentation":
                output = result_to_instance_detections(
                    result=result,
                    image_metadata=img_meta,
                    model=self._model,
                    config=self._config,
                    prompts=classes,
                )
            else:
                output = result_to_detections(
                    result=result,
                    image_metadata=img_meta,
                    config=self._config,
                    prompts=classes,
                )
            outputs.append(output)

        return outputs

    def __call__(
        self,
        images: Union[
            torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]
        ],
        classes: Union[str, List[str]] = None,
        prompts: Union[str, List[str]] = None,
        task: str = "segmentation",
        **kwargs,
    ) -> List[Union[Detections, InstanceDetections]]:
        """Convenience method: supports both 'classes' and 'prompts' parameter names."""
        if classes is None and prompts is not None:
            classes = prompts
        if classes is None:
            raise ValueError("Must provide either 'classes' or 'prompts' argument.")
        return self.infer(images=images, classes=classes, task=task, **kwargs)


def _to_rgb_numpy_list(
    images: Union[
        torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]
    ],
    input_color_format: Optional[str] = None,
) -> List[np.ndarray]:
    """Normalize input images to a list of RGB uint8 numpy arrays."""
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            images = [images]
        elif images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]
    elif isinstance(images, torch.Tensor):
        if images.ndim == 3:
            # (3, H, W) -> (H, W, 3)
            img = images.cpu().numpy()
            if img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            return [img]
        elif images.ndim == 4:
            result = []
            for i in range(images.shape[0]):
                img = images[i].cpu().numpy()
                if img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                result.append(img)
            return result
    elif isinstance(images, list):
        result = []
        for img in images:
            result.extend(_to_rgb_numpy_list(img, input_color_format))
        return result

    # At this point images is a list of numpy arrays
    result = []
    for img in images:
        # Default: numpy arrays from OpenCV are BGR
        fmt = input_color_format or "bgr"
        if fmt == "bgr":
            img = np.ascontiguousarray(img[:, :, ::-1])
        result.append(img)
    return result


def _load_config(model_dir: str) -> FalconPerceptionConfig:
    """Load model configuration, falling back to defaults."""
    import json

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG

    with open(config_path, "r") as f:
        raw = json.load(f)

    # Map config.json keys to our dataclass fields
    field_mapping = {
        "hidden_size": "hidden_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "intermediate_size": "ffn_hidden_dim",
        "vocab_size": "vocab_size",
        "patch_size": "patch_size",
        "max_image_size": "max_image_size",
        "coord_bins": "coord_bins",
        "size_bins": "size_bins",
        "seg_dim": "seg_dim",
        "anyup_levels": "anyup_levels",
        "anyup_hidden_dim": "anyup_hidden_dim",
        "log2_size_range": "log2_size_range",
        "mask_threshold": "mask_threshold",
        "max_instances_per_query": "max_instances_per_query",
    }

    config_kwargs = {}
    for json_key, field_name in field_mapping.items():
        if json_key in raw:
            config_kwargs[field_name] = raw[json_key]
        elif field_name in raw:
            config_kwargs[field_name] = raw[field_name]

    return FalconPerceptionConfig(**config_kwargs)


def _load_weights(
    model: FalconPerceptionModel,
    model_dir: str,
    device: torch.device,
) -> None:
    """Load model weights from safetensors files.

    Supports both single-file and sharded weight formats.
    """
    from safetensors.torch import load_file

    # Find safetensors files
    weight_files = sorted(
        f
        for f in os.listdir(model_dir)
        if f.endswith(".safetensors") and not f.startswith(".")
    )

    if not weight_files:
        # Try .pth fallback
        pth_path = os.path.join(model_dir, "model.pth")
        if os.path.exists(pth_path):
            state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            return
        raise FileNotFoundError(
            f"No .safetensors or .pth weight files found in {model_dir}. "
            f"Expected model.safetensors or sharded model-00001-of-*.safetensors files."
        )

    # Load all safetensors shards
    full_state_dict = {}
    for wf in weight_files:
        shard = load_file(os.path.join(model_dir, wf), device="cpu")
        full_state_dict.update(shard)

    # Load into model with non-strict matching (allows missing/extra keys)
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
    if missing:
        import logging

        logger = logging.getLogger("inference_models.falcon_perception")
        logger.warning(
            f"Missing keys when loading Falcon Perception weights: {len(missing)} keys. "
            f"First few: {missing[:5]}"
        )
