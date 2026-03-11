"""RF-DETR few-shot adapter model.

Wraps an RF-DETR base model with LoRA-based few-shot fine-tuning.
Follows the same singleton + adapter pattern as OWLv2.
"""

import logging
import os
import threading
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from inference.core.entities.responses import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import API_KEY, DEVICE
from inference.core.models.base import Model
from inference.core.utils.image_utils import load_image_bgr
from inference_models.models.rfdetr.few_shot.cache import (
    DEFAULT_CACHE_DIR,
    DEFAULT_LRU_SIZE,
    FewShotAdapterCache,
)
from inference_models.models.rfdetr.few_shot.trainer import RFDETRFewShotTrainer
from inference_models.models.rfdetr.post_processor import PostProcess
from inference_models.models.rfdetr.rfdetr_base_pytorch import (
    LWDETR,
    build_model,
)
from inference_models.models.rfdetr.rfdetr_object_detection_pytorch import (
    CONFIG_FOR_MODEL_TYPE,
    RFDetrForObjectDetectionTorch,
)

logger = logging.getLogger(__name__)


class RFDETRBaseModelSingleton:
    """Singleton holder for pretrained RF-DETR base models (one per variant)."""

    _instances: Dict[str, RFDetrForObjectDetectionTorch] = {}
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        variant: str,
        api_key: Optional[str] = None,
    ) -> RFDetrForObjectDetectionTorch:
        key = variant
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    logger.info("Loading base RF-DETR model: %s", variant)
                    model = RFDetrForObjectDetectionTorch.from_pretrained(
                        model_name_or_path=variant,
                        model_type=variant,
                        labels="coco",
                        device=torch.device(DEVICE if DEVICE else "cpu"),
                    )
                    cls._instances[key] = model
        return cls._instances[key]


class RFDETRFewShot(Model):
    """Few-shot RF-DETR model using inline LoRA fine-tuning.

    Accepts training images with bounding box annotations as a "prompt",
    hashes them, checks for a cached LoRA adapter, trains one if needed,
    and returns object detection predictions.
    """

    def __init__(
        self,
        model_id: str = "rfdetr_few_shot",
        api_key: Optional[str] = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        lru_size: int = DEFAULT_LRU_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.api_key = api_key or API_KEY
        self.task_type = "object-detection"
        self._cache = FewShotAdapterCache(cache_dir=cache_dir, lru_size=lru_size)
        self._device = torch.device(DEVICE if DEVICE else "cpu")
        self._training_locks: Dict[str, threading.Lock] = {}
        self._training_locks_lock = threading.Lock()
        self._post_processor = PostProcess()

    def infer_from_request(self, request):
        from inference.core.entities.requests.rfdetr_few_shot import (
            RFDETRFewShotInferenceRequest,
        )

        return self.infer(
            image=request.image,
            training_data=request.training_data,
            model_hash=request.model_hash,
            confidence=request.confidence,
            iou_threshold=request.iou_threshold,
            model_variant=request.model_variant,
            lora_rank=request.lora_rank,
            learning_rate=request.learning_rate,
            num_epochs=request.num_epochs,
        )

    def _get_training_lock(self, key: str) -> threading.Lock:
        with self._training_locks_lock:
            if key not in self._training_locks:
                self._training_locks[key] = threading.Lock()
            return self._training_locks[key]

    def infer(
        self,
        image: Any,
        training_data: Optional[list] = None,
        model_hash: Optional[str] = None,
        confidence: float = 0.5,
        iou_threshold: float = 0.5,
        model_variant: str = "rfdetr-base",
        lora_rank: int = 8,
        learning_rate: float = 1e-4,
        num_epochs: int = 1,
        **kwargs,
    ):
        if training_data is None and model_hash is None:
            raise ValueError("Either training_data or model_hash must be provided")

        # Resolve variant config
        if model_variant not in CONFIG_FOR_MODEL_TYPE:
            raise ValueError(
                f"Unknown model_variant '{model_variant}'. "
                f"Supported: {list(CONFIG_FOR_MODEL_TYPE.keys())}"
            )

        # Compute or use provided hash
        if model_hash is not None:
            computed_hash = model_hash
        else:
            # Convert training_data from Pydantic models to plain dicts for hashing/training
            training_dicts = self._training_data_to_dicts(training_data)
            computed_hash = FewShotAdapterCache.compute_hash(training_dicts, model_variant)

        # Try memory cache first
        cached = self._cache.get_from_memory(computed_hash)
        if cached is not None:
            logger.info("Cache hit (memory): %s", computed_hash)
            model, class_names = cached
        else:
            # Try disk cache
            base_model_wrapper = RFDETRBaseModelSingleton.get(
                model_variant, api_key=self.api_key
            )
            config_cls = CONFIG_FOR_MODEL_TYPE[model_variant]
            config = config_cls(device=self._device)

            adapter_state = self._cache.load_from_disk(model_variant, computed_hash)
            if adapter_state is not None:
                logger.info("Cache hit (disk): %s", computed_hash)
                model, class_names = self._cache.rebuild_model_from_adapter_state(
                    base_model=base_model_wrapper._model,
                    config=config,
                    adapter_state=adapter_state,
                    device=self._device,
                )
                self._cache.put_in_memory(computed_hash, model, class_names)
            else:
                # Need to train — requires training_data
                if training_data is None:
                    raise ValueError(
                        f"No cached adapter found for model_hash '{computed_hash}'. "
                        "Please provide training_data."
                    )

                training_dicts = self._training_data_to_dicts(training_data)

                # Use per-hash lock to prevent duplicate training
                lock = self._get_training_lock(computed_hash)
                with lock:
                    # Double-check cache after acquiring lock
                    cached = self._cache.get_from_memory(computed_hash)
                    if cached is not None:
                        model, class_names = cached
                    else:
                        logger.info("Cache miss, training LoRA: %s", computed_hash)
                        class_names = self._extract_class_names(training_dicts)
                        trainer = RFDETRFewShotTrainer(
                            device=self._device,
                            lora_rank=lora_rank,
                            learning_rate=learning_rate,
                            num_epochs=num_epochs,
                        )
                        model, adapter_state = trainer.train(
                            base_model=base_model_wrapper._model,
                            config=config,
                            training_data=training_dicts,
                            class_names=class_names,
                        )
                        self._cache.save_to_disk(model_variant, computed_hash, adapter_state)
                        self._cache.put_in_memory(computed_hash, model, class_names)

        # Run inference
        if isinstance(image, list):
            images_decoded = [load_image_bgr(i) for i in image]
        else:
            images_decoded = [load_image_bgr(image)]

        image_sizes: List[Tuple[int, int]] = [
            img.shape[:2][::-1] for img in images_decoded  # (w, h)
        ]

        predictions = self._run_inference(
            model=model,
            images=images_decoded,
            class_names=class_names,
            model_variant=model_variant,
            confidence=confidence,
        )

        return self._make_response(
            predictions=predictions,
            image_sizes=image_sizes,
            class_names=class_names,
            model_hash=computed_hash,
        )

    def _run_inference(
        self,
        model: LWDETR,
        images: List[np.ndarray],
        class_names: List[str],
        model_variant: str,
        confidence: float,
    ) -> List[dict]:
        config_cls = CONFIG_FOR_MODEL_TYPE[model_variant]
        config = config_cls(device=self._device)
        resolution = config.resolution

        # Pre-process images
        processed_images = []
        for img in images:
            img_rgb = img[:, :, ::-1].copy()
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            processed_images.append(tensor)

        batch = torch.stack(processed_images).to(self._device)

        with torch.inference_mode():
            outputs = model(batch)

        # Post-process per image
        orig_sizes = [(img.shape[0], img.shape[1]) for img in images]
        target_sizes = torch.tensor(orig_sizes, device=self._device)
        results = self._post_processor(outputs, target_sizes=target_sizes)

        # Filter by confidence
        filtered = []
        for result in results:
            keep = result["scores"] > confidence
            filtered.append({
                "scores": result["scores"][keep],
                "labels": result["labels"][keep],
                "boxes": result["boxes"][keep],
            })
        return filtered

    def _make_response(
        self,
        predictions: List[dict],
        image_sizes: List[Tuple[int, int]],
        class_names: List[str],
        model_hash: str,
    ):
        from inference.core.entities.responses.rfdetr_few_shot import (
            RFDETRFewShotInferenceResponse,
        )

        responses = []
        for pred, (img_w, img_h) in zip(predictions, image_sizes):
            instances = []
            for i in range(pred["scores"].shape[0]):
                x_min, y_min, x_max, y_max = pred["boxes"][i].tolist()
                width = x_max - x_min
                height = y_max - y_min
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                class_id = pred["labels"][i].item()
                score = pred["scores"][i].item()
                cls_name = (
                    class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                )
                instances.append(
                    ObjectDetectionPrediction(
                        **{
                            "x": center_x,
                            "y": center_y,
                            "width": width,
                            "height": height,
                            "confidence": score,
                            "class": cls_name,
                            "class_id": class_id,
                        }
                    )
                )
            responses.append(
                RFDETRFewShotInferenceResponse(
                    predictions=instances,
                    image=InferenceResponseImage(width=img_w, height=img_h),
                    model_hash=model_hash,
                )
            )
        return responses

    @staticmethod
    def _training_data_to_dicts(training_data: list) -> list:
        """Convert Pydantic TrainingImage models to plain dicts for the trainer."""
        result = []
        for item in training_data:
            if hasattr(item, "model_dump"):
                d = item.model_dump()
                # The image field is an InferenceRequestImage; extract its value
                img = d.get("image", {})
                if isinstance(img, dict):
                    img_val = img.get("value", img)
                else:
                    img_val = img
                boxes = []
                for box in d.get("boxes", []):
                    boxes.append({
                        "x": box["x"],
                        "y": box["y"],
                        "w": box["w"],
                        "h": box["h"],
                        "cls": box["cls"],
                    })
                result.append({"image": img_val, "boxes": boxes})
            else:
                result.append(item)
        return result

    @staticmethod
    def _extract_class_names(training_dicts: list) -> List[str]:
        """Extract sorted unique class names from training data."""
        names = set()
        for item in training_dicts:
            for box in item["boxes"]:
                names.add(box["cls"])
        return sorted(names)

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        pass
