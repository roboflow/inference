"""Disk + in-memory LRU cache for few-shot LoRA adapters."""

import hashlib
import logging
import os
import pickle
from collections import OrderedDict
from threading import Lock
from typing import Dict, List, Optional, Tuple

import torch

from inference_models.models.rfdetr.rfdetr_base_pytorch import (
    LWDETR,
    ModelConfig,
    build_model,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "inference", "rfdetr_few_shot"
)
DEFAULT_LRU_SIZE = 3


class FewShotAdapterCache:
    """Two-level cache for fine-tuned RF-DETR LoRA adapters.

    Level 1: In-memory LRU cache of merged LWDETR models.
    Level 2: Disk cache of adapter_state dicts (LoRA + head weights).
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        lru_size: int = DEFAULT_LRU_SIZE,
    ):
        self.cache_dir = cache_dir
        self.lru_size = lru_size
        self._memory_cache: OrderedDict[str, Tuple[LWDETR, List[str]]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def compute_hash(training_data: list, variant: str) -> str:
        """Compute a deterministic hash of training data + variant.

        Uses SHA1 of pickled (variant, [(image_value, boxes)]) tuples.
        """
        hashable = []
        for item in training_data:
            image_val = item["image"]
            if isinstance(image_val, dict):
                image_val = image_val.get("value", str(image_val))
            boxes_key = []
            for box in item["boxes"]:
                boxes_key.append((box["x"], box["y"], box["w"], box["h"], box["cls"]))
            if isinstance(image_val, bytes):
                img_hash = hashlib.sha1(image_val).hexdigest()
            elif isinstance(image_val, str):
                img_hash = hashlib.sha1(image_val.encode("utf-8", errors="replace")).hexdigest()
            else:
                img_hash = hashlib.sha1(pickle.dumps(image_val)).hexdigest()
            hashable.append((img_hash, tuple(boxes_key)))
        payload = pickle.dumps((variant, tuple(hashable)))
        return hashlib.sha1(payload).hexdigest()

    def get_from_memory(
        self, model_hash: str
    ) -> Optional[Tuple[LWDETR, List[str]]]:
        with self._lock:
            if model_hash in self._memory_cache:
                self._memory_cache.move_to_end(model_hash)
                return self._memory_cache[model_hash]
        return None

    def put_in_memory(
        self, model_hash: str, model: LWDETR, class_names: List[str]
    ) -> None:
        with self._lock:
            if model_hash in self._memory_cache:
                self._memory_cache.move_to_end(model_hash)
                return
            if len(self._memory_cache) >= self.lru_size:
                evicted_key, (evicted_model, _) = self._memory_cache.popitem(last=False)
                logger.info("Evicting cached model %s from memory", evicted_key)
                del evicted_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self._memory_cache[model_hash] = (model, class_names)

    def _disk_path(self, variant: str, model_hash: str) -> str:
        return os.path.join(self.cache_dir, variant, model_hash)

    def save_to_disk(
        self, variant: str, model_hash: str, adapter_state: Dict
    ) -> None:
        path = self._disk_path(variant, model_hash)
        os.makedirs(path, exist_ok=True)
        torch.save(adapter_state, os.path.join(path, "adapter_state.pt"))
        logger.info("Saved adapter to disk: %s", path)

    def load_from_disk(
        self, variant: str, model_hash: str
    ) -> Optional[Dict]:
        path = self._disk_path(variant, model_hash)
        state_path = os.path.join(path, "adapter_state.pt")
        if not os.path.exists(state_path):
            return None
        logger.info("Loading adapter from disk: %s", path)
        return torch.load(state_path, map_location="cpu", weights_only=False)

    def rebuild_model_from_adapter_state(
        self,
        base_model: LWDETR,
        config: ModelConfig,
        adapter_state: Dict,
        device: torch.device,
    ) -> Tuple[LWDETR, List[str]]:
        """Rebuild a merged model from cached adapter state.

        Loads base weights, applies LoRA, loads adapter + head weights,
        merges, and returns eval-mode model.
        """
        from peft import LoraConfig, PeftModel, get_peft_model

        num_classes = adapter_state["num_classes"]
        class_names = adapter_state["class_names"]

        fresh_config = config.model_copy(update={"num_classes": num_classes})
        model = build_model(config=fresh_config)

        base_state = base_model.state_dict()
        model_state = model.state_dict()
        filtered_state = {
            k: v
            for k, v in base_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(filtered_state, strict=False)
        model.reinitialize_detection_head(num_classes + 1)
        model = model.to(device)

        # Apply LoRA structure
        from inference_models.models.rfdetr.few_shot.trainer import (
            BACKBONE_LORA_TARGETS,
            DECODER_LORA_TARGETS,
        )

        all_targets = BACKBONE_LORA_TARGETS + DECODER_LORA_TARGETS
        lora_config = LoraConfig(
            r=adapter_state["lora_rank"],
            lora_alpha=adapter_state["lora_alpha"],
            lora_dropout=0.0,
            target_modules=all_targets,
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)

        # Load LoRA weights
        current_state = peft_model.state_dict()
        lora_state = adapter_state["lora_state_dict"]
        for k, v in lora_state.items():
            if k in current_state:
                current_state[k] = v.to(device)
        peft_model.load_state_dict(current_state)

        # Load head weights
        head_state = adapter_state["head_state_dict"]
        peft_model.base_model.model.class_embed.load_state_dict(
            {k: v.to(device) for k, v in head_state["class_embed"].items()}
        )
        peft_model.base_model.model.bbox_embed.load_state_dict(
            {k: v.to(device) for k, v in head_state["bbox_embed"].items()}
        )
        if "enc_out_class_embed" in head_state:
            peft_model.base_model.model.transformer.enc_out_class_embed.load_state_dict(
                {k: v.to(device) for k, v in head_state["enc_out_class_embed"].items()}
            )

        merged_model = peft_model.merge_and_unload()
        merged_model = merged_model.eval()

        return merged_model, class_names
