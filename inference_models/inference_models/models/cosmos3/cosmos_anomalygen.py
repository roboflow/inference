from threading import Lock
from typing import List, Optional, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import ModelInputError
from inference_models.models.cosmos3.runtime_loading import load_runtime_from_package

RUNTIME_MODULE_FILE = "cosmos_anomalygen_runtime.py"
RUNTIME_CLASS_NAME = "CosmosAnomalyGenRuntime"

DEFAULT_GUIDANCE = 7.0
DEFAULT_NUM_STEPS = 35


class CosmosAnomalyGen:
    """NVIDIA Cosmos AnomalyGen - mask-conditioned defect inpainting.

    Given a clean image, a binary placement mask, and an anomaly type, the
    model inpaints a realistic defect into the mask's region. Fine-tuned
    per-defect checkpoints load through the same weight-package path as the
    base model. Parameters mirror the upstream SDG generation-entry contract
    (guidance / num_steps / crop-and-paste), so callers currently driving
    NVIDIA's `synthetic_dataset_generation` script via JSONL can map each
    entry onto one `generate()` call.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "CosmosAnomalyGen":
        runtime_class = load_runtime_from_package(
            model_name_or_path=model_name_or_path,
            runtime_module_file=RUNTIME_MODULE_FILE,
            runtime_class_name=RUNTIME_CLASS_NAME,
        )
        runtime = runtime_class.load(model_name_or_path, device=device)
        return cls(runtime=runtime, device=device)

    def __init__(self, runtime, device: torch.device):
        self._runtime = runtime
        self._device = device
        self._lock = Lock()

    def generate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        anomaly_type: str,
        guidance: float = DEFAULT_GUIDANCE,
        num_steps: int = DEFAULT_NUM_STEPS,
        seed: int = 0,
        num_images: int = 1,
        crop_and_paste: bool = True,
        crop_ratio: float = 2.0,
        poisson_blend: bool = False,
        input_color_format: ColorFormat = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """Inpaint `anomaly_type` into the white region of `mask` on `image`.

        Returns `num_images` generated BGR images at the input resolution.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ModelInputError(
                message="generate() expects a single HxWx3 numpy image.",
                help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
            )
        binary_mask = _normalize_mask(mask=mask, image=image)
        if input_color_format != "rgb":
            image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)
        with self._lock:
            generated = self._runtime.generate(
                image=image,
                mask=binary_mask,
                anomaly_type=anomaly_type,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
                num_images=num_images,
                crop_and_paste=crop_and_paste,
                crop_ratio=crop_ratio,
                poisson_blend=poisson_blend,
            )
        return [
            np.ascontiguousarray(np.asarray(result)[:, :, ::-1]) for result in generated
        ]


def _normalize_mask(mask: Union[np.ndarray, None], image: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim not in (2, 3):
        raise ModelInputError(
            message="generate() expects a HxW binary placement mask.",
            help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
        )
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != image.shape[:2]:
        raise ModelInputError(
            message=f"Mask resolution {mask.shape[:2]} does not match image "
            f"resolution {image.shape[:2]}.",
            help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
        )
    binary_mask = (mask.astype(np.uint8) >= 128) if mask.dtype == np.uint8 else (
        mask.astype(bool)
    )
    if not binary_mask.any():
        raise ModelInputError(
            message="Placement mask is empty - draw or place at least one "
            "defect region before generating.",
            help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
        )
    return binary_mask.astype(np.uint8) * 255
