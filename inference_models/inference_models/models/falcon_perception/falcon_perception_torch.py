from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MASK_THRESHOLD,
    INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MAX_IMAGE_SIZE,
    INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MAX_NEW_TOKENS,
    INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MIN_IMAGE_SIZE,
)
from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.errors import ModelInputError
from inference_models.models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)
from inference_models.models.base.object_detection import (
    Detections,
    OpenVocabularyObjectDetectionModel,
)


@dataclass
class _PreProcessedInputs:
    """Container for batch-processed inputs ready for the model."""

    batch_inputs: Dict[str, Any]
    original_images: List[Any]  # PIL images for mask finalization


@dataclass
class _PreProcessingMetadata:
    """Metadata from pre-processing needed for post-processing."""

    image_dimensions: List[ImageDimensions]
    prompts: List[str]
    task: str


@dataclass
class PromptResult:
    """Result for a single text prompt."""

    prompt: str
    present: bool
    xyxy: torch.Tensor  # (n_instances, 4)
    confidence: torch.Tensor  # (n_instances,)
    mask: Optional[torch.Tensor] = None  # (n_instances, H, W) if segmentation


class FalconPerceptionForObjectDetectionTorch(
    OpenVocabularyObjectDetectionModel[
        _PreProcessedInputs,
        _PreProcessingMetadata,
        List[List[Any]],
    ]
):
    """Falcon Perception model for open-vocabulary object detection (Torch backend).

    This wraps the ``falcon-perception`` package's batch inference engine
    to provide bounding-box detection from natural language text prompts.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        dtype: str = "float32",
        compile: bool = False,
        **kwargs,
    ) -> "FalconPerceptionForObjectDetectionTorch":
        from falcon_perception import load_and_prepare_model, setup_torch_config

        setup_torch_config()
        model, tokenizer, model_args = load_and_prepare_model(
            hf_local_dir=model_name_or_path,
            device=str(device),
            dtype=dtype,
            compile=compile,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
            device=device,
        )

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_args: Any,
        device: torch.device,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._model_args = model_args
        self._device = device
        self._lock = Lock()

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[_PreProcessedInputs, _PreProcessingMetadata]:
        from PIL import Image

        pil_images = _to_pil_images(images, input_color_format)
        image_dimensions = [
            ImageDimensions(height=img.height, width=img.width) for img in pil_images
        ]
        prompts = kwargs.get("classes", [])
        if isinstance(prompts, str):
            prompts = [prompts]
        task = kwargs.get("task", "detection")
        return (
            _PreProcessedInputs(batch_inputs={}, original_images=pil_images),
            _PreProcessingMetadata(
                image_dimensions=image_dimensions,
                prompts=prompts,
                task=task,
            ),
        )

    def forward(
        self,
        pre_processed_images: _PreProcessedInputs,
        classes: List[str],
        task: str = "detection",
        max_new_tokens: int = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MAX_NEW_TOKENS,
        min_image_size: int = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MIN_IMAGE_SIZE,
        max_image_size: int = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MAX_IMAGE_SIZE,
        mask_threshold: float = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MASK_THRESHOLD,
        **kwargs,
    ) -> List[List[Any]]:
        from falcon_perception import build_prompt_for_task
        from falcon_perception.batch_inference import (
            BatchInferenceEngine,
            process_batch_and_generate,
        )

        pil_images = pre_processed_images.original_images
        if isinstance(classes, str):
            classes = [classes]
        all_results: List[List[Any]] = []

        engine = BatchInferenceEngine(self._model, self._tokenizer)
        stop_token_ids = [
            self._tokenizer.eos_token_id,
            self._tokenizer.end_of_query_token_id,
        ]

        with self._lock, torch.inference_mode():
            for pil_image in pil_images:
                per_image_results = []
                for prompt_text in classes:
                    prompt = build_prompt_for_task(prompt_text, task)
                    batch_inputs = process_batch_and_generate(
                        self._tokenizer,
                        [(pil_image, prompt)],
                        max_length=4096,
                        min_dimension=min_image_size,
                        max_dimension=max_image_size,
                    )
                    batch_inputs = {
                        k: (v.to(self._device) if torch.is_tensor(v) else v)
                        for k, v in batch_inputs.items()
                    }
                    _, aux_out = engine.generate(
                        **batch_inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        stop_token_ids=stop_token_ids,
                        seed=42,
                        task=task,
                    )
                    per_image_results.append(
                        (prompt_text, aux_out[0] if aux_out else None)
                    )
                all_results.append(per_image_results)
        return all_results

    def post_process(
        self,
        model_results: List[List[Any]],
        pre_processing_meta: _PreProcessingMetadata,
        **kwargs,
    ) -> List[Detections]:
        results = []
        for image_idx, per_image_results in enumerate(model_results):
            dims = pre_processing_meta.image_dimensions[image_idx]
            all_xyxy = []
            all_confidence = []
            all_class_ids = []

            for prompt_idx, (prompt_text, aux) in enumerate(per_image_results):
                if aux is None:
                    continue
                boxes = _extract_boxes_from_aux(aux, dims.width, dims.height)
                for box in boxes:
                    all_xyxy.append(box["xyxy"])
                    all_confidence.append(box["confidence"])
                    all_class_ids.append(prompt_idx)

            if all_xyxy:
                results.append(
                    Detections(
                        xyxy=torch.tensor(all_xyxy, dtype=torch.float32),
                        confidence=torch.tensor(all_confidence, dtype=torch.float32),
                        class_id=torch.tensor(all_class_ids, dtype=torch.int64),
                    )
                )
            else:
                results.append(
                    Detections(
                        xyxy=torch.zeros((0, 4), dtype=torch.float32),
                        confidence=torch.zeros((0,), dtype=torch.float32),
                        class_id=torch.zeros((0,), dtype=torch.int64),
                    )
                )
        return results


class FalconPerceptionForInstanceSegmentationTorch(
    InstanceSegmentationModel[
        _PreProcessedInputs,
        _PreProcessingMetadata,
        List[List[Any]],
    ]
):
    """Falcon Perception model for open-vocabulary instance segmentation (Torch backend).

    This wraps the ``falcon-perception`` package's batch inference engine
    to provide bounding-box detection + binary masks from natural language
    text prompts.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        dtype: str = "float32",
        compile: bool = False,
        **kwargs,
    ) -> "FalconPerceptionForInstanceSegmentationTorch":
        from falcon_perception import load_and_prepare_model, setup_torch_config

        setup_torch_config()
        model, tokenizer, model_args = load_and_prepare_model(
            hf_local_dir=model_name_or_path,
            device=str(device),
            dtype=dtype,
            compile=compile,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_args=model_args,
            device=device,
        )

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_args: Any,
        device: torch.device,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._model_args = model_args
        self._device = device
        self._lock = Lock()

    @property
    def class_names(self) -> List[str]:
        return []

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompts: Optional[List[str]] = None,
        task: str = "segmentation",
        **kwargs,
    ) -> List[InstanceDetections]:
        kwargs["classes"] = prompts or []
        kwargs["task"] = task
        pre_processed, meta = self.pre_process(images, **kwargs)
        raw = self.forward(pre_processed, **kwargs)
        return self.post_process(raw, meta, **kwargs)

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[_PreProcessedInputs, _PreProcessingMetadata]:
        from PIL import Image

        pil_images = _to_pil_images(images, input_color_format)
        image_dimensions = [
            ImageDimensions(height=img.height, width=img.width) for img in pil_images
        ]
        prompts = kwargs.get("classes", [])
        if isinstance(prompts, str):
            prompts = [prompts]
        task = kwargs.get("task", "segmentation")
        return (
            _PreProcessedInputs(batch_inputs={}, original_images=pil_images),
            _PreProcessingMetadata(
                image_dimensions=image_dimensions,
                prompts=prompts,
                task=task,
            ),
        )

    def forward(
        self,
        pre_processed_images: _PreProcessedInputs,
        task: str = "segmentation",
        max_new_tokens: int = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MAX_NEW_TOKENS,
        min_image_size: int = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MIN_IMAGE_SIZE,
        max_image_size: int = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MAX_IMAGE_SIZE,
        mask_threshold: float = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MASK_THRESHOLD,
        **kwargs,
    ) -> List[List[Any]]:
        from falcon_perception import build_prompt_for_task
        from falcon_perception.batch_inference import (
            BatchInferenceEngine,
            process_batch_and_generate,
        )

        pil_images = pre_processed_images.original_images
        classes = kwargs.get("classes", [])
        if isinstance(classes, str):
            classes = [classes]
        all_results: List[List[Any]] = []

        engine = BatchInferenceEngine(self._model, self._tokenizer)
        stop_token_ids = [
            self._tokenizer.eos_token_id,
            self._tokenizer.end_of_query_token_id,
        ]

        with self._lock, torch.inference_mode():
            for pil_image in pil_images:
                per_image_results = []
                for prompt_text in classes:
                    prompt = build_prompt_for_task(prompt_text, task)
                    batch_inputs = process_batch_and_generate(
                        self._tokenizer,
                        [(pil_image, prompt)],
                        max_length=4096,
                        min_dimension=min_image_size,
                        max_dimension=max_image_size,
                    )
                    batch_inputs = {
                        k: (v.to(self._device) if torch.is_tensor(v) else v)
                        for k, v in batch_inputs.items()
                    }
                    _, aux_out = engine.generate(
                        **batch_inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        stop_token_ids=stop_token_ids,
                        seed=42,
                        task=task,
                    )
                    per_image_results.append(
                        (prompt_text, aux_out[0] if aux_out else None)
                    )
                all_results.append(per_image_results)
        return all_results

    def post_process(
        self,
        model_results: List[List[Any]],
        pre_processing_meta: _PreProcessingMetadata,
        mask_threshold: float = INFERENCE_MODELS_FALCON_PERCEPTION_DEFAULT_MASK_THRESHOLD,
        **kwargs,
    ) -> List[InstanceDetections]:
        results = []
        for image_idx, per_image_results in enumerate(model_results):
            dims = pre_processing_meta.image_dimensions[image_idx]
            h, w = dims.height, dims.width
            all_xyxy = []
            all_confidence = []
            all_class_ids = []
            all_masks = []

            for prompt_idx, (prompt_text, aux) in enumerate(per_image_results):
                if aux is None:
                    continue
                boxes = _extract_boxes_from_aux(aux, w, h)
                masks = _extract_masks_from_aux(aux, h, w)

                for i, box in enumerate(boxes):
                    all_xyxy.append(box["xyxy"])
                    all_confidence.append(box["confidence"])
                    all_class_ids.append(prompt_idx)
                    if i < len(masks):
                        all_masks.append(masks[i])
                    else:
                        all_masks.append(np.zeros((h, w), dtype=np.uint8))

            if all_xyxy:
                mask_tensor = torch.from_numpy(np.stack(all_masks)).bool()
                results.append(
                    InstanceDetections(
                        xyxy=torch.tensor(all_xyxy, dtype=torch.float32),
                        confidence=torch.tensor(all_confidence, dtype=torch.float32),
                        class_id=torch.tensor(all_class_ids, dtype=torch.int64),
                        mask=mask_tensor,
                    )
                )
            else:
                results.append(
                    InstanceDetections(
                        xyxy=torch.zeros((0, 4), dtype=torch.float32),
                        confidence=torch.zeros((0,), dtype=torch.float32),
                        class_id=torch.zeros((0,), dtype=torch.int64),
                        mask=torch.zeros((0, h, w), dtype=torch.bool),
                    )
                )
        return results


# ── Post-processing helpers ───────────────────────────────────────────


def pair_bbox_entries(raw: List[Dict]) -> List[Dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...].

    Coordinate and size predictions are normalised to [0, 1].
    """
    bboxes: List[Dict] = []
    current: Dict = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes


def normalized_bbox_to_xyxy(
    bbox: Dict[str, float], image_width: int, image_height: int
) -> List[float]:
    """Convert a normalized {x, y, h, w} center-format bbox to [x1, y1, x2, y2] pixel coords."""
    cx = bbox["x"] * image_width
    cy = bbox["y"] * image_height
    bw = bbox["w"] * image_width
    bh = bbox["h"] * image_height
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(float(image_width), cx + bw / 2.0)
    y2 = min(float(image_height), cy + bh / 2.0)
    return [x1, y1, x2, y2]


def _extract_boxes_from_aux(
    aux: Any, image_width: int, image_height: int
) -> List[Dict]:
    """Extract bounding boxes from an AuxOutput object."""
    from falcon_perception.aux_output import AuxOutput

    if not isinstance(aux, AuxOutput):
        return []

    raw_bboxes = aux.bboxes_raw
    if not raw_bboxes:
        raw_bboxes = aux.materialize_bboxes()

    paired = pair_bbox_entries(raw_bboxes)
    results = []
    for bbox in paired:
        xyxy = normalized_bbox_to_xyxy(bbox, image_width, image_height)
        confidence = _compute_bbox_confidence(bbox)
        results.append({"xyxy": xyxy, "confidence": confidence})
    return results


def _compute_bbox_confidence(bbox: Dict[str, float]) -> float:
    """Derive a confidence score from bbox coordinate values.

    Falcon Perception doesn't produce explicit per-instance confidence.
    We use a fixed value of 1.0 since the model only emits instances it
    considers present. Prompt-level presence/absence is handled by
    checking whether any instances were generated at all.
    """
    return 1.0


def _extract_masks_from_aux(
    aux: Any, image_height: int, image_width: int
) -> List[np.ndarray]:
    """Extract binary masks from AuxOutput RLE data."""
    from falcon_perception.aux_output import AuxOutput
    from pycocotools import mask as mask_utils

    if not isinstance(aux, AuxOutput):
        return []

    masks = []
    for rle in aux.masks_rle:
        try:
            rle_for_decode = rle
            if isinstance(rle.get("counts"), str):
                rle_for_decode = {**rle, "counts": rle["counts"].encode("utf-8")}
            binary = mask_utils.decode(rle_for_decode).astype(np.uint8)
            if (binary.shape[0], binary.shape[1]) != (image_height, image_width):
                from PIL import Image

                mask_img = Image.fromarray(binary * 255)
                mask_img = mask_img.resize((image_width, image_height), Image.NEAREST)
                binary = (np.array(mask_img) > 127).astype(np.uint8)
            masks.append(binary)
        except Exception:
            masks.append(np.zeros((image_height, image_width), dtype=np.uint8))
    return masks


# ── Image conversion helpers ──────────────────────────────────────────


def _to_pil_images(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat] = None,
) -> List[Any]:
    """Convert various image input formats to a list of PIL Images (RGB)."""
    from PIL import Image

    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            images = [images]
        elif images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]
        else:
            raise ModelInputError(
                message=f"Unexpected numpy array dimensions: {images.ndim}",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
    elif isinstance(images, torch.Tensor):
        if images.ndim == 3:
            images = [images]
        elif images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]
        else:
            raise ModelInputError(
                message=f"Unexpected tensor dimensions: {images.ndim}",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
    elif not isinstance(images, list):
        raise ModelInputError(
            message=f"Unsupported image input type: {type(images)}",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )

    pil_images = []
    for img in images:
        if isinstance(img, Image.Image):
            pil_images.append(img.convert("RGB"))
        elif isinstance(img, np.ndarray):
            color_fmt = input_color_format or "bgr"
            if color_fmt != "rgb":
                img = np.ascontiguousarray(img[:, :, ::-1])
            pil_images.append(Image.fromarray(img))
        elif isinstance(img, torch.Tensor):
            color_fmt = input_color_format or "rgb"
            arr = img.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if color_fmt != "rgb":
                arr = np.ascontiguousarray(arr[:, :, ::-1])
            pil_images.append(Image.fromarray(arr))
        else:
            raise ModelInputError(
                message=f"Unsupported image element type: {type(img)}",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
    return pil_images
