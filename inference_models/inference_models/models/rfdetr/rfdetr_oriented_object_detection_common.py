"""Backend-independent parts of the native RF-DETR OBB consumers. Angles are clockwise image-space radians.

The ONNX and TRT models share the package contract (frozen class heading policy, class mapping, declared resize),
the preprocessing that reproduces the training images and the decoding of ``dets``/``labels``/``heading``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from inference_models.entities import ImageDimensions
from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    InferenceConfig,
    NetworkInputDefinition,
    ResizeMode,
    align_training_input_size_with_model,
    ensure_input_size_within_limit,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.oriented_box_metadata import (
    OrientedBoxMetadata,
)
from inference_models.models.common.roboflow.pre_processing import (
    apply_pre_processing_to_numpy_image,
)
from inference_models.models.rfdetr.class_remapping import prepare_class_remapping

OUTPUT_NAMES = ("dets", "labels", "heading")


@dataclass
class OrientedDetections(Detections):
    """Float native boxes; compatibility bounds and corners are derived from obb."""

    obb: torch.Tensor = field(kw_only=True)
    heading: torch.Tensor = field(kw_only=True)
    detection_id: List[str] = field(kw_only=True)
    class_names: List[str] = field(kw_only=True)

    def to_supervision(self):
        """Convert to native Workflow transport without dropping front.

        Returns:
            Supervision detections with canonical OBB and derived ordered corners.
        """
        import supervision as sv

        result = sv.Detections(
            xyxy=self.xyxy.detach().cpu().numpy(),
            confidence=self.confidence.detach().cpu().numpy(),
            class_id=self.class_id.detach().cpu().numpy(),
            data={
                "xyxyxyxy": _corners(self.obb).detach().cpu().numpy(),
                "heading": self.heading.detach().cpu().numpy(),
                "native_obb": np.ones(len(self), dtype=bool),
                "detection_id": np.asarray(self.detection_id),
                "class_name": np.asarray(self.class_names),
            },
            metadata=self.image_metadata or {},
        )
        return result


def _corners(obb):
    signs = obb.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    local = signs[None] * obb[:, None, 2:4] / 2
    c, s = obb[:, 4].cos(), obb[:, 4].sin()
    x = local[:, :, 0] * c[:, None] - local[:, :, 1] * s[:, None]
    y = local[:, :, 0] * s[:, None] + local[:, :, 1] * c[:, None]
    return torch.stack((x, y), dim=-1) + obb[:, None, :2]


@dataclass(frozen=True)
class OrientedPreprocessingMetadata:
    """Realized input transform: per-axis scale, then the offset of the image's origin in the model input.

    A letterbox scales and pads (offsets >= 0); a center crop keeps the scale at 1 and offsets by the crop (< 0) or
    the padding (> 0).
    """

    original_size: ImageDimensions
    scale_width: float
    scale_height: float
    pad_left: float
    pad_top: float
    inference_size: ImageDimensions
    crop_offset_x: int = 0
    crop_offset_y: int = 0


@dataclass(frozen=True)
class OrientedPackage:
    """A native OBB package's parsed contract, independent of the backend."""

    metadata: OrientedBoxMetadata
    class_names: List[str]
    output_indices: List[int]
    inference_config: InferenceConfig
    files: Dict[str, str]


def load_oriented_package(
    model_name_or_path: str,
    raw_metadata: Optional[dict],
    weights_files: List[str],
    max_input_resolution=None,
) -> OrientedPackage:
    """Parse the files every native OBB package shares, plus the backend's *weights_files*.

    Raises:
        ValueError: Without frozen class heading metadata, or when the package's classes disagree with it.
    """
    if raw_metadata is None:
        raise ValueError("Native OBB requires frozen orientedBoxMetadata from Core")
    metadata = OrientedBoxMetadata.model_validate(raw_metadata)
    files = get_model_package_contents(
        model_package_dir=model_name_or_path,
        elements=["class_names.txt", "inference_config.json", *weights_files],
    )
    class_names = parse_class_names_file(files["class_names.txt"])
    config = parse_inference_config(
        config_path=files["inference_config.json"],
        allowed_resize_modes={ResizeMode.LETTERBOX, ResizeMode.CENTER_CROP},
        max_allowed_input_size=max_input_resolution,
    )
    indices = list(range(len(class_names)))
    if config.class_names_operations:
        class_names, remapping = prepare_class_remapping(
            class_names=class_names,
            class_names_operations=config.class_names_operations,
            device=torch.device("cpu"),
        )
        indices = remapping.remaining_class_ids.tolist()
    if not class_names or any(
        name not in metadata.classHeading for name in class_names
    ):
        raise ValueError(
            "Native OBB class names disagree with frozen class heading policy"
        )
    return OrientedPackage(
        metadata=metadata,
        class_names=class_names,
        output_indices=indices,
        inference_config=config,
        files=files,
    )


def check_model_input(
    package: OrientedPackage, resolution: int, max_input_resolution=None
) -> None:
    """Check the declared preprocessing against the model's own square input size.

    Raises:
        ValueError: When the declared preprocessing is not the training one (RGB, /255, ImageNet normalization, a
            letterbox padding value in [0, 255]).
    """
    config = align_training_input_size_with_model(
        package.inference_config,
        model_input_height=resolution,
        model_input_width=resolution,
    )
    ensure_input_size_within_limit(
        inference_config=config, max_allowed_input_size=max_input_resolution
    )
    network = config.network_input
    if (
        network.color_mode != ColorMode.RGB
        or network.input_channels != 3
        or (
            network.resize_mode == ResizeMode.LETTERBOX
            and (
                not isinstance(network.padding_value, int)
                or not 0 <= network.padding_value <= 255
            )
        )
        or network.scaling_factor != 255
        or network.normalization != ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ):
        raise ValueError("Native OBB package preprocessing differs from training")


def check_output_shapes(shapes: Dict[str, Tuple], output_indices: List[int]) -> None:
    """Check ``dets[B,Q,5]``, ``labels[B,Q,C]`` and ``heading[B,Q,4]`` against the class mapping."""
    if set(shapes) != set(OUTPUT_NAMES) or any(len(s) != 3 for s in shapes.values()):
        raise ValueError(
            "Native OBB requires dets[B,Q,5], labels[B,Q,C], heading[B,Q,4]"
        )
    if shapes["dets"][-1] != 5 or shapes["heading"][-1] != 4:
        raise ValueError("Invalid OBB output geometry or heading shape")
    labels = shapes["labels"][-1]
    if not isinstance(labels, int) or labels <= 0 or max(output_indices) >= labels:
        raise ValueError("OBB output class mapping exceeds labels dimension")


class RFDetrForOrientedObjectDetectionBase(ObjectDetectionModel):
    """Preprocessing and decoding shared by the native OBB backends; subclasses load the graph and run ``forward``.

    Preprocessing follows the package's declared resize, as the training images saw it: ``letterbox`` is the training
    Letterbox (rounded max-side resize, integer centered padding in the declared ``padding_value``) and
    ``center-crop`` is the version's Fill (with center crop) in (no scaling, centered crop, black padding when the
    image is smaller). Angles are unchanged by either.
    """

    def __init__(
        self,
        *,
        metadata: OrientedBoxMetadata,
        class_names: List[str],
        output_indices: List[int],
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        resolution: int,
    ):
        self.metadata = metadata
        self.image_pre_processing = image_pre_processing
        self._resize_mode = network_input.resize_mode
        self._padding_value = network_input.padding_value
        self.resolution = resolution
        self._class_names = class_names
        self._output_indices = output_indices

    @property
    def class_names(self):
        """Frozen native class names in artifact order."""
        return self._class_names

    def pre_process(self, images, **kwargs):
        """Apply the package's resize (letterbox or center crop) and ImageNet normalization, as in training.

        Args:
            images: RGB uint8 HWC NumPy images or CHW tensors; lists/batches accepted.
            **kwargs: input_color_format may explicitly be rgb or bgr.

        Returns:
            Float32 BCHW array and exact per-image inverse transform metadata.
        """
        if isinstance(images, (torch.Tensor, np.ndarray)):
            images = list(images) if images.ndim == 4 else [images]
        if not len(images):
            raise ValueError("Native OBB requires a nonempty image batch")
        arrays, metas = [], []
        for image in images:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().permute(1, 2, 0).numpy()
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Native OBB requires uint8 three-channel images")
            if kwargs.get("input_color_format", "rgb") == "bgr":
                image = image[:, :, ::-1]
            original_height, original_width = image.shape[:2]
            image, crop = apply_pre_processing_to_numpy_image(
                image=image,
                image_pre_processing=self.image_pre_processing,
                network_input_channels=3,
                input_color_mode=ColorMode.RGB,
                pre_processing_overrides=kwargs.get("pre_processing_overrides"),
            )
            h, w = image.shape[:2]
            size = self.resolution
            if self._resize_mode == ResizeMode.CENTER_CROP:
                # The version's Fill (with center crop) in: no scaling; the image is centered on the input, cropped
                # where larger and padded black where smaller, with the extra odd pixel on the right / bottom.
                resized_width, resized_height = w, h
                left = (size - w) // 2 if size >= w else -((w - size) // 2)
                top = (size - h) // 2 if size >= h else -((h - size) // 2)
                resized = Image.fromarray(image)
                fill = (0, 0, 0)
            else:
                scale = size / max(h, w)
                resized_width = max(1, round(w * scale))
                resized_height = max(1, round(h * scale))
                left, top = (size - resized_width) // 2, (size - resized_height) // 2
                resized = Image.fromarray(image).resize(
                    (resized_width, resized_height), resample=Image.Resampling.BILINEAR
                )
                fill = (self._padding_value,) * 3
            canvas = Image.new("RGB", (size, size), fill)
            canvas.paste(resized, (left, top))
            array = np.asarray(canvas).astype(np.float32) / 255.0
            array = (
                array - np.array([0.485, 0.456, 0.406], dtype=np.float32)
            ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arrays.append(array.transpose(2, 0, 1))
            metas.append(
                OrientedPreprocessingMetadata(
                    original_size=ImageDimensions(
                        height=original_height, width=original_width
                    ),
                    scale_width=resized_width / w,
                    scale_height=resized_height / h,
                    pad_left=left,
                    pad_top=top,
                    inference_size=ImageDimensions(height=size, width=size),
                    crop_offset_x=crop.offset_x,
                    crop_offset_y=crop.offset_y,
                )
            )
        batch = np.stack(arrays)
        return batch, metas

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        """Select class/query rows and invert the input transform (letterbox or center crop) in float.

        Args:
            model_results: Three raw arrays or tensors in dets, labels, heading order.
            pre_processing_meta: Metadata returned by pre_process.
            **kwargs: confidence (default .5), max_detections (default 300).

        Returns:
            Native boxes with category identity, class semantics and unique detection IDs.

        Raises:
            ValueError: If graph values, row alignment or heading are malformed.
        """
        boxes, logits, headings = (
            torch.as_tensor(v).detach().cpu() for v in model_results
        )
        if (
            boxes.ndim != 3
            or boxes.shape[-1] != 5
            or headings.shape != (*boxes.shape[:2], 4)
            or logits.shape[:2] != boxes.shape[:2]
            or len(boxes) != len(pre_processing_meta)
        ):
            raise ValueError("Unaligned native OBB graph outputs")
        if not all(torch.isfinite(v).all() for v in (boxes, logits, headings)):
            raise ValueError(
                "Native OBB geometry/logits must be finite with positive sides"
            )
        threshold = kwargs.get("confidence")
        threshold = 0.5 if threshold is None else threshold
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("Native OBB confidence must be a number in [0,1]")
        cap = kwargs.get("max_detections")
        cap = 300 if cap is None else cap
        if not isinstance(cap, int) or cap < 0:
            raise ValueError("Native OBB max_detections must be a nonnegative integer")
        indices = torch.tensor(self._output_indices)
        results = []
        for image_boxes, image_logits, image_heading, meta in zip(
            boxes, logits, headings, pre_processing_meta
        ):
            scores = image_logits[:, indices].float().sigmoid()
            # Exported graphs can have zero-filled unused query slots. They cannot
            # describe a rectangle, regardless of the class logits in that slot.
            scores = torch.where(
                (image_boxes[:, 2:4] > 0).all(-1)[:, None], scores, -1.0
            )
            flat = scores.flatten()
            selected = torch.argsort(flat, descending=True, stable=True)[:cap]
            selected = selected[flat[selected] > threshold]
            query, cls = selected // len(indices), selected % len(indices)
            chosen = image_boxes[query].clone().float()
            sides = image_heading[query].argmax(-1)
            directed = torch.tensor(
                [self.metadata.classHeading[self._class_names[int(c)]] for c in cls],
                dtype=torch.bool,
            )
            sides = torch.where(directed, sides, 0)
            swap = sides % 2 == 1
            chosen[swap, 2:4] = chosen[swap, 2:4].flip(-1)
            chosen[:, 4] = (chosen[:, 4] + sides * (torch.pi / 2) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            chosen[:, :4] *= self.resolution
            chosen[:, 0] = (
                chosen[:, 0] - meta.pad_left
            ) / meta.scale_width + meta.crop_offset_x
            chosen[:, 1] = (
                chosen[:, 1] - meta.pad_top
            ) / meta.scale_height + meta.crop_offset_y
            # Training scales ordered sides by the mean of the realized resize
            # ratios, while centers use each axis's own ratio.
            chosen[:, 2:4] /= (meta.scale_width + meta.scale_height) / 2
            corners = _corners(chosen)
            bounds = torch.cat((corners.amin(1), corners.amax(1)), dim=-1)
            result_id = uuid4().hex
            results.append(
                OrientedDetections(
                    image_metadata={
                        "native_obb": True,
                        "class_heading": dict(self.metadata.classHeading),
                        "image_dimensions": [
                            meta.original_size.height,
                            meta.original_size.width,
                        ],
                    },
                    xyxy=bounds,
                    confidence=flat[selected],
                    class_id=cls.to(device="cpu", dtype=torch.int32),
                    obb=chosen,
                    heading=directed,
                    detection_id=[
                        f"{result_id}:q{int(q)}:c{int(c)}" for q, c in zip(query, cls)
                    ],
                    class_names=[self._class_names[int(c)] for c in cls],
                )
            )
        return results
