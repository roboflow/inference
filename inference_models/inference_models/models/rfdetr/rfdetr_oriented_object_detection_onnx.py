"""Native RF-DETR OBB ONNX consumer. Angles are clockwise image-space radians."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from inference_models.entities import ImageDimensions
from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
)
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
)
from inference_models.models.common.roboflow.pre_processing import (
    apply_pre_processing_to_numpy_image,
)


class OrientedClass(BaseModel):
    """Frozen class identity and heading semantics from the training artifact."""

    model_config = ConfigDict(extra="forbid")
    categoryId: int = Field(gt=0, description="Native dataset category identifier.")
    outputIndex: int = Field(
        gt=0, description="Class-logit column; zero is background."
    )
    name: str = Field(min_length=1, description="Frozen class name.")
    heading: StrictBool = Field(description="Whether the class has semantic front.")


class OrientedMetadata(BaseModel):
    """Required semantic contract; an ordinary detection package is never accepted."""

    schemaVersion: int = Field(description="OBB artifact schema version.")
    taskType: str = Field(description="Native model task identity.")
    annotationFormat: str = Field(description="Native annotation geometry format.")
    transformPolicy: str = Field(description="Native shape/front transform policy.")
    classHeading: dict[str, StrictBool] = Field(description="Frozen class heading map.")
    classes: List[OrientedClass] = Field(
        min_length=1, description="Model output mapping."
    )
    versionSchema: dict = Field(
        description="Frozen generated-version annotation schema."
    )
    resolvedRecipe: dict = Field(description="Resolved training recipe.")
    modelRevision: str = Field(
        pattern=r"^[0-9a-f]{40}$", description="Exact trained model commit SHA."
    )
    preprocessing: dict = Field(
        description="Frozen model normalization and letterbox contract."
    )

    @model_validator(mode="after")
    def _check_contract(self):
        if (
            self.schemaVersion,
            self.taskType,
            self.annotationFormat,
            self.transformPolicy,
        ) != (1, "oriented-object-detection", "oriented-box-v1", "obb-front-axis-v1"):
            raise ValueError("Unsupported native OBB artifact contract")
        for key in ("categoryId", "outputIndex", "name"):
            values = [getattr(c, key) for c in self.classes]
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate OBB class {key}")
        if any(self.classHeading.get(c.name) is not c.heading for c in self.classes):
            raise ValueError("classHeading disagrees with frozen classes")
        return self


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
    """The continuous uniform letterbox transform used by the trained model."""

    original_size: ImageDimensions
    scale_width: float
    scale_height: float
    pad_left: float
    pad_top: float
    inference_size: ImageDimensions
    crop_offset_x: int = 0
    crop_offset_y: int = 0


class RFDetrForOrientedObjectDetectionONNX(ObjectDetectionModel):
    """Run the three-output native OBB artifact on the ONNX CPU execution provider.

    CPU ONNX is the advertised initial backend. Other backends require their own
    validation before registration. Model preprocessing is the training Letterbox:
    uniform max-side scale, continuous centered padding and unchanged angle.
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load an exported native artifact with mandatory frozen semantics.

        Args:
            model_name_or_path: Local package directory containing weights.onnx and obb_metadata.json.
            **kwargs: Shared loader options; onnx_execution_providers can select CPU only.

        Returns:
            Initialized native model.

        Raises:
            ValueError: For unsupported backend, malformed outputs or missing metadata.
        """
        import onnxruntime as ort

        directory = Path(model_name_or_path)
        metadata = OrientedMetadata.model_validate_json(
            (directory / "obb_metadata.json").read_text()
        )
        providers = kwargs.get("onnx_execution_providers") or ["CPUExecutionProvider"]
        if providers != ["CPUExecutionProvider"]:
            raise ValueError("Native OBB currently supports CPUExecutionProvider only")
        maximum = kwargs.get("rf_detr_max_input_resolution")
        if maximum is not None:
            maximum = min(maximum) if isinstance(maximum, tuple) else maximum
            if metadata.preprocessing.get("inputSize", 0) > maximum:
                raise ValueError("Native OBB input resolution exceeds configured limit")
        session = ort.InferenceSession(
            str(directory / "weights.onnx"), providers=providers
        )
        config = json.loads((directory / "inference_config.json").read_text())
        image_pre_processing = ImagePreProcessing.model_validate(
            config["image_pre_processing"]
        )
        if config.get("class_names_operations"):
            raise ValueError(
                "Native OBB class remapping requires a new frozen artifact schema"
            )
        model = cls(
            session=session,
            metadata=metadata,
            image_pre_processing=image_pre_processing,
        )
        return model

    def __init__(
        self,
        *,
        session,
        metadata: OrientedMetadata,
        image_pre_processing: ImagePreProcessing,
    ):
        self._session = session
        self.metadata = metadata
        self.image_pre_processing = image_pre_processing
        inputs = session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError("OBB ONNX must have one BCHW image input")
        self._input_name = inputs[0].name
        self._input_batch_size = inputs[0].shape[0]
        shape = inputs[0].shape
        if not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError("OBB ONNX requires a static square input size")
        self.resolution = shape[2]
        expected = {
            "type": "rfdetr-obb-letterbox-v1",
            "inputSize": self.resolution,
            "paddingValue": 114,
            "scale": "uniform-max-side",
            "paddingOffset": "continuous-centered",
            "interpolation": "pillow-bilinear-affine",
            "angle": "unchanged",
            "colorMode": "rgb",
            "scalingFactor": 255,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        if metadata.preprocessing != expected:
            raise ValueError(
                "Native OBB preprocessing metadata is incompatible with this runtime"
            )
        outputs = {o.name: o.shape for o in session.get_outputs()}
        if set(outputs) != {"dets", "labels", "heading"} or any(
            len(s) != 3 for s in outputs.values()
        ):
            raise ValueError(
                "Native OBB requires dets[B,Q,5], labels[B,Q,C], heading[B,Q,4]"
            )
        if outputs["dets"][-1] != 5 or outputs["heading"][-1] != 4:
            raise ValueError("Invalid OBB output geometry or heading shape")
        if (
            not isinstance(outputs["labels"][-1], int)
            or max(c.outputIndex for c in metadata.classes) >= outputs["labels"][-1]
        ):
            raise ValueError("OBB output class mapping exceeds labels dimension")
        self._class_names = [c.name for c in metadata.classes]

    @property
    def class_names(self):
        """Frozen native class names in artifact order."""
        return self._class_names

    def pre_process(self, images, **kwargs):
        """Apply the same continuous letterbox and ImageNet normalization as training.

        Args:
            images: RGB uint8 HWC NumPy images or CHW tensors; lists/batches accepted.
            **kwargs: input_color_format may explicitly be rgb or bgr.

        Returns:
            Float32 BCHW array and exact per-image inverse transform metadata.
        """
        if isinstance(images, torch.Tensor):
            images = list(images) if images.ndim == 4 else [images]
        elif isinstance(images, np.ndarray):
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
            scale = size / max(h, w)
            left, top = (size - w * scale) / 2, (size - h * scale) / 2
            canvas = Image.fromarray(image).transform(
                (size, size),
                Image.Transform.AFFINE,
                (1 / scale, 0, -left / scale, 0, 1 / scale, -top / scale),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(114, 114, 114),
            )
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
                    scale_width=scale,
                    scale_height=scale,
                    pad_left=left,
                    pad_top=top,
                    inference_size=ImageDimensions(height=size, width=size),
                    crop_offset_x=crop.offset_x,
                    crop_offset_y=crop.offset_y,
                )
            )
        batch = (
            np.stack(arrays)
            if arrays
            else np.empty((0, 3, self.resolution, self.resolution), dtype=np.float32)
        )
        return batch, metas

    def forward(self, pre_processed_images, **kwargs):
        """Execute the exported graph, respecting a fixed model batch dimension.

        Args:
            pre_processed_images: Normalized float32 BCHW array.
            **kwargs: Reserved shared inference options.

        Returns:
            Named raw dets, labels and heading arrays before selection.
        """
        batch_limit = (
            self._input_batch_size
            if isinstance(self._input_batch_size, int)
            else len(pre_processed_images)
        )
        outputs = []
        for start in range(0, len(pre_processed_images), batch_limit):
            batch = pre_processed_images[start : start + batch_limit]
            count = len(batch)
            if count < batch_limit:
                batch = np.concatenate(
                    [batch, np.repeat(batch[-1:], batch_limit - count, axis=0)]
                )
            values = self._session.run(
                ["dets", "labels", "heading"], {self._input_name: batch}
            )
            outputs.append([v[:count] for v in values])
        result = tuple(np.concatenate([o[i] for o in outputs]) for i in range(3))
        return result

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        """Select class/query rows and invert the exact training letterbox in float.

        Args:
            model_results: Three raw ONNX arrays in dets, labels, heading order.
            pre_processing_meta: Metadata returned by pre_process.
            **kwargs: confidence (default .5), max_detections (default 300).

        Returns:
            Native boxes with category identity, class semantics and stable query IDs.

        Raises:
            ValueError: If graph values, row alignment or heading are malformed.
        """
        boxes, logits, headings = (torch.as_tensor(v) for v in model_results)
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
        indices = torch.tensor([c.outputIndex for c in self.metadata.classes])
        results = []
        for image_boxes, image_logits, image_heading, meta in zip(
            boxes, logits, headings, pre_processing_meta
        ):
            scores = image_logits[:, indices].sigmoid()
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
                [self.metadata.classes[int(c)].heading for c in cls], dtype=torch.bool
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
            chosen[:, 2:4] /= meta.scale_width
            corners = _corners(chosen)
            bounds = torch.cat((corners.amin(1), corners.amax(1)), dim=-1)
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
                    class_id=torch.tensor(
                        [self.metadata.classes[int(c)].categoryId for c in cls],
                        dtype=torch.int32,
                    ),
                    obb=chosen,
                    heading=directed,
                    detection_id=[
                        f"q{int(q)}:c{self.metadata.classes[int(c)].categoryId}"
                        for q, c in zip(query, cls)
                    ],
                    class_names=[self.metadata.classes[int(c)].name for c in cls],
                )
            )
        return results
