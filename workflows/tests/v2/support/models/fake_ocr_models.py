"""Deterministic stand-ins for the ``inference_models`` OCR backends.

NOT model inference: text and boxes are a fixed function of each image's
size and mean brightness, so tests can predict them exactly::

    brightness b = mean pixel / 255

    FakeDocTR        round(3 * b) words "w0".."w{n-1}" on one line in one block;
                     rows: words (class 2), then the line (1), then the block (0)
    FakeEasyOCR      round(4 * b) regions "r0".., confidence 0.9, 0.6, 0.3, 0.1;
                     post_process drops regions below ``confidence`` (default
                     0.25, so only the block's 0.0 keeps all four)
    FakePPDetector   round(3 * b) line quads, bottom line first; line i spans
                     x 2..12 + 10 * i
    FakePPRecognizer a crop reads "line<width>"; a whole image "image<width>"

Each fake copies its backend's call signature and output types and dtypes
from the source: DocTR's int32 boxes, EasyOCR's float class ids for an image
without text, PP-OCR's ``"polygon"`` row metadata. The PP-OCR fakes run
inside the real ``PPOCRv6Pipeline``. Every fake records its calls.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelRequest

from inference_models.models.base.object_detection import Detections

EASY_OCR_CONFIDENCES = (0.9, 0.6, 0.3, 0.1)
EASY_OCR_DEFAULT_CONFIDENCE = 0.25


def brightness(image: Any) -> float:
    """Mean pixel / 255 of a CHW tensor or HWC array."""
    pixels = image if torch.is_tensor(image) else torch.from_numpy(np.array(image))
    value = float(pixels.float().mean()) / 255.0

    return value


class _Recorded:
    def __init__(self, request: Optional[ModelRequest] = None):
        self.request = request
        self.calls: List[str] = []
        self.batch_sizes: List[int] = []
        self.received: List[Dict[str, Any]] = []


class FakeDocTR(_Recorded):
    """``StructuredOCRModel`` phases shaped like ``DocTR``."""

    class_names = ["block", "line", "word"]

    def pre_process(
        self,
        images: List[torch.Tensor],
        input_color_format: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        assert input_color_format == "rgb"
        self.calls.append("pre_process")
        self.batch_sizes.append(len(images))
        # DocTR: RGB tensors become BGR HWC arrays.
        arrays = [image[[2, 1, 0]].permute(1, 2, 0).cpu().numpy() for image in images]
        dimensions = [array.shape[:2] for array in arrays]

        return arrays, dimensions

    def forward(self, arrays: List[np.ndarray], **kwargs: Any) -> List[int]:
        self.calls.append("forward")
        word_counts = [round(3 * brightness(array)) for array in arrays]

        return word_counts

    def post_process(
        self,
        word_counts: List[int],
        dimensions: List[Tuple[int, int]],
        **kwargs: Any,
    ) -> Tuple[List[str], List[Detections]]:
        self.calls.append("post_process")
        self.received.append(kwargs)
        texts, batch = [], []
        for count, (height, width) in zip(word_counts, dimensions):
            words = [f"w{index}" for index in range(count)]
            texts.append(" ".join(words))
            if not words:
                batch.append(
                    Detections(
                        xyxy=torch.empty((0, 4), dtype=torch.int32),
                        confidence=torch.empty((0,), dtype=torch.float32),
                        class_id=torch.empty((0,), dtype=torch.int32),
                        bboxes_metadata=[],
                    )
                )
                continue

            # Relative geometry, scaled and rounded as DocTR does.
            word_boxes = [[0.1 * i, 0.1, 0.1 * i + 0.08, 0.3] for i in range(count)]
            line_box = [0.0, 0.1, 0.1 * (count - 1) + 0.08, 0.3]
            rows = [
                *[(box, 2, 0.9, word) for box, word in zip(word_boxes, words)],
                (line_box, 1, 0.9, " ".join(words)),
                (line_box, 0, 0.9, " ".join(words)),
            ]
            scale = torch.tensor([width, height, width, height])
            xyxy = (torch.tensor([row[0] for row in rows]) * scale).round().int()
            batch.append(
                Detections(
                    xyxy=xyxy,
                    class_id=torch.tensor([row[1] for row in rows]),
                    confidence=torch.tensor([row[2] for row in rows]),
                    bboxes_metadata=[{"text": row[3]} for row in rows],
                )
            )

        return texts, batch


class FakeEasyOCR(_Recorded):
    """``StructuredOCRModel`` phases shaped like ``EasyOCRTorch``."""

    class_names = ["text-region"]

    def pre_process(
        self,
        images: List[torch.Tensor],
        input_color_format: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        assert input_color_format == "rgb"
        self.calls.append("pre_process")
        self.batch_sizes.append(len(images))
        arrays = [image[[2, 1, 0]].permute(1, 2, 0).cpu().numpy() for image in images]
        dimensions = [array.shape[:2] for array in arrays]

        return arrays, dimensions

    def forward(self, arrays: List[np.ndarray], **kwargs: Any) -> List[List[Any]]:
        # readtext: per region (4 corner points, text, confidence).
        self.calls.append("forward")
        results = []
        for array in arrays:
            count = round(4 * brightness(array))
            results.append(
                [
                    (
                        [
                            [1, 1 + 4 * i],
                            [9, 1 + 4 * i],
                            [9, 4 + 4 * i],
                            [1, 4 + 4 * i],
                        ],
                        f"r{i}",
                        EASY_OCR_CONFIDENCES[i],
                    )
                    for i in range(count)
                ]
            )

        return results

    def post_process(
        self,
        model_results: List[List[Any]],
        dimensions: List[Tuple[int, int]],
        confidence: float = EASY_OCR_DEFAULT_CONFIDENCE,
        text_regions_separator: str = " ",
        **kwargs: Any,
    ) -> Tuple[List[str], List[Detections]]:
        # The EasyOCRTorch.post_process body, including its tensor dtypes.
        self.calls.append("post_process")
        self.received.append({"confidence": confidence, **kwargs})
        texts, batch = [], []
        for regions in model_results:
            kept = [region for region in regions if region[2] >= confidence]
            xyxy = [
                [
                    min(p[0] for p in box),
                    min(p[1] for p in box),
                    max(p[0] for p in box),
                    max(p[1] for p in box),
                ]
                for box, _, _ in kept
            ]
            texts.append(text_regions_separator.join(text for _, text, _ in kept))
            batch.append(
                Detections(
                    xyxy=torch.tensor(xyxy) if xyxy else torch.empty((0, 4)),
                    class_id=torch.tensor([0 for _ in kept]),
                    confidence=torch.tensor([float(c) for _, _, c in kept]),
                    bboxes_metadata=[{"text": text} for _, text, _ in kept],
                )
            )

        return texts, batch


class FakePPDetector(_Recorded):
    """Callable text-line detector shaped like the PP-OCRv6 ONNX detector."""

    class_names = ["text"]

    def __call__(
        self,
        images: List[torch.Tensor],
        input_color_format: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Detections]:
        assert input_color_format == "rgb"
        self.calls.append("detect")
        self.batch_sizes.append(len(images))
        self.received.append(kwargs)
        batch = []
        for image in images:
            count = round(3 * brightness(image))
            # Bottom line first, so the pipeline's reading order is visible.
            quads = [
                [
                    [2.0, top],
                    [12.0 + 10 * i, top],
                    [12.0 + 10 * i, top + 6],
                    [2.0, top + 6],
                ]
                for i, top in ((i, 2.0 + 8 * (count - 1 - i)) for i in range(count))
            ]
            xyxy = [[q[0][0], q[0][1], q[2][0], q[2][1]] for q in quads]
            batch.append(
                Detections(
                    xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
                    class_id=torch.zeros((count,), dtype=torch.int64),
                    confidence=torch.tensor(
                        [0.9 - 0.1 * i for i in range(count)], dtype=torch.float32
                    ),
                    bboxes_metadata=[{"polygon": quad} for quad in quads],
                )
            )

        return batch


class FakePPRecognizer(_Recorded):
    """Callable line recognizer shaped like the PP-OCRv6 ONNX recognizer."""

    def __call__(
        self,
        images: Sequence[Any],
        input_color_format: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        self.calls.append("recognize")
        self.batch_sizes.append(len(images))
        self.received.append({"input_color_format": input_color_format, **kwargs})
        texts = [
            # Tensors are whole CHW images; arrays are the pipeline's HWC crops.
            (
                f"image{image.shape[-1]}"
                if torch.is_tensor(image)
                else f"line{image.shape[1]}"
            )
            for image in images
        ]

        return texts


FAKES_BY_PREFIX = {
    "doctr/": FakeDocTR,
    "easy_ocr/": FakeEasyOCR,
    "pp-ocrv6-det/": FakePPDetector,
    "pp-ocrv6-rec/": FakePPRecognizer,
}


class FakeOCRLoader:
    """``ModelCache`` loader: the fake whose prefix starts the model id."""

    def __init__(self) -> None:
        self.loaded: List[ModelRequest] = []
        self.models: Dict[ModelRequest, _Recorded] = {}

    def __call__(self, request: ModelRequest) -> _Recorded:
        self.loaded.append(request)
        fake = next(
            fake
            for prefix, fake in FAKES_BY_PREFIX.items()
            if request.model_id.startswith(prefix)
        )
        model = fake(request)
        self.models[request] = model

        return model
