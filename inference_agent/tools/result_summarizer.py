"""Summarizes raw inference output into concise, token-efficient text for Claude."""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Optional

from inference_agent.core.protocols import InferenceResult

logger = logging.getLogger(__name__)


class ResultSummarizer:
    """Converts raw inference output to human-readable summaries.

    This is critical for controlling token usage. Raw workflow output can be
    10K+ tokens per frame. The summarizer reduces it to ~100 tokens.
    """

    def __init__(self, max_image_dimension: int = 640):
        self._max_image_dimension = max_image_dimension

    def summarize(
        self,
        result: InferenceResult,
        include_frame: bool = False,
    ) -> str | list:
        """Summarize an inference result.

        Returns a string, or a list of content blocks if include_frame is True.
        """
        text = self._summarize_predictions(result.predictions, result.frame_id)

        if include_frame and result.frame:
            image_data = self._prepare_image(result.frame)
            if image_data:
                return [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": text},
                ]
        return text

    def summarize_multiple(
        self,
        results: list[InferenceResult],
        include_last_frame: bool = False,
    ) -> str | list:
        """Summarize multiple pipeline results into a single summary."""
        if not results:
            return "No results available."

        parts = []
        for i, result in enumerate(results):
            summary = self._summarize_predictions(result.predictions, result.frame_id)
            parts.append(summary)

        text = "\n".join(parts)
        text += f"\n\n({len(results)} result(s) total)"

        if include_last_frame and results[-1].frame:
            image_data = self._prepare_image(results[-1].frame)
            if image_data:
                return [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": text},
                ]
        return text

    def _summarize_predictions(
        self, predictions: dict, frame_id: Optional[int] = None
    ) -> str:
        """Convert raw prediction dict to concise text."""
        prefix = f"Frame #{frame_id}: " if frame_id is not None else ""

        # Handle workflow outputs (list of dicts from run_workflow)
        if "outputs" in predictions and isinstance(predictions["outputs"], list):
            return self._summarize_workflow_outputs(predictions["outputs"], prefix)

        # Handle direct model predictions
        if "predictions" in predictions:
            return self._summarize_detections(predictions, prefix)

        # Handle classification output
        if "top" in predictions or "confidence" in predictions:
            return self._summarize_classification(predictions, prefix)

        # Fallback: compact JSON
        compact = json.dumps(predictions, default=str)
        if len(compact) > 500:
            compact = compact[:500] + "... (truncated)"
        return f"{prefix}{compact}"

    def _summarize_workflow_outputs(self, outputs: list, prefix: str) -> str:
        """Summarize workflow output list."""
        parts = [prefix] if prefix else []
        for output in outputs:
            if not isinstance(output, dict):
                continue
            for key, value in output.items():
                if isinstance(value, dict) and "predictions" in value:
                    det_summary = self._format_detections(value["predictions"])
                    parts.append(f"{key}: {det_summary}")
                elif isinstance(value, (int, float, str, bool)):
                    parts.append(f"{key}={value}")
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict) and "class" in value[0]:
                        det_summary = self._format_detections(value)
                        parts.append(f"{key}: {det_summary}")
                    else:
                        parts.append(f"{key}: {len(value)} items")
                elif isinstance(value, dict):
                    compact = json.dumps(value, default=str)
                    if len(compact) > 200:
                        compact = compact[:200] + "..."
                    parts.append(f"{key}: {compact}")
        return " | ".join(parts) if parts else f"{prefix}(empty output)"

    def _summarize_detections(self, predictions: dict, prefix: str) -> str:
        """Summarize object detection output."""
        detections = predictions.get("predictions", [])
        det_summary = self._format_detections(detections)
        return f"{prefix}{det_summary}"

    def _summarize_classification(self, predictions: dict, prefix: str) -> str:
        """Summarize classification output."""
        top = predictions.get("top", "unknown")
        confidence = predictions.get("confidence", 0)
        return f"{prefix}Classification: {top} ({confidence:.2f})"

    def _format_detections(self, detections: list) -> str:
        """Format a list of detections into a concise summary."""
        if not detections:
            return "0 detections"

        # Group by class
        by_class: dict[str, list[float]] = {}
        for det in detections:
            cls = det.get("class", det.get("class_name", "object"))
            conf = det.get("confidence", 0)
            by_class.setdefault(cls, []).append(conf)

        total = sum(len(v) for v in by_class.values())
        parts = []
        for cls, confs in sorted(by_class.items(), key=lambda x: -len(x[1])):
            avg_conf = sum(confs) / len(confs)
            if len(confs) == 1:
                parts.append(f"{cls} ({confs[0]:.2f})")
            else:
                parts.append(f"{len(confs)} {cls} (avg {avg_conf:.2f})")

        return f"{total} detections: {', '.join(parts)}"

    def _prepare_image(self, frame_bytes: bytes) -> Optional[str]:
        """Resize and base64-encode a JPEG frame for Claude."""
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(frame_bytes))
            # Resize if needed
            max_dim = max(img.size)
            if max_dim > self._max_image_dimension:
                scale = self._max_image_dimension / max_dim
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode("ascii")
        except ImportError:
            # PIL not available — send raw bytes
            return base64.b64encode(frame_bytes).decode("ascii")
        except Exception as e:
            logger.warning("Failed to prepare image: %s", e)
            return None
