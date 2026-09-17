"""Load the versioned, self-contained V-JEPA action-recognition package."""

import json
import math
from contextlib import nullcontext
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torchvision.transforms import v2

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.base.action_recognition import (
    ActionRecognitionModel,
    ActionRecognitionPrediction,
    VideoSampling,
)
from inference_models.models.common.model_packages import get_model_package_contents

from .head import SpanHead
from .vendor.encoder import vit_base


def validate_config(config):
    expected = {
        "schema_version": 1,
        "architecture": "vjepa2_1",
        "variant": "vitb-384",
        "method_id": "frame_anchored_multilabel_spans_v1",
        "temporal_semantics": {
            "attention_mode": "bidirectional",
            "context_mode": "independent_windows",
            "cache_mode": "none",
            "position_origin": "window_relative",
        },
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported V-JEPA contract: {key}")
    inputs, head, post = (
        config["network_input"],
        config["head"],
        config["post_processing"],
    )
    frames = inputs["frames"]
    if type(frames) is not int or not 2 <= frames <= 256 or frames % 2:
        raise ValueError("V-JEPA requires an even frame count between 2 and 256")
    if not math.isfinite(inputs["fps"]) or inputs["fps"] <= 0:
        raise ValueError("V-JEPA requires a positive finite FPS")
    contracts = [
        (
            inputs,
            {
                "layout": "BCTHW",
                "resize": "direct_square",
                "height": 384,
                "width": 384,
                "color_mode": "rgb",
                "interpolation": "bilinear",
                "antialias": True,
                "scaling_factor": 255,
                "padding": "repeat_last_frame_with_validity_mask",
                "sampling": "first_decoded_frame_at_or_after_requested_timestamp",
            },
        ),
        (
            head,
            {
                "type": "frame_anchored_class_spans",
                "version": 1,
                "queries": frames,
                "query_type": "learned_per_frame",
                "dim": 768,
                "num_heads": 12,
                "depth": 1,
                "mlp_ratio": 4,
                "pooling": "global_meta_cross_attention",
                "added_memory_position": None,
                "channel_layout": "class_logits_then_class_specific_left_right",
                "class_activation": "independent_sigmoid",
                "offset_activation": "sigmoid",
                "offset_scale_frames": frames,
                "anchor": "frame_index_plus_half",
            },
        ),
        (
            post,
            {
                "type": "multilabel_class_union",
                "version": 1,
                "tail_policy": "end_aligned",
                "clip_to_window": True,
                "same_class_merge": "overlap_or_touch",
                "merged_confidence": "maximum",
                "cross_class_overlap": "preserved",
                "single_label_projection": None,
                "fill_unlabeled_frames": False,
                "threshold_comparison": ">=",
                "nms": None,
            },
        ),
    ]
    for actual, required in contracts:
        if any(
            key not in actual or actual[key] != value for key, value in required.items()
        ):
            raise ValueError(
                "Unsupported V-JEPA preprocessing, head, or postprocessing"
            )
    expected_encoder = dict(
        img_size=[384, 384],
        num_frames=frames,
        patch_size=16,
        tubelet_size=2,
        use_rope=True,
        uniform_power=True,
        interpolate_rope=True,
        img_temporal_dim_size=1,
        modality_embedding=True,
        n_output_distillation=1,
        use_sdpa=True,
        is_causal=False,
    )
    if (
        config["encoder"]["factory"] != "vit_base"
        or config["encoder"]["arguments"] != expected_encoder
    ):
        raise ValueError("Unsupported V-JEPA encoder arguments")
    if (
        type(post["overlap_frames"]) is not int
        or not 0 <= post["overlap_frames"] < frames
    ):
        raise ValueError("Invalid V-JEPA overlap")
    if not 0 <= post["confidence_threshold"] <= 1:
        raise ValueError("Invalid V-JEPA confidence threshold")
    if inputs["autocast"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported V-JEPA precision")
    mean, std = inputs["normalization"]
    if (
        len(mean) != 3
        or len(std) != 3
        or not all(math.isfinite(v) for v in mean + std)
        or min(std) <= 0
    ):
        raise ValueError("Invalid V-JEPA normalization")


class VJepaActionRecognition(ActionRecognitionModel):
    span_semantics = "class_union"

    @classmethod
    def from_pretrained(cls, model_name_or_path, device=DEFAULT_DEVICE, **kwargs):
        files = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["model.safetensors", "inference_config.json", "class_names.txt"],
        )
        config = json.loads(Path(files["inference_config.json"]).read_text())
        validate_config(config)
        classes = Path(files["class_names.txt"]).read_text().splitlines()
        if (
            not classes
            or len(set(classes)) != len(classes)
            or config["class_names"] != classes
        ):
            raise ValueError("V-JEPA class list is empty, duplicated, or inconsistent")
        model = nn.Module()
        model.encoder = vit_base(**config["encoder"]["arguments"])
        model.head = SpanHead(config["head"]["queries"], len(classes))
        model.load_state_dict(
            load_file(files["model.safetensors"], device="cpu"), strict=True
        )
        return cls(
            model.to(device).float().eval(), config, classes, torch.device(device)
        )

    def __init__(self, model, config, classes, device):
        self._model, self._config, self._classes, self._device = (
            model,
            config,
            classes,
            device,
        )
        self._lock = Lock()
        inputs = config["network_input"]
        self._dtype = getattr(torch, inputs["autocast"])
        self._transform = v2.Compose(
            [
                v2.Resize(
                    (inputs["height"], inputs["width"]), antialias=inputs["antialias"]
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(*inputs["normalization"]),
            ]
        )

    @property
    def class_names(self):
        return list(self._classes)

    @property
    def confidence_threshold(self):
        return self._config["post_processing"]["confidence_threshold"]

    @property
    def video_sampling(self):
        inputs, post = self._config["network_input"], self._config["post_processing"]
        return VideoSampling(
            window_seconds=inputs["frames"] / inputs["fps"],
            sample_fps=inputs["fps"],
            min_frames=1,
            max_frames=inputs["frames"],
            overlap_frames=post["overlap_frames"],
            end_aligned=True,
            fixed_sample_fps=True,
        )

    @torch.inference_mode()
    def infer(
        self,
        frames,
        class_names=None,
        fps=None,
        confidence=None,
        duration_seconds=None,
        **kwargs,
    ):
        sampling = self.video_sampling
        if not frames or len(frames) > sampling.max_frames:
            raise ValueError(
                "V-JEPA needs between one frame and its recorded window length"
            )
        if fps is not None and not math.isclose(fps, sampling.sample_fps):
            raise ValueError("V-JEPA input FPS must match its recorded sampling rate")
        if class_names is not None and not set(class_names) <= set(self._classes):
            raise ValueError("Unknown V-JEPA class filter")
        threshold = self.confidence_threshold if confidence is None else confidence
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("Confidence must be a number between zero and one")
        images = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().permute(1, 2, 0).numpy()
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("V-JEPA expects RGB uint8 frames")
            images.append(self._transform(Image.fromarray(frame)))
        count = len(images)
        end_limit = (
            count
            if duration_seconds is None
            else duration_seconds * sampling.sample_fps
        )
        if (
            not math.isfinite(end_limit)
            or not 0 < end_limit <= sampling.max_frames + 1e-6
        ):
            raise ValueError("Invalid V-JEPA window duration")
        images.extend([images[-1]] * (sampling.max_frames - count))
        inputs = torch.stack(images, dim=1)[None].to(self._device)
        autocast = (
            torch.autocast("cuda", dtype=self._dtype)
            if self._device.type == "cuda" and self._dtype != torch.float32
            else nullcontext()
        )
        with self._lock, autocast:
            logits, intervals = self._model.head(self._model.encoder(inputs))
        if not torch.isfinite(logits).all() or not torch.isfinite(intervals).all():
            raise FloatingPointError("V-JEPA produced nonfinite predictions")
        scores = logits[0, :count].float().sigmoid()
        rows, columns = (scores >= threshold).nonzero(as_tuple=True)
        spans = intervals[0, rows, columns].float().clamp(0, end_limit).cpu().tolist()
        confidences = scores[rows, columns].cpu().tolist()
        return [
            ActionRecognitionPrediction(
                left, right, self._classes[label], score, end_exclusive=True
            )
            for (left, right), score, label in zip(
                spans, confidences, columns.cpu().tolist()
            )
            if right > left
            and (class_names is None or self._classes[label] in class_names)
        ]
