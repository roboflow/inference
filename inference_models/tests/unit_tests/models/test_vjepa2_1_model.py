import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_models.models.vjepa2_1.model import (
    VJepaActionRecognition,
    validate_config,
)


def config():
    return {
        "schema_version": 1,
        "architecture": "vjepa2_1",
        "variant": "vitb-384",
        "method_id": "frame_anchored_multilabel_spans_v1",
        "class_names": ["a", "b"],
        "temporal_semantics": {
            "attention_mode": "bidirectional",
            "context_mode": "independent_windows",
            "cache_mode": "none",
            "position_origin": "window_relative",
        },
        "network_input": {
            "layout": "BCTHW",
            "frames": 4,
            "fps": 4.0,
            "height": 384,
            "width": 384,
            "resize": "direct_square",
            "color_mode": "rgb",
            "interpolation": "bilinear",
            "antialias": True,
            "scaling_factor": 255,
            "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
            "autocast": "bfloat16",
            "sampling": "first_decoded_frame_at_or_after_requested_timestamp",
            "padding": "repeat_last_frame_with_validity_mask",
        },
        "encoder": {
            "factory": "vit_base",
            "arguments": {
                "img_size": [384, 384],
                "num_frames": 4,
                "patch_size": 16,
                "tubelet_size": 2,
                "use_rope": True,
                "uniform_power": True,
                "interpolate_rope": True,
                "img_temporal_dim_size": 1,
                "modality_embedding": True,
                "n_output_distillation": 1,
                "use_sdpa": True,
                "is_causal": False,
            },
        },
        "head": {
            "type": "frame_anchored_class_spans",
            "version": 1,
            "queries": 4,
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
            "offset_scale_frames": 4,
            "anchor": "frame_index_plus_half",
        },
        "post_processing": {
            "type": "multilabel_class_union",
            "version": 1,
            "overlap_frames": 2,
            "confidence_threshold": 0.5,
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
    }


def test_contract_rejects_causal_or_different_head_artifacts():
    metadata = config()
    validate_config(metadata)
    for section, key, value in [
        ("temporal_semantics", "attention_mode", "causal"),
        ("head", "class_activation", "softmax"),
        ("network_input", "resize", "center_crop"),
    ]:
        altered = copy.deepcopy(metadata)
        altered[section][key] = value
        with pytest.raises(ValueError):
            validate_config(altered)


def test_infer_thresholds_candidates_and_masks_padded_queries():
    captured = []

    def encoder(inputs):
        captured.append(inputs)
        return inputs

    def head(_features):
        logits = torch.tensor([[[2.0, -2.0], [2.0, -2.0], [20.0, 20.0], [20.0, 20.0]]])
        intervals = torch.tensor([[[[0.0, 2.0], [0.0, 2.0]]] * 4])
        return logits, intervals

    model = VJepaActionRecognition(
        SimpleNamespace(encoder=encoder, head=head),
        config(),
        ["a", "b"],
        torch.device("cpu"),
    )
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    frame[:, :10, 0] = 255
    frame[:, -10:, 1] = 255
    predictions = model.infer([frame, frame], fps=4)
    assert len(predictions) == 2
    assert all(row.class_name == "a" and row.end_exclusive for row in predictions)
    assert len(model.infer([frame, frame], confidence=0)) == 4
    assert model.infer([frame, frame], confidence=1) == []
    assert captured[0].shape == (1, 3, 4, 384, 384)
    assert captured[0][0, 0, 0, 192, 0] > 0
    assert captured[0][0, 1, 0, 192, -1] > 0
    assert torch.equal(captured[0][:, :, 1], captured[0][:, :, 3])
