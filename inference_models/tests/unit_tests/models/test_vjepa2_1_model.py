import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from inference_models.errors import ModelInputError
from inference_models.models.base.action_recognition import plan_windows
from inference_models.models.vjepa2_1.model import (
    VJepaActionRecognition,
    validate_config,
)
from inference_models.weights_providers.entities import RecommendedParameters


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

    metadata["network_input"].update(frames=160, height=512, width=512)
    metadata["encoder"]["arguments"].update(num_frames=160, img_size=[512, 512])
    metadata["head"].update(queries=160, offset_scale_frames=160)
    validate_config(metadata)
    metadata["network_input"]["frames"] = 162
    metadata["encoder"]["arguments"]["num_frames"] = 162
    metadata["head"].update(queries=162, offset_scale_frames=162)
    with pytest.raises(ValueError, match="81920 vision tokens"):
        validate_config(metadata)


def test_action_recognition_capabilities_and_cosmos_ignored_options(monkeypatch):
    from inference_models.models.cosmos3.cosmos3_action_recognition import (
        Cosmos3EdgeActionRecognition,
    )

    assert VJepaActionRecognition.supports_confidence
    assert VJepaActionRecognition.supports_observed_duration
    cosmos = Cosmos3EdgeActionRecognition.__new__(Cosmos3EdgeActionRecognition)
    cosmos._fine_tune_prefix_allowed_tokens_fn = object()
    calls = []
    monkeypatch.setattr(
        cosmos, "_infer_fine_tuned", lambda **kwargs: calls.append(kwargs) or []
    )

    cosmos.infer(
        frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
        fps=4,
        confidence=0.9,
        duration_seconds=0.2,
    )

    assert not cosmos.supports_confidence
    assert not cosmos.supports_observed_duration
    assert "confidence" not in calls[0]
    assert "duration_seconds" not in calls[0]


@pytest.mark.parametrize("sample_fps", [6.0, 7.0, 30.0])
def test_planned_fractional_duration_is_accepted_and_clipped(sample_fps):
    metadata = config()
    metadata["network_input"]["fps"] = sample_fps
    network = SimpleNamespace(
        encoder=lambda inputs: inputs,
        head=lambda inputs: (
            torch.ones((1, 4, 2)),
            torch.tensor([[[[0.0, 5.0], [0.0, 5.0]]] * 4]),
        ),
    )
    model = VJepaActionRecognition(network, metadata, ["a", "b"], torch.device("cpu"))
    window = plan_windows(120, 30.0, model.video_sampling)[0]
    frames = [np.zeros((384, 384, 3), dtype=np.uint8)] * 4

    predictions = model.infer(
        frames, fps=sample_fps, duration_seconds=window.duration_seconds
    )

    assert predictions
    assert all(prediction.end_frame_idx <= 4 for prediction in predictions)
    with pytest.raises(ModelInputError, match="window duration"):
        model.infer(frames, duration_seconds=4 / sample_fps + 0.000002)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"class_names": ["not-a-class"]}, "Unknown V-JEPA class filter"),
        ({"fps": 30.0}, "input FPS"),
        ({"confidence": 2.0}, "Confidence"),
        ({"duration_seconds": 2.0}, "window duration"),
        ({"frames": [np.zeros((8, 8, 3), dtype=np.float32)]}, "RGB uint8"),
        ({"frames": [torch.zeros((8, 8), dtype=torch.uint8)]}, "RGB uint8"),
    ],
)
def test_request_validation_raises_model_input_errors(kwargs, message):
    model = VJepaActionRecognition(None, config(), ["a", "b"], torch.device("cpu"))

    with pytest.raises(ModelInputError, match=message):
        model.infer(**{"frames": [np.zeros((384, 384, 3), dtype=np.uint8)], **kwargs})


@pytest.mark.parametrize("side", [256, 384, 512])
def test_infer_thresholds_candidates_and_masks_padded_queries(side):
    captured = []

    def encoder(inputs):
        captured.append(inputs)
        return inputs

    def head(_features):
        logits = torch.tensor([[[2.0, -2.0], [2.0, -2.0], [20.0, 20.0], [20.0, 20.0]]])
        intervals = torch.tensor([[[[0.0, 2.0], [0.0, 2.0]]] * 4])
        return logits, intervals

    metadata = config()
    metadata["network_input"]["height"] = side
    metadata["network_input"]["width"] = side
    metadata["encoder"]["arguments"]["img_size"] = [side, side]
    validate_config(metadata)
    model = VJepaActionRecognition(
        SimpleNamespace(encoder=encoder, head=head),
        metadata,
        ["a", "b"],
        torch.device("cpu"),
    )
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    frame[:, :10, 0] = 255
    frame[:, -10:, 1] = 255
    predictions = model.infer([frame, frame], fps=4)
    direct_inputs = captured[0].clone()
    prepared = model.frame_storage_transform(frame)
    assert prepared.shape == (side, side, 3)
    assert prepared.dtype == np.uint8
    assert model.frame_storage_transform(prepared) is prepared
    tensor_frame = torch.from_numpy(frame).permute(2, 0, 1)
    np.testing.assert_array_equal(model.frame_storage_transform(tensor_frame), prepared)
    original_transform = v2.Compose(
        [
            v2.Resize((side, side), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(*metadata["network_input"]["normalization"]),
        ]
    )
    assert torch.equal(
        direct_inputs[0, :, 0], original_transform(Image.fromarray(frame))
    )
    assert model.infer([prepared, prepared], fps=4) == predictions
    assert torch.equal(captured[-1], direct_inputs)
    assert len(predictions) == 2
    assert all(row.class_name == "a" and row.end_exclusive for row in predictions)
    assert len(model.infer([frame, frame], confidence=0)) == 4
    assert model.infer([frame, frame], confidence=1) == []
    assert captured[0].shape == (1, 3, 4, side, side)
    assert captured[0][0, 0, 0, side // 2, 0] > 0
    assert captured[0][0, 1, 0, side // 2, -1] > 0
    assert torch.equal(captured[0][:, :, 1], captured[0][:, :, 3])

    model.recommended_parameters = RecommendedParameters(
        confidence=0.9, per_class_confidence={"b": 0.1}
    )
    assert model.infer([frame, frame]) == predictions
    assert model.infer([frame, frame], confidence="default") == predictions
    assert model.infer([frame, frame], confidence=0.5) == predictions
    recommended = model.infer([frame, frame], confidence="best")
    assert len(recommended) == 2
    assert all(row.class_name == "b" for row in recommended)

    model.recommended_parameters = RecommendedParameters(confidence=0.9)
    assert model.infer([frame, frame], confidence="best") == []
    model.recommended_parameters = RecommendedParameters(confidence=0.0)
    assert len(model.infer([frame, frame], confidence="best")) == 4
    model.recommended_parameters = None
    assert model.infer([frame, frame], confidence="best") == predictions


def test_contract_rejects_mismatched_or_unaligned_frame_side():
    for height, width in [(256, 384), (255, 255), (48, 48), (384, 384.0)]:
        metadata = config()
        metadata["network_input"]["height"] = height
        metadata["network_input"]["width"] = width
        metadata["encoder"]["arguments"]["img_size"] = [height, width]
        with pytest.raises(ValueError, match="square input side"):
            validate_config(metadata)

    metadata = config()
    metadata["network_input"]["height"] = 256
    metadata["network_input"]["width"] = 256
    with pytest.raises(ValueError, match="encoder arguments"):
        validate_config(metadata)
