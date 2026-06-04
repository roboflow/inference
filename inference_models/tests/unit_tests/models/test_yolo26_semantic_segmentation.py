from unittest.mock import MagicMock

import torch


def test_post_process_semantic_segmentation_logits_argmax_collapses_to_winning_class():
    """Logits (B, K, H, W) with class 2 dominant collapse to a (H, W) seg_map
    of class 2 across the board after softmax + argmax."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_semantic_segmentation_logits,
    )

    h, w, num_classes = 8, 8, 3
    logits = torch.zeros(1, num_classes, h, w)
    logits[0, 2] = 5.0  # class 2 dominates everywhere

    meta = [
        MagicMock(
            inference_size=MagicMock(height=h, width=w),
            pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
            size_after_pre_processing=MagicMock(height=h, width=w),
            original_size=MagicMock(height=h, width=w),
            static_crop_offset=MagicMock(offset_x=0, offset_y=0),
        )
    ]

    results = post_process_semantic_segmentation_logits(
        model_results=logits,
        pre_processing_meta=meta,
        class_names=["a", "b", "c"],
        background_class_id=-1,
        device=torch.device("cpu"),
        confidence=0.0,
        recommended_parameters=None,
        default_confidence=0.0,
    )

    assert len(results) == 1
    assert results[0].segmentation_map.shape == (h, w)
    assert torch.all(results[0].segmentation_map == 2)
    assert results[0].confidence.shape == (h, w)
    # softmax of [0, 0, 5] for class 2 dominant = e^5 / (2 + e^5) ~= 0.987
    assert torch.all(results[0].confidence > 0.98)


def test_post_process_semantic_segmentation_logits_sub_threshold_collapses_to_background():
    """Pixels whose softmax confidence is below the threshold collapse to background_class_id."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_semantic_segmentation_logits,
    )

    h, w, num_classes = 4, 4, 3
    # Uniform logits → softmax = 1/3 everywhere; with threshold 0.5 all pixels go below
    logits = torch.zeros(1, num_classes, h, w)

    meta = [
        MagicMock(
            inference_size=MagicMock(height=h, width=w),
            pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
            size_after_pre_processing=MagicMock(height=h, width=w),
            original_size=MagicMock(height=h, width=w),
            static_crop_offset=MagicMock(offset_x=0, offset_y=0),
        )
    ]

    results = post_process_semantic_segmentation_logits(
        model_results=logits,
        pre_processing_meta=meta,
        class_names=["a", "b", "c"],
        background_class_id=-1,
        device=torch.device("cpu"),
        confidence=0.5,
        recommended_parameters=None,
        default_confidence=0.5,
    )

    assert torch.all(results[0].segmentation_map == -1)
    assert torch.all(results[0].confidence == 0.0)


def test_post_process_semantic_segmentation_logits_shifts_when_class_names_prepends_background():
    """When class_names has one extra entry (background at index 0) compared to
    the number of logit channels, argmax outputs are shifted by +1 so class 0
    stays reserved for sub-threshold pixels."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_semantic_segmentation_logits,
    )

    h, w, num_channels = 8, 8, 3
    logits = torch.zeros(1, num_channels, h, w)
    logits[0, 2] = 5.0  # channel 2 dominates → expect class_id 3 after shift

    meta = [
        MagicMock(
            inference_size=MagicMock(height=h, width=w),
            pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
            size_after_pre_processing=MagicMock(height=h, width=w),
            original_size=MagicMock(height=h, width=w),
            static_crop_offset=MagicMock(offset_x=0, offset_y=0),
        )
    ]

    results = post_process_semantic_segmentation_logits(
        model_results=logits,
        pre_processing_meta=meta,
        class_names=["background", "a", "b", "c"],
        background_class_id=0,
        device=torch.device("cpu"),
        confidence=0.0,
        recommended_parameters=None,
        default_confidence=0.0,
    )

    assert torch.all(results[0].segmentation_map == 3)


def _binary_meta(h, w):
    return [
        MagicMock(
            inference_size=MagicMock(height=h, width=w),
            pad_top=0,
            pad_bottom=0,
            pad_left=0,
            pad_right=0,
            size_after_pre_processing=MagicMock(height=h, width=w),
            original_size=MagicMock(height=h, width=w),
            static_crop_offset=MagicMock(offset_x=0, offset_y=0),
        )
    ]


def test_post_process_semantic_segmentation_logits_binary_single_channel_foreground():
    """A single-channel (Ultralytics nc==1) output uses the sigmoid foreground
    probability — high-positive logits map to the lone foreground class, not the
    degenerate softmax-over-one-channel."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_semantic_segmentation_logits,
    )

    h, w = 6, 6
    logits = torch.full((1, 1, h, w), 5.0)  # sigmoid ~ 0.993 -> foreground

    results = post_process_semantic_segmentation_logits(
        model_results=logits,
        pre_processing_meta=_binary_meta(h, w),
        class_names=["background", "object"],
        background_class_id=0,
        device=torch.device("cpu"),
        confidence=0.5,
        recommended_parameters=None,
        default_confidence=0.5,
    )

    assert results[0].segmentation_map.shape == (h, w)
    assert torch.all(results[0].segmentation_map == 1)  # lone foreground class id
    assert torch.all(results[0].confidence > 0.99)


def test_post_process_semantic_segmentation_logits_binary_sub_threshold_to_background():
    """Single-channel logits with low sigmoid confidence collapse to background."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_semantic_segmentation_logits,
    )

    h, w = 4, 4
    logits = torch.full((1, 1, h, w), -5.0)  # sigmoid ~ 0.0067 -> below threshold

    results = post_process_semantic_segmentation_logits(
        model_results=logits,
        pre_processing_meta=_binary_meta(h, w),
        class_names=["background", "object"],
        background_class_id=0,
        device=torch.device("cpu"),
        confidence=0.5,
        recommended_parameters=None,
        default_confidence=0.5,
    )

    assert torch.all(results[0].segmentation_map == 0)  # background
    assert torch.all(results[0].confidence == 0.0)


def test_post_process_semantic_segmentation_logits_maps_channels_when_background_not_first():
    """K foreground channels with background NOT at index 0: channel j maps to
    the j-th non-background class id, not a blanket +1."""
    from inference_models.models.common.roboflow.post_processing import (
        post_process_semantic_segmentation_logits,
    )

    h, w, num_channels = 6, 6, 3
    logits = torch.zeros(1, num_channels, h, w)
    logits[0, 0] = 5.0  # channel 0 dominant

    results = post_process_semantic_segmentation_logits(
        model_results=logits,
        pre_processing_meta=_binary_meta(h, w),
        class_names=["a", "background", "b", "c"],
        background_class_id=1,
        device=torch.device("cpu"),
        confidence=0.0,
        recommended_parameters=None,
        default_confidence=0.0,
    )

    assert torch.all(results[0].segmentation_map == 0)


def test_yolo26_semantic_segmentation_registered():
    from inference_models.models.auto_loaders.entities import BackendType
    from inference_models.models.auto_loaders.models_registry import (
        REGISTERED_MODELS,
        SEMANTIC_SEGMENTATION_TASK,
    )

    for backend in (BackendType.ONNX, BackendType.TORCH_SCRIPT, BackendType.TRT):
        assert ("yolo26", SEMANTIC_SEGMENTATION_TASK, backend) in REGISTERED_MODELS, (
            f"Missing yolo26 semantic seg entry for backend {backend}"
        )
