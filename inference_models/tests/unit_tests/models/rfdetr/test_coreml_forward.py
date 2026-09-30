from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.coreml import CoreMLModel, CoreMLModelSignature
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    NetworkInputDefinition,
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.rfdetr.common import post_process_object_detection_results
from inference_models.models.rfdetr.coreml_forward import (
    run_rfdetr_coreml,
    selections_to_logits,
    to_package_image,
)
from inference_models.models.rfdetr.post_processor import select_topk_predictions

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FakeModel:
    def __init__(self, outputs: Dict[str, Any]):
        self.outputs = outputs
        self.feeds: List[Mapping[str, Any]] = []

    def predict(self, feed: Mapping[str, Any]) -> Dict[str, Any]:
        self.feeds.append(feed)
        return self.outputs


def _network_input(color_mode: ColorMode = ColorMode.RGB) -> NetworkInputDefinition:
    return NetworkInputDefinition.model_validate(
        {
            "training_input_size": {"height": 4, "width": 6},
            "dynamic_spatial_size_supported": False,
            "color_mode": color_mode.value,
            "resize_mode": "stretch",
            "input_channels": 3,
            "scaling_factor": 255,
            "normalization": [MEAN, STD],
        }
    )


def _normalize(image_hwc: np.ndarray) -> torch.Tensor:
    pixels = torch.from_numpy(image_hwc).permute(2, 0, 1).float() / 255
    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]
    return (pixels - mean) / std


def _package_top_k(logits: torch.Tensor, bboxes: torch.Tensor, k: int):
    """The image-input package's in-graph selection: sigmoid, flat top-k over (query, class)."""
    scores = logits.sigmoid()
    flat_scores, flat_indices = scores.reshape(-1).topk(k)
    queries = flat_indices // logits.shape[1]
    return flat_scores, flat_indices % logits.shape[1], bboxes[queries]


def test_selections_to_logits_round_trips_through_shared_top_k() -> None:
    scores = torch.tensor([0.9, 0.9, 0.2, 0.0, 1.0])
    labels = torch.tensor([3, 1, 3, 0, 2])

    logits = selections_to_logits(scores=scores, labels=labels, num_logit_classes=5)
    selected_scores, selected_labels, _, rows = select_topk_predictions(
        logits_sigmoid=logits.sigmoid(), bboxes_cxcywh=torch.zeros(5, 4)
    )

    assert logits.shape == (5, 5)
    assert sorted(rows.tolist()) == [0, 1, 2, 3, 4]
    assert torch.equal(selected_labels, labels[rows])
    assert torch.allclose(selected_scores, scores[rows], atol=1e-6)
    assert torch.all(selected_scores > 0)


def test_selections_to_logits_widens_for_labels_past_declared_classes() -> None:
    logits = selections_to_logits(
        scores=torch.tensor([0.5]), labels=torch.tensor([7]), num_logit_classes=3
    )

    assert logits.shape == (1, 8)
    assert logits.argmax().item() == 7


def test_package_selections_give_the_same_detections_as_raw_outputs() -> None:
    generator = torch.Generator().manual_seed(0)
    num_queries, num_logit_classes = 30, 6
    logits = torch.randn(num_queries, num_logit_classes, generator=generator) * 3
    bboxes = torch.rand(num_queries, 4, generator=generator) * 0.5 + 0.25
    scores, labels, selected_bboxes = _package_top_k(logits, bboxes, k=num_queries)
    meta = [
        PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=ImageDimensions(height=480, width=640),
            size_after_pre_processing=ImageDimensions(height=384, width=384),
            inference_size=ImageDimensions(height=384, width=384),
            scale_width=384 / 640,
            scale_height=384 / 480,
            static_crop_offset=StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=640, crop_height=480
            ),
        )
    ]

    def detect(raw_bboxes: torch.Tensor, raw_logits: torch.Tensor):
        return post_process_object_detection_results(
            bboxes=raw_bboxes[None],
            logits=raw_logits[None],
            pre_processing_meta=meta,
            threshold=0.3,
            num_classes=num_logit_classes - 1,
            classes_re_mapping=None,
            device=torch.device("cpu"),
        )[0]

    reference = detect(bboxes, logits)
    from_package = detect(
        selected_bboxes,
        selections_to_logits(
            scores=scores, labels=labels, num_logit_classes=num_logit_classes
        ),
    )

    assert len(reference.class_id) > 0
    assert torch.equal(from_package.class_id, reference.class_id)
    assert torch.equal(from_package.xyxy, reference.xyxy)
    assert torch.allclose(from_package.confidence, reference.confidence, atol=1e-6)


def test_to_package_image_recovers_the_original_pixels() -> None:
    image = np.random.default_rng(0).integers(0, 256, (4, 6, 3), dtype=np.uint8)

    recovered = to_package_image(
        image=_normalize(image), network_input=_network_input()
    )

    assert recovered.mode == "RGB"
    assert np.array_equal(np.asarray(recovered), image)


def test_to_package_image_converts_bgr_network_input_to_rgb() -> None:
    image = np.random.default_rng(1).integers(0, 256, (4, 6, 3), dtype=np.uint8)

    recovered = to_package_image(
        image=_normalize(image[:, :, ::-1].copy()),
        network_input=_network_input(ColorMode.BGR),
    )

    assert np.array_equal(np.asarray(recovered), image)


def test_run_rfdetr_coreml_feeds_tensor_contract_and_returns_outputs_by_position() -> (
    None
):
    fake = FakeModel(
        {
            "dets": np.ones((1, 300, 4), dtype=np.float16),
            "labels": np.full((1, 300, 91), 2.0, dtype=np.float32),
            "masks": np.zeros((1, 300, 78, 78), dtype=np.float32),
        }
    )
    model = CoreMLModel(
        model=fake,
        signature=CoreMLModelSignature(
            input_name="input",
            image_input=False,
            input_height=312,
            input_width=312,
            output_names=("dets", "labels", "masks"),
        ),
    )
    images = torch.zeros(2, 3, 312, 312)

    bboxes, logits = run_rfdetr_coreml(
        model=model,
        pre_processed_images=images,
        network_input=_network_input(),
        num_logit_classes=91,
        with_masks=False,
    )

    assert len(fake.feeds) == 2
    assert fake.feeds[0]["input"].shape == (1, 3, 312, 312)
    assert fake.feeds[0]["input"].dtype == np.float32
    assert bboxes.shape == (2, 300, 4) and bboxes.dtype == torch.float32
    assert logits.shape == (2, 300, 91)


def test_run_rfdetr_coreml_expands_image_contract_selections() -> None:
    fake = FakeModel(
        {
            "boxes": np.array([[[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]]),
            "scores": np.array([[0.8, 0.4]], dtype=np.float16),
            "labels": np.array([[4.0, 1.0]]),
            "masks": np.ones((1, 2, 8, 8)),
        }
    )
    model = CoreMLModel(
        model=fake,
        signature=CoreMLModelSignature(
            input_name="image",
            image_input=True,
            input_height=4,
            input_width=6,
            output_names=("boxes", "scores", "labels", "masks"),
        ),
    )

    bboxes, logits, masks = run_rfdetr_coreml(
        model=model,
        pre_processed_images=torch.zeros(1, 3, 4, 6),
        network_input=_network_input(),
        num_logit_classes=6,
        with_masks=True,
    )

    assert fake.feeds[0]["image"].size == (6, 4)
    assert bboxes.shape == (1, 2, 4)
    assert logits.shape == (1, 2, 6)
    assert logits[0].argmax(dim=1).tolist() == [4, 1]
    assert torch.allclose(
        logits[0].sigmoid().max(dim=1).values, torch.tensor([0.8, 0.4]), atol=1e-3
    )
    assert masks.shape == (1, 2, 8, 8)
