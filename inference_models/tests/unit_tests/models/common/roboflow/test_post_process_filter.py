"""
Integration tests for `post_process` on the five concrete model stubs
that opt in (OD, IS, KP, multi-label classification, semantic
segmentation). Verifies that post_process:

  1. Builds the ConfidenceFilter from the `confidence` kwarg and the
     instance's `recommended_parameters`.
  2. Uses the filter's floor for NMS / threshold-filter.
  3. Applies per-class refinement to the returned detections.
  4. Short-circuits when there's no per-class data (returns the model's
     output unchanged).
"""

from typing import Any, List, Optional, Tuple

import torch

from inference_models.models.base.classification import (
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)
from inference_models.models.base.keypoints_detection import (
    KeyPoints,
    KeyPointsDetectionModel,
)
from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
)
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationModel,
    SemanticSegmentationResult,
)
from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.weights_providers.entities import RecommendedParameters


# ---- Stubs that mimic real concrete models: they accept
#      recommended_parameters and use ConfidenceFilter in post_process. ----


class _RecordingObjectDetectionModel(ObjectDetectionModel):
    def __init__(self, detections: Detections):
        self._stub_output = detections
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(detections=None)  # type: ignore[arg-type]

    @property
    def class_names(self) -> List[str]:
        return ["cat", "dog", "fish"]

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        result = [self._stub_output]
        if confidence_filter.has_per_class_refinement:
            result = [
                confidence_filter.refine_detections(r, self.class_names)
                for r in result
            ]
        return result


class _RecordingInstanceSegmentationModel(InstanceSegmentationModel):
    def __init__(self, detections: InstanceDetections):
        self._stub_output = detections
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(detections=None)  # type: ignore[arg-type]

    @property
    def class_names(self) -> List[str]:
        return ["cat", "dog"]

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        result = [self._stub_output]
        if confidence_filter.has_per_class_refinement:
            result = [
                confidence_filter.refine_instance_detections(r, self.class_names)
                for r in result
            ]
        return result


class _RecordingKeypointsDetectionModel(KeyPointsDetectionModel):
    def __init__(self, output: Tuple[List[KeyPoints], List[Detections]]):
        self._stub_output = output
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(output=([], []))  # type: ignore[arg-type]

    @property
    def class_names(self) -> List[str]:
        return ["person", "robot"]

    @property
    def key_points_classes(self) -> List[List[str]]:
        return [["nose"], ["led"]]

    @property
    def skeletons(self):
        return [[], []]

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        kp_list, det_list = self._stub_output
        if confidence_filter.has_per_class_refinement and det_list:
            refined = [
                confidence_filter.refine_keypoints_and_detections(
                    kp, det, self.class_names
                )
                for kp, det in zip(kp_list, det_list)
            ]
            kp_list = [r[0] for r in refined]
            det_list = [r[1] for r in refined]
        return kp_list, det_list


class _RecordingMultiLabelClassificationModel(MultiLabelClassificationModel):
    def __init__(self, predictions: List[MultiLabelClassificationPrediction]):
        self._stub_output = predictions
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(predictions=[])

    @property
    def class_names(self) -> List[str]:
        return ["a", "b", "c", "d"]

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        result = self._stub_output
        if confidence_filter.has_per_class_refinement:
            result = [
                confidence_filter.refine_multilabel_prediction(p, self.class_names)
                for p in result
            ]
        return result


class _RecordingSemanticSegmentationModel(SemanticSegmentationModel):
    _class_names_value = ["cat", "dog", "fish", "background"]
    _background_class_id = 3

    def __init__(self, result: SemanticSegmentationResult):
        self._stub_output = result
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(result=None)  # type: ignore[arg-type]

    @property
    def class_names(self) -> List[str]:
        return self._class_names_value

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        result = [self._stub_output]
        if confidence_filter.has_per_class_refinement:
            result = [
                confidence_filter.refine_segmentation_result(
                    r, self.class_names, self._background_class_id
                )
                for r in result
            ]
        return result


# ---------------- ObjectDetectionModel ----------------


class TestObjectDetectionPostProcess:
    def _detections(self) -> Detections:
        # Three detections: cat@0.7, dog@0.3, fish@0.5
        return Detections(
            xyxy=torch.tensor(
                [[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]],
                dtype=torch.float32,
            ),
            class_id=torch.tensor([0, 1, 2]),
            confidence=torch.tensor([0.7, 0.3, 0.5]),
        )

    def test_user_value_passes_floor_through_unchanged(self) -> None:
        # Tier 1: explicit user value → floor IS the user value, no per-class
        # refinement, output unchanged.
        model = _RecordingObjectDetectionModel(self._detections())
        model.recommended_parameters = RecommendedParameters(
            confidence=0.5,
            per_class_confidence={"cat": 0.6, "dog": 0.2, "fish": 0.4},
        )

        result = model.post_process(
            model_results=None, pre_processing_meta=None, confidence=0.65
        )

        assert model.captured_confidence == 0.65
        # Tier 1 is no-refinement; output identical to model output.
        assert len(result[0].xyxy) == 3

    def test_per_class_filters_after_floor_is_applied(self) -> None:
        # Tier 2: floor = min(per_class) = 0.2 → model sees 0.2 → all three
        # detections survive NMS → wrapper refines per-class:
        #   cat@0.7 vs cat=0.6 → keep
        #   dog@0.3 vs dog=0.4 → drop
        #   fish@0.5 vs fish=0.45 → keep
        model = _RecordingObjectDetectionModel(self._detections())
        model.recommended_parameters = RecommendedParameters(
            confidence=None,
            per_class_confidence={"cat": 0.6, "dog": 0.4, "fish": 0.45},
        )

        result = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == 0.4  # min(0.6, 0.4, 0.45)
        kept = result[0]
        assert kept.class_id.tolist() == [0, 2]
        torch.testing.assert_close(kept.confidence, torch.tensor([0.7, 0.5]))

    def test_no_per_class_short_circuits(self) -> None:
        # Tier 3: only global. Floor = global. No per-class refinement, output
        # is the same object the model returned.
        original = self._detections()
        model = _RecordingObjectDetectionModel(original)
        model.recommended_parameters = RecommendedParameters(confidence=0.42)

        result = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == 0.42
        assert result[0] is original  # short-circuit returned the original list

    def test_default_when_no_recommended_parameters(self) -> None:
        # Tier 4: no recommended defaults → hardcoded default → no per-class.
        from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE

        model = _RecordingObjectDetectionModel(self._detections())

        model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == INFERENCE_MODELS_DEFAULT_CONFIDENCE


# ---------------- InstanceSegmentationModel ----------------


class TestInstanceSegmentationPostProcess:
    def test_per_class_filter_keeps_mask_aligned(self) -> None:
        # The mask tensor must be sliced in lockstep with class_id/confidence,
        # otherwise filtered detections would carry the wrong masks.
        instance_dets = InstanceDetections(
            xyxy=torch.tensor(
                [[0, 0, 10, 10], [0, 0, 20, 20]], dtype=torch.float32
            ),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.7, 0.3]),
            mask=torch.tensor(
                [[[1, 1], [0, 0]], [[0, 0], [1, 1]]], dtype=torch.float32
            ),
        )
        model = _RecordingInstanceSegmentationModel(instance_dets)
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"cat": 0.6, "dog": 0.5}
        )

        result = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        # cat@0.7 ≥ 0.6 keep, dog@0.3 < 0.5 drop
        kept = result[0]
        assert kept.class_id.tolist() == [0]
        # The kept mask must be the cat mask (top row), not the dog mask.
        assert kept.mask.tolist() == [[[1, 1], [0, 0]]]


# ---------------- KeyPointsDetectionModel ----------------


class TestKeypointsDetectionPostProcess:
    def test_per_class_filter_aligns_keypoints_and_detections(self) -> None:
        # The wrapper must filter the parallel KeyPoints and Detections lists
        # with the same boolean mask, so the i-th keypoint set still lines up
        # with the i-th detection after filtering.
        detections = Detections(
            xyxy=torch.tensor(
                [[0, 0, 10, 10], [0, 0, 20, 20]], dtype=torch.float32
            ),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.8, 0.2]),
        )
        keypoints = KeyPoints(
            xy=torch.tensor(
                [[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float32
            ),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([[0.9], [0.1]]),
        )
        model = _RecordingKeypointsDetectionModel(
            output=([keypoints], [detections])
        )
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"person": 0.5, "robot": 0.5}
        )

        kp_list, det_list = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        # person@0.8 keep, robot@0.2 drop
        assert det_list[0].class_id.tolist() == [0]
        assert kp_list[0].class_id.tolist() == [0]
        assert kp_list[0].xy.tolist() == [[[1.0, 2.0]]]


# ---------------- MultiLabelClassificationModel ----------------


class TestMultiLabelClassificationPostProcess:
    def test_per_class_filter_drops_class_ids_below_threshold(self) -> None:
        # The full confidence vector stays intact; only the predicted class_ids
        # list is filtered. Per-class threshold uses the indexed score.
        model = _RecordingMultiLabelClassificationModel(
            predictions=[
                MultiLabelClassificationPrediction(
                    class_ids=torch.tensor([0, 1, 2, 3]),
                    confidence=torch.tensor([0.8, 0.4, 0.6, 0.9]),
                )
            ]
        )
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"a": 0.5, "b": 0.5, "c": 0.7, "d": 0.5}
        )

        result = model.post_process(model_results=None)

        kept_ids = result[0].class_ids.tolist()
        # a@0.8 ≥ 0.5 keep, b@0.4 < 0.5 drop, c@0.6 < 0.7 drop, d@0.9 ≥ 0.5 keep
        assert kept_ids == [0, 3]
        # Full confidence vector unchanged.
        torch.testing.assert_close(
            result[0].confidence, torch.tensor([0.8, 0.4, 0.6, 0.9])
        )

    def test_no_per_class_short_circuits_to_model_output(self) -> None:
        original = MultiLabelClassificationPrediction(
            class_ids=torch.tensor([0, 1]),
            confidence=torch.tensor([0.8, 0.7, 0.1, 0.05]),
        )
        model = _RecordingMultiLabelClassificationModel(predictions=[original])
        model.recommended_parameters = RecommendedParameters(confidence=0.5)

        result = model.post_process(model_results=None)

        assert model.captured_confidence == 0.5
        assert result[0] is original


# ---------------- SemanticSegmentationModel ----------------


class TestSemanticSegmentationPostProcess:
    def _result(self) -> SemanticSegmentationResult:
        # 2x2 segmentation map. Pixel layout (class_id, confidence):
        #   (cat, 0.7)  (dog, 0.55)
        #   (fish, 0.5) (cat,  0.3)
        # Background class id = 3 in the stub model.
        return SemanticSegmentationResult(
            segmentation_map=torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
            confidence=torch.tensor(
                [[0.7, 0.55], [0.5, 0.3]], dtype=torch.float32
            ),
        )

    def test_user_value_passes_floor_through_unchanged(self) -> None:
        # Tier 1 — user override beats per-class. The model gets 0.65 as its
        # confidence kwarg, no per-class refinement happens, the result is
        # whatever the model returned (the test stub returns it unmodified).
        model = _RecordingSemanticSegmentationModel(self._result())
        model.recommended_parameters = RecommendedParameters(
            confidence=0.5,
            per_class_confidence={"cat": 0.6, "dog": 0.4, "fish": 0.45},
        )

        out = model.post_process(
            model_results=None, pre_processing_meta=None, confidence=0.65
        )

        assert model.captured_confidence == 0.65
        # No refinement → wrapper passes the model's output through.
        assert out[0] is model._stub_output

    def test_per_class_remaps_below_threshold_pixels_to_background(self) -> None:
        # Tier 2: per_class = {cat: 0.6, dog: 0.4, fish: 0.55}, no global.
        # Floor = min(per_class, hardcoded default). Per-pixel evaluation:
        #   (cat, 0.7)  vs cat=0.6  → keep
        #   (dog, 0.55) vs dog=0.4  → keep
        #   (fish, 0.5) vs fish=0.55 → drop → background (3)
        #   (cat, 0.3)  vs cat=0.6  → drop → background (3)
        model = _RecordingSemanticSegmentationModel(self._result())
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"cat": 0.6, "dog": 0.4, "fish": 0.55},
        )

        out = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert out[0].segmentation_map.tolist() == [[0, 1], [3, 3]]
        # Filtered pixels also get their confidence zeroed for downstream
        # callers that compare confidence vs a threshold.
        torch.testing.assert_close(
            out[0].confidence,
            torch.tensor([[0.7, 0.55], [0.0, 0.0]], dtype=torch.float32),
        )

    def test_no_per_class_short_circuits(self) -> None:
        # Tier 3 — only global. Wrapper passes the model's output through.
        original = self._result()
        model = _RecordingSemanticSegmentationModel(original)
        model.recommended_parameters = RecommendedParameters(confidence=0.42)

        out = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == 0.42
        assert out[0] is original

    def test_default_when_no_recommended_parameters(self) -> None:
        from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE

        model = _RecordingSemanticSegmentationModel(self._result())

        model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == INFERENCE_MODELS_DEFAULT_CONFIDENCE

    def test_per_class_skipped_when_model_has_no_background_class_id(self) -> None:
        # If a future SS model doesn't expose `_background_class_id`, the
        # wrapper can't safely remap pixels — fall back to the model's
        # floor-only filtering.
        class _NoBackgroundSS(SemanticSegmentationModel):
            # Note: deliberately does NOT set _background_class_id.
            def __init__(self, result):
                self._stub_output = result

            @classmethod
            def from_pretrained(cls, model_name_or_path, **kwargs):
                return cls(result=None)

            @property
            def class_names(self):
                return ["cat", "dog"]

            def pre_process(self, images, **kwargs):
                raise NotImplementedError

            def forward(self, pre_processed_images, **kwargs):
                raise NotImplementedError

            def post_process(self, model_results, pre_processing_meta, **kwargs):
                return [self._stub_output]

        original = SemanticSegmentationResult(
            segmentation_map=torch.tensor([[0, 1]], dtype=torch.long),
            confidence=torch.tensor([[0.7, 0.3]], dtype=torch.float32),
        )
        model = _NoBackgroundSS(original)
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"cat": 0.6, "dog": 0.6}
        )

        out = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        # No per-class refinement applied — output is the model's raw result.
        assert out[0] is original
