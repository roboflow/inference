"""
Tests for the `recommended_parameters` instance attribute on concrete models.
Concrete models accept `recommended_parameters` via `from_pretrained` kwargs
and store it as an instance attribute (defaulting to None). The auto-loader
passes the value at construction time; ABCs do NOT declare the attribute.
"""

from typing import List

import pytest

from inference_models.models.base.classification import (
    ClassificationModel,
    MultiLabelClassificationModel,
)
from inference_models.models.base.instance_segmentation import (
    InstanceSegmentationModel,
)
from inference_models.models.base.keypoints_detection import KeyPointsDetectionModel
from inference_models.models.base.object_detection import ObjectDetectionModel
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationModel,
)
from inference_models.weights_providers.entities import RecommendedParameters


# --- Minimal concrete stubs that mirror the real from_pretrained pattern ---


class _StubOD(ObjectDetectionModel):
    def __init__(self, recommended_parameters=None):
        self.recommended_parameters = recommended_parameters

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(recommended_parameters=kwargs.get("recommended_parameters"))

    @property
    def class_names(self) -> List[str]:
        return []

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        raise NotImplementedError


class _StubIS(InstanceSegmentationModel):
    def __init__(self, recommended_parameters=None):
        self.recommended_parameters = recommended_parameters

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(recommended_parameters=kwargs.get("recommended_parameters"))

    @property
    def class_names(self) -> List[str]:
        return []

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        raise NotImplementedError


class _StubKP(KeyPointsDetectionModel):
    def __init__(self, recommended_parameters=None):
        self.recommended_parameters = recommended_parameters

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(recommended_parameters=kwargs.get("recommended_parameters"))

    @property
    def class_names(self) -> List[str]:
        return []

    @property
    def key_points_classes(self):
        return []

    @property
    def skeletons(self):
        return []

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        raise NotImplementedError


class _StubML(MultiLabelClassificationModel):
    def __init__(self, recommended_parameters=None):
        self.recommended_parameters = recommended_parameters

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(recommended_parameters=kwargs.get("recommended_parameters"))

    @property
    def class_names(self) -> List[str]:
        return []

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, **kwargs):
        raise NotImplementedError


class _StubSS(SemanticSegmentationModel):
    def __init__(self, recommended_parameters=None):
        self.recommended_parameters = recommended_parameters

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(recommended_parameters=kwargs.get("recommended_parameters"))

    @property
    def class_names(self) -> List[str]:
        return []

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        raise NotImplementedError


CONCRETE_STUBS = [_StubOD, _StubIS, _StubKP, _StubML, _StubSS]


class TestConcreteModelDefaultsToNone:
    @pytest.mark.parametrize("stub_cls", CONCRETE_STUBS)
    def test_concrete_instance_defaults_to_none(self, stub_cls) -> None:
        # When constructed without recommended_parameters, the instance
        # attribute defaults to None so the confidence filter falls back
        # to the hardcoded default.
        instance = stub_cls()
        assert instance.recommended_parameters is None


class TestFromPretrainedPropagation:
    @pytest.mark.parametrize("stub_cls", CONCRETE_STUBS)
    def test_from_pretrained_propagates_recommended_parameters(self, stub_cls) -> None:
        rp = RecommendedParameters(confidence=0.42)
        instance = stub_cls.from_pretrained("test", recommended_parameters=rp)
        assert instance.recommended_parameters is rp


class TestAbcsDoNotDeclareAttribute:
    def test_abc_object_detection_has_no_recommended_parameters(self) -> None:
        assert not hasattr(ObjectDetectionModel, "recommended_parameters")

    def test_abc_instance_segmentation_has_no_recommended_parameters(self) -> None:
        assert not hasattr(InstanceSegmentationModel, "recommended_parameters")

    def test_abc_keypoints_detection_has_no_recommended_parameters(self) -> None:
        assert not hasattr(KeyPointsDetectionModel, "recommended_parameters")

    def test_abc_multi_label_classification_has_no_recommended_parameters(self) -> None:
        assert not hasattr(MultiLabelClassificationModel, "recommended_parameters")

    def test_abc_semantic_segmentation_has_no_recommended_parameters(self) -> None:
        assert not hasattr(SemanticSegmentationModel, "recommended_parameters")

    def test_single_label_classification_has_no_recommended_parameters(self) -> None:
        # Single-label deliberately opts out — top-1 always wins regardless
        # of confidence, so per-class refinement isn't a meaningful semantic.
        assert not hasattr(ClassificationModel, "recommended_parameters")


class TestInstanceAssignment:
    """Instance assignment must NOT leak across instances — the auto-loader
    sets it on each instance, and one model getting recommendedParameters
    must not pollute another."""

    def test_instance_assignment_does_not_leak_across_instances(self) -> None:
        a = _StubOD()
        b = _StubOD()
        # The auto-loader's pattern: direct instance assignment.
        a.recommended_parameters = RecommendedParameters(confidence=0.42)

        assert a.recommended_parameters.confidence == 0.42
        # b still has its own None default.
        assert b.recommended_parameters is None
