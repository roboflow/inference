import gc
import weakref
from unittest.mock import patch

import numpy as np
import pytest
import torch

from inference_model_manager import configuration as cfg
from inference_model_manager.dispatch import list_actions_for_class
from inference_model_manager.model_manager import ModelManager
from inference_model_manager.pipelines import (
    DISABLED_STAGE,
    PPOCRv6StructuredOCR,
    default_stage_tokens,
    pipeline_model_id,
    pipeline_stage_model_ids,
    resolve_pipeline_request,
    stage_tokens,
)
from inference_model_manager.registry_defaults import lazy_register
from inference_models.errors import ModelNotFoundError
from inference_models.models.base.object_detection import Detections
from inference_models.models.pp_ocrv6.pp_ocrv6_pipeline import PPOCRv6Pipeline


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("pp_ocr/small-small", ("pp-ocrv6-det/small", "pp-ocrv6-rec/small")),
        ("pp_ocr/tiny-medium", ("pp-ocrv6-det/tiny", "pp-ocrv6-rec/medium")),
        ("pp_ocr/small-none", ("pp-ocrv6-det/small", None)),
        ("pp_ocr/none-small", (None, "pp-ocrv6-rec/small")),
        ("pp_ocr/medium", ("pp-ocrv6-det/medium", "pp-ocrv6-rec/medium")),
        ("pp_ocr", ("pp-ocrv6-det/small", "pp-ocrv6-rec/small")),
    ],
)
def test_pp_ocr_ids_expand_to_stage_model_ids(model_id, expected):
    request = resolve_pipeline_request(model_id)
    assert request.family.pipeline_id == "pp-ocrv6"
    assert request.stage_model_ids == expected


@pytest.mark.parametrize(
    "model_id", ["yolov8n-640", "coins/3", "/tmp/local-package", "pp_ocr_v2/1"]
)
def test_non_pipeline_ids_resolve_to_none(model_id):
    assert resolve_pipeline_request(model_id) is None
    assert pipeline_stage_model_ids(model_id) == []


@pytest.mark.parametrize(
    "model_id",
    [
        "pp_ocr/none-none",
        "pp_ocr/a-b-c",
        "pp_ocr/-small",
        "pp_ocr/large-small",
        "pp_ocr/Small",
    ],
)
def test_malformed_or_unknown_pp_ocr_ids_raise_value_error(model_id):
    with pytest.raises(ValueError):
        resolve_pipeline_request(model_id)


def test_stage_tokens_are_the_family_table():
    assert stage_tokens("pp_ocr") == frozenset({"none", "tiny", "small", "medium"})
    assert DISABLED_STAGE in stage_tokens("pp_ocr")


def test_default_stage_tokens_derive_from_library_defaults(monkeypatch):
    from inference_models.model_pipelines.auto_loaders import pipelines_registry

    assert default_stage_tokens("pp_ocr") == ("small", "small")
    monkeypatch.setitem(
        pipelines_registry.DEFAULT_PIPELINES_PARAMETERS,
        "pp-ocrv6",
        ["pp-ocrv6-det/tiny", "pp-ocrv6-rec/medium"],
    )
    assert default_stage_tokens("pp_ocr") == ("tiny", "medium")
    assert resolve_pipeline_request("pp_ocr").stage_model_ids == (
        "pp-ocrv6-det/tiny",
        "pp-ocrv6-rec/medium",
    )


def test_pipeline_model_id_round_trips_through_expansion():
    model_id = pipeline_model_id("pp_ocr", ("tiny", "none"))
    assert model_id == "pp_ocr/tiny-none"
    assert resolve_pipeline_request(model_id).stage_model_ids == (
        "pp-ocrv6-det/tiny",
        None,
    )


def test_stage_model_ids_skip_disabled_stage():
    assert pipeline_stage_model_ids("pp_ocr/none-small") == ["pp-ocrv6-rec/small"]
    assert pipeline_stage_model_ids("pp_ocr/small-small") == [
        "pp-ocrv6-det/small",
        "pp-ocrv6-rec/small",
    ]


@pytest.mark.parametrize(
    "defaults",
    [
        [
            {"model_id_or_path": "pp-ocrv6-det/small", "device": "cpu"},
            "pp-ocrv6-rec/small",
        ],
        ["pp-ocrv6-det/none", "pp-ocrv6-rec/small"],
        ["other-det/small", "pp-ocrv6-rec/small"],
        ["pp-ocrv6-det/large", "pp-ocrv6-rec/small"],
        ["pp-ocrv6-det/small"],
    ],
    ids=["dict", "disabled-token", "wrong-prefix", "unknown-token", "wrong-length"],
)
def test_invalid_library_defaults_are_rejected_on_both_paths(monkeypatch, defaults):
    from inference_models.model_pipelines.auto_loaders import pipelines_registry

    monkeypatch.setitem(
        pipelines_registry.DEFAULT_PIPELINES_PARAMETERS, "pp-ocrv6", defaults
    )
    with pytest.raises(ValueError):
        resolve_pipeline_request("pp_ocr")
    with pytest.raises(ValueError):
        default_stage_tokens("pp_ocr")


def test_missing_library_defaults_are_rejected_on_both_paths(monkeypatch):
    from inference_models.model_pipelines.auto_loaders import pipelines_registry

    monkeypatch.delitem(pipelines_registry.DEFAULT_PIPELINES_PARAMETERS, "pp-ocrv6")
    with pytest.raises(ValueError):
        resolve_pipeline_request("pp_ocr")
    with pytest.raises(ValueError):
        default_stage_tokens("pp_ocr")


def test_bare_family_id_and_entity_defaults_select_the_same_stages():
    bare = resolve_pipeline_request("pp_ocr").stage_model_ids
    via_entity = resolve_pipeline_request(
        pipeline_model_id("pp_ocr", default_stage_tokens("pp_ocr"))
    ).stage_model_ids
    assert bare == via_entity == ("pp-ocrv6-det/small", "pp-ocrv6-rec/small")


LOWER_BOX = [0.0, 10.0, 8.0, 14.0]
UPPER_BOX = [0.0, 0.0, 6.0, 3.0]


class _FakeDetector:
    def __call__(self, images, input_color_format=None, **kwargs):
        return [
            Detections(
                xyxy=torch.tensor([LOWER_BOX, UPPER_BOX]),
                class_id=torch.tensor([0, 0]),
                confidence=torch.tensor([0.8, 0.9]),
                bboxes_metadata=[
                    {"polygon": [[0, 10], [8, 10], [8, 14], [0, 14]]},
                    {"polygon": [[0, 0], [6, 0], [6, 3], [0, 3]]},
                ],
            )
        ]


class _FakeRecognizer:
    def __call__(self, crops, input_color_format=None, **kwargs):
        if isinstance(crops, np.ndarray) and crops.ndim == 3:
            crops = [crops]
        return [f"{crop.shape[0]}x{crop.shape[1]}" for crop in crops]


def _image():
    return np.zeros((20, 20, 3), dtype=np.uint8)


def test_facade_reorders_boxes_and_recognizes_the_matching_crops():
    facade = PPOCRv6StructuredOCR(
        PPOCRv6Pipeline(det_model=_FakeDetector(), rec_model=_FakeRecognizer())
    )
    texts, detections = facade.infer(_image())
    assert texts == ["3x6\n4x8"]
    assert len(detections) == 1 and len(detections[0]) == 2
    assert detections[0].xyxy.tolist() == [UPPER_BOX, LOWER_BOX]
    assert detections[0].confidence.tolist() == pytest.approx([0.9, 0.8])
    assert [m["text"] for m in detections[0].bboxes_metadata] == ["3x6", "4x8"]
    assert detections[0].bboxes_metadata[0]["polygon"][1] == [6, 0]


def test_facade_detect_only_yields_empty_text_per_reordered_box():
    facade = PPOCRv6StructuredOCR(PPOCRv6Pipeline(det_model=_FakeDetector()))
    texts, detections = facade.infer(_image())
    assert texts == [""]
    assert detections[0].xyxy.tolist() == [UPPER_BOX, LOWER_BOX]
    assert [m["text"] for m in detections[0].bboxes_metadata] == ["", ""]


def test_facade_recognition_only_yields_empty_detections():
    facade = PPOCRv6StructuredOCR(PPOCRv6Pipeline(rec_model=_FakeRecognizer()))
    texts, detections = facade.infer([_image()])
    assert texts == ["20x20"]
    assert tuple(detections[0].xyxy.shape) == (0, 4) and len(detections[0]) == 0


def test_facade_registers_structured_ocr_infer_action():
    lazy_register(PPOCRv6StructuredOCR)
    actions = list_actions_for_class(PPOCRv6StructuredOCR)
    assert actions["infer"]["default"] is True
    assert actions["infer"]["response_type"] == "roboflow-structured-ocr-compact-v1"


_FROM_PRETRAINED = "inference_models.models.auto_loaders.core.AutoModel.from_pretrained"


def _stage_for(model_id, **kwargs):
    if model_id.startswith("pp-ocrv6-det/"):
        return _FakeDetector()
    if model_id.startswith("pp-ocrv6-rec/"):
        return _FakeRecognizer()
    raise ModelNotFoundError(f"unexpected stage id {model_id}")


def test_load_pp_ocr_small_small_expands_to_stage_models():
    mm = ModelManager()
    try:
        with patch(_FROM_PRETRAINED, side_effect=_stage_for) as fp:
            mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0)
        assert [c.args[0] for c in fp.call_args_list] == [
            "pp-ocrv6-det/small",
            "pp-ocrv6-rec/small",
        ]
        assert all(c.kwargs["api_key"] == "k" for c in fp.call_args_list)
        assert all(
            c.kwargs["torchscript_state_global_lock"]
            is mm.torchscript_state_global_lock
            for c in fp.call_args_list
        )
        entry = next(
            m for m in mm.stats()["models"] if m["model_id"] == "pp_ocr/small-small"
        )
        assert entry["model_class_name"] == "PPOCRv6StructuredOCR"
        assert entry["model_mro_names"][0] == "PPOCRv6StructuredOCR"
        assert "infer" in entry["actions"] and entry["class_names"] is None
        assert mm.list_models()[0]["state"] == "loaded"
    finally:
        mm.shutdown()


def test_process_pp_ocr_returns_structured_tuple_with_numpy():
    mm = ModelManager()
    try:
        with patch(_FROM_PRETRAINED, side_effect=_stage_for):
            mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0)
        texts, detections = mm.process(
            "pp_ocr/small-small",
            action="infer",
            serialize=False,
            wire_marshalling=True,
            images=_image(),
        )
        assert texts == ["3x6\n4x8"]
        assert isinstance(detections[0].xyxy, np.ndarray)
        assert detections[0].bboxes_metadata[1]["text"] == "4x8"
    finally:
        mm.shutdown()


def test_load_pp_ocr_disabled_stage_loads_one_model():
    mm = ModelManager()
    try:
        with patch(_FROM_PRETRAINED, side_effect=_stage_for) as fp:
            mm.load("pp_ocr/none-small", api_key="k", warmup_iters=0)
        assert [c.args[0] for c in fp.call_args_list] == ["pp-ocrv6-rec/small"]
        texts, detections = mm.process(
            "pp_ocr/none-small", serialize=False, wire_marshalling=True, images=_image()
        )
        assert texts == ["20x20"] and len(detections[0]) == 0
    finally:
        mm.shutdown()


def test_partial_stage_failure_registers_nothing_and_allows_retry():
    mm = ModelManager()
    created = []

    def _det_then_fail(model_id, **kwargs):
        if model_id.startswith("pp-ocrv6-det/"):
            created.append(_FakeDetector())
            return created[-1]
        raise ModelNotFoundError("rec")

    try:
        with patch(_FROM_PRETRAINED, side_effect=_det_then_fail):
            with pytest.raises(ModelNotFoundError) as exc_info:
                mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0)
        assert "pp_ocr/small-small" not in mm and len(mm) == 0
        ref = weakref.ref(created[0])
        created.clear()
        assert exc_info.value.__traceback__ is not None
        assert ref() is None
        del exc_info
        with patch(_FROM_PRETRAINED, side_effect=_stage_for):
            mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0)
        assert "pp_ocr/small-small" in mm
    finally:
        mm.shutdown()


def test_composition_failure_releases_stage_models():
    mm = ModelManager()
    created = []

    def _both_stages(model_id, **kwargs):
        created.append(_FakeDetector() if "-det/" in model_id else _FakeRecognizer())
        return created[-1]

    gc.disable()
    try:
        with patch(_FROM_PRETRAINED, side_effect=_both_stages), patch(
            "inference_models.model_pipelines.auto_loaders.pipelines_registry.resolve_pipeline_class",
            side_effect=RuntimeError("no class"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0)
        assert "pp_ocr/small-small" not in mm and len(mm) == 0
        refs = [weakref.ref(stage) for stage in created]
        created.clear()
        assert exc_info.value.__traceback__ is not None
        assert [ref() for ref in refs] == [None, None]
        del exc_info
    finally:
        gc.enable()
        mm.shutdown()


def test_stage_models_get_caches_attached():
    mm = ModelManager()
    try:
        with patch(_FROM_PRETRAINED, side_effect=_stage_for), patch(
            "inference_model_manager.backends.base.attach_model_caches"
        ) as attach:
            mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0)
        attached = [type(c.args[0]).__name__ for c in attach.call_args_list]
        assert attached[:2] == ["_FakeDetector", "_FakeRecognizer"]
    finally:
        mm.shutdown()


def test_unload_pipeline_releases_stage_models():
    mm = ModelManager()
    created = []

    def _tracked_stage(model_id, **kwargs):
        stage = _stage_for(model_id)
        created.append(stage)
        return stage

    try:
        with patch(_FROM_PRETRAINED, side_effect=_tracked_stage):
            mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0)
        ref = weakref.ref(created[0])
        created.clear()
        mm.unload("pp_ocr/small-small")
        assert ref() is None and "pp_ocr/small-small" not in mm
    finally:
        mm.shutdown()


def test_pipeline_counts_as_one_entry_for_capacity(monkeypatch):
    monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 1)
    mm = ModelManager()
    try:
        with patch(_FROM_PRETRAINED, side_effect=_stage_for):
            mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0)
        assert len(mm) == 1
        with patch(_FROM_PRETRAINED, return_value=_FakeRecognizer()):
            mm.load("other-model", api_key="k", warmup_iters=0)
        assert "pp_ocr/small-small" not in mm and "other-model" in mm
    finally:
        mm.shutdown()


def test_pinned_pipeline_survives_eviction(monkeypatch):
    monkeypatch.setattr(cfg, "INFERENCE_MAX_ACTIVE_MODELS", 1)
    mm = ModelManager()
    try:
        with patch(_FROM_PRETRAINED, side_effect=_stage_for):
            mm.load("pp_ocr/small-small", api_key="k", warmup_iters=0, pinned=True)
        with patch(_FROM_PRETRAINED, return_value=_FakeRecognizer()):
            mm.load("other-model", api_key="k", warmup_iters=0)
        assert "pp_ocr/small-small" in mm and "other-model" in mm
    finally:
        mm.shutdown()
