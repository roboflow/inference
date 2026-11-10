"""Default discovery composes local models, direct OCR and retirement guards."""
import roboflow_workflows.v2.blocks.models.vision_language_models._catalogue_openrouter_vendors as _v2_family_0
import roboflow_workflows.v2.blocks.models.vision_language_models._catalogue_compatible_vendors as _v2_family_1
import roboflow_workflows.v2.blocks.models.vision_language_models.kimi_openrouter.v3 as _v2_family_2
import roboflow_workflows.v2.blocks.models.vision_language_models.meta_vlm.v3 as _v2_family_3
import roboflow_workflows.v2.blocks.models.vision_language_models.zai_vlm.v2 as _v2_family_4
import roboflow_workflows.v2.blocks.models.vision_language_models.google_gemma.v4 as _v2_family_5
import roboflow_workflows.v2.blocks.models.vision_language_models.spacexai.v3 as _v2_family_6

import json
import re

import pytest
import torch
from roboflow_workflows.v2 import blocks
from roboflow_workflows.v2.blocks import models
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import UnknownBlockError
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.object_detection import Detections
from tests.v2.support.language.fake_chat import FakeChatServer

_ADDED = (
    ("Cosmos3EdgeBlock", "roboflow_core/cosmos3_edge@v1"),
    ("Florence2Block", "roboflow_core/florence_2@v2"),
    ("GoogleVisionOCRBlock", "roboflow_core/google_vision_ocr@v1"),
    ("Moondream2Block", "roboflow_core/moondream2@v1"),
    ("QwenVLMBlock", "roboflow_core/qwen_vlm@v4"),
)
_RETIRED = (
    "roboflow_core/byte_tracker@v3",
    "v2/byte_tracker@v2",
    "v2/byte_tracker@v3",
    "v2/gaze@v1",
    "roboflow_core/llama_vision@v2",
    "v2/llama_3_2_vision@v1",
    "v2/lmm@v1",
    "v2/lmm_for_classification@v1",
    "v2/modbus_tcp@v1",
    "roboflow_core/qwen25vl@v1",
    "roboflow_core/qwen3_5_openrouter@v1",
    "roboflow_core/qwen3_6_openrouter@v1",
    "roboflow_core/qwen3vl@v1",
    "v2/sinks@v1",
    "roboflow_core/vlm_as_classifier@v2",
    "roboflow_core/vlm_as_detector@v2",
    "v2/cog_vlm@v1",
    "v2/yolo_world_model@v1",
)


def _catalogue(route):
    if route == "create":
        return blocks.create_catalogue()
    if route == "plugin":
        return Catalogue.from_modules(["roboflow_workflows.v2.blocks"])

    catalogue = Catalogue.from_modules(
        ["roboflow_workflows.v2.blocks"], attribute="create_catalogue"
    )

    return catalogue


@pytest.mark.parametrize("route", ["create", "plugin", "factory"])
def test_five_registrations_share_package_exports_and_default_discovery(route):
    catalogue = _catalogue(route)
    for name, identity in _ADDED:
        block = getattr(models, name)
        assert getattr(blocks, name) is block
        assert name in models.__all__ and name in blocks.__all__
        assert models.BLOCKS.count(block) == 1
        assert catalogue.block_types.count(identity) == 1
        assert catalogue.entry(identity).spec is spec_of(block)
    assert "roboflow_core/roboflow_action_recognition_model@v1" in catalogue.block_types


@pytest.mark.parametrize("route", ["create", "plugin", "factory"])
@pytest.mark.parametrize("identity", _RETIRED)
def test_retired_families_and_core_spellings_cannot_compile(route, identity):
    catalogue = _catalogue(route)
    family = identity.partition("@")[0]
    assert all(value.partition("@")[0] != family for value in catalogue.block_types)
    for spelling in (identity, identity.replace("v2/", "roboflow_core/", 1)):
        definition = {
            "version": "2.0",
            "inputs": [],
            "steps": [{"type": spelling, "name": "retired"}],
            "outputs": [],
        }
        with pytest.raises(UnknownBlockError, match=re.escape(spelling)) as error:
            compile_workflow(definition, catalogue=catalogue)
        assert error.value.step_path == ("retired",)


def test_retired_classes_are_not_default_package_exports():
    retired = {
        "LlamaVisionBlock",
        "Qwen35OpenRouterBlock",
        "Qwen36OpenRouterBlock",
        "VLMAsClassifierBlock",
        "VLMAsDetectorBlock",
        "LMMBlock",
        "LMMForClassificationBlock",
        "Qwen25VLBlock",
        "Qwen3VLBlock",
    }
    for package in (blocks, models):
        assert retired.isdisjoint(package.__all__)
        assert all(not hasattr(package, name) for name in retired)


class _PromptModel:
    def __init__(self, family):
        self.family = family
        self.calls = []

    def prompt(
        self,
        images,
        prompt=None,
        input_color_format=None,
        max_new_tokens=512,
        do_sample=False,
        skip_special_tokens=True,
        enable_thinking=False,
        task=None,
        **kwargs,
    ):
        assert images[0].device.type == "cpu"
        self.calls.append((images[0], prompt))
        if self.family == "florence":
            answer = {task: f"caption-{int(images[0][0, 0, 0])}"}
        elif self.family == "qwen":
            answer = '[{"box_2d": [0, 0, 500, 500], "label": "car"}]'
        else:
            answer = json.dumps({"answer": "clear"})

        return [answer]


class _DetectionModel:
    def __init__(self):
        self.calls = []

    def detect(self, images, classes, max_new_tokens=700, input_color_format=None):
        assert images[0].device.type == "cpu"
        self.calls.append((images[0], classes))
        prediction = Detections(
            xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.9]),
        )

        return [prediction]


@pytest.mark.parametrize("route", ["create", "plugin", "factory"])
@pytest.mark.parametrize("mode", ["run", "phases"])
def test_default_compiled_local_chain_preserves_rows_and_host_cache(route, mode):
    loaded = []
    instances = {
        "florence": _PromptModel("florence"),
        "cosmos": _PromptModel("cosmos"),
        "qwen3_5-2b": _PromptModel("qwen"),
        "moon": _DetectionModel(),
    }

    def load(request):
        loaded.append(request.model_id)
        return instances[request.model_id]

    steps = [
        {
            "name": "caption",
            "type": "roboflow_core/florence_2@v2",
            "image": "$inputs.image",
            "model_id": "florence",
            "task_type": "caption",
        },
        {
            "name": "moon",
            "type": "roboflow_core/moondream2@v1",
            "image": "$inputs.image",
            "model_id": "moon",
            "prompt": "car",
        },
        {
            "name": "cosmos",
            "type": "roboflow_core/cosmos3_edge@v1",
            "image": "$inputs.image",
            "model_id": "cosmos",
            "prompt": "Return JSON",
        },
        {
            "name": "parse",
            "type": "roboflow_core/json_parser@v1",
            "raw_json": "$steps.cosmos.output",
            "expected_fields": ["answer"],
        },
        {
            "name": "qwen",
            "type": "roboflow_core/qwen_vlm@v4",
            "images": "$inputs.image",
            "task_type": "object-detection",
            "classes": ["car"],
        },
        {
            "name": "offset",
            "type": "roboflow_core/detection_offset@v1",
            "predictions": "$steps.qwen.predictions",
            "offset_width": 10,
            "offset_height": 10,
        },
    ]
    selectors = {
        "caption": "$steps.caption.text",
        "moon": "$steps.moon.predictions",
        "answer": "$steps.parse.answer",
        "qwen": "$steps.offset.predictions",
    }
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": n, "selector": s}
            for n, s in selectors.items()
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=_catalogue(route),
        options=CompileOptions(block_execution=mode),
    )
    assert loaded == []
    images = [
        ImageData.from_tensor(
            torch.full((3, 20, 40), i, dtype=torch.uint8), image_id=f"source-{i}"
        )
        for i in (1, 2)
    ]
    session = plan.create_session({"models": ModelCache(load)})
    try:
        for _ in range(2):
            rows = session.run({"image": images}).rows()
            assert [r["caption"] for r in rows] == ["caption-1", "caption-2"]
            assert [r["answer"] for r in rows] == ["clear", "clear"]
            assert [r["moon"].xyxy.tolist() for r in rows] == [
                [[1.0, 2.0, 3.0, 4.0]]
            ] * 2
            assert [r["qwen"].xyxy.tolist() for r in rows] == [
                [[0.0, 0.0, 25.0, 15.0]]
            ] * 2
        assert sorted(loaded) == sorted(instances)
        assert all(len(model.calls) == 4 for model in instances.values())
    finally:
        session.close()


class _GoogleResponse:
    status_code = 200

    def json(self):
        return {
            "responses": [
                {"textAnnotations": [{"description": "STOP", "locale": "en"}]}
            ]
        }


class _GoogleTransport:
    def __init__(self):
        self.calls = []

    def post(self, url, *, params, json, timeout):
        self.calls.append((url, params, json, timeout))
        return _GoogleResponse()


@pytest.mark.parametrize("route", ["create", "plugin", "factory"])
@pytest.mark.parametrize("mode", ["run", "phases"])
def test_default_compiled_google_ocr_uses_only_injected_direct_transport(route, mode):
    transport = _GoogleTransport()
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": [
            {
                "name": "ocr",
                "type": "roboflow_core/google_vision_ocr@v1",
                "image": "$inputs.image",
                "api_key": "test-google-key",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": n, "selector": f"$steps.ocr.{n}"}
            for n in ("text", "language", "predictions")
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=_catalogue(route),
        options=CompileOptions(block_execution=mode),
    )
    assert transport.calls == []
    session = plan.create_session(
        {
            "google_vision_transport": transport,
            "google_vision_base_url": "https://vision.googleapis.test",
        }
    )
    image = ImageData.from_tensor(torch.zeros((3, 20, 40), dtype=torch.uint8))
    try:
        (row,) = session.run({"image": [image]}).rows()
        assert (row["text"], row["language"]) == ("STOP", "en")
        assert len(row["predictions"]) == 0
        assert len(transport.calls) == 1
        url, params, payload, timeout = transport.calls[0]
        assert url == "https://vision.googleapis.test/v1/images:annotate"
        assert params == {"key": "test-google-key"}
        assert payload["requests"][0]["image"]["content"]
        assert timeout > 0
    finally:
        session.close()


def test_default_qwen_openrouter_backend_uses_mock_transport():
    server = FakeChatServer(
        lambda path, body: (
            200,
            {
                "choices": [{"message": {"content": '{"answer": "mocked"}'}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )
    )
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": [
            {
                "name": "qwen",
                "type": "roboflow_core/qwen_vlm@v4",
                "backend": "openrouter",
                "images": "$inputs.image",
                "task_type": "structured-answering",
                "output_structure": {"answer": "text"},
                "api_key": "test-openrouter-key",
            },
            {
                "name": "parse",
                "type": "roboflow_core/json_parser@v1",
                "raw_json": "$steps.qwen.output",
                "expected_fields": ["answer"],
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "answer", "selector": "$steps.parse.answer"}
        ],
    }
    plan = compile_workflow(definition, catalogue=blocks.create_catalogue())
    assert server.requests == []
    with server.http_client() as client:
        session = plan.create_session({"openrouter_http_client": client})
        try:
            image = ImageData.from_tensor(torch.zeros((3, 20, 40), dtype=torch.uint8))
            assert session.run({"image": [image]}).rows() == [{"answer": "mocked"}]
            assert len(server.requests) == 1
        finally:
            session.close()


def test_active_q3_and_g04_module_contracts_remain_available():
    from roboflow_workflows.v2.blocks.models.vision_language_models._shared import (
        compatible_vendors as compatible_vendors,
        openrouter_vendors as openrouter_vendors,
        parsers as parsers,
    )
    from roboflow_workflows.v2.blocks.models.qwen3_5vl.v2 import Qwen35VLBlock

    assert Qwen35VLBlock.type == "roboflow_core/qwen3_5vl@v2"
    assert _v2_family_0.BLOCKS == (
        _v2_family_2.KimiOpenRouterBlock,
        _v2_family_3.MetaVlmBlock,
        _v2_family_4.ZaiVlmBlock,
    )
    assert _v2_family_1.BLOCKS == (
        _v2_family_5.GoogleGemmaBlock,
        _v2_family_6.SpaceXAIBlock,
    )
    assert {
        "decode_florence_2_detections",
        "FLORENCE_2_TASKS",
        "MODEL_BOX_FORMATS",
    }.issubset(vars(parsers))
    assert {"VLMAsClassifierBlock", "VLMAsDetectorBlock"}.isdisjoint(parsers.__all__)
