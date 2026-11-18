"""Extra images on the Vision Events sink, for both the numpy and tensor variants."""

import base64
import importlib
from types import ModuleType
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from roboflow_workflows.prototypes.platform_client import OFFLINE_PLATFORM_CLIENT

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

SINK_MODULES = [
    "roboflow_workflows.core_steps.sinks.roboflow.vision_events.v1",
    "roboflow_workflows.core_steps.sinks.roboflow.vision_events.v1_tensor",
]


@pytest.fixture(params=SINK_MODULES, ids=["v1", "v1_tensor"])
def sink(request: pytest.FixtureRequest) -> ModuleType:
    try:
        module = importlib.import_module(request.param)
    except ImportError as error:  # v1_tensor needs the tensor-native stack
        pytest.skip(f"{request.param} does not import here: {error}")

    return module


@pytest.fixture
def sent_local(
    sink: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> List[Tuple[str, dict]]:
    calls: List[Tuple[str, dict]] = []

    def fake_send_local_event(url: str, payload: dict) -> Tuple[bool, str, str]:
        calls.append((url, payload))
        return False, "ok", f"evt-{len(calls)}"

    monkeypatch.setattr(sink, "_send_local_event", fake_send_local_event)

    return calls


def _image(value: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame"),
        numpy_image=np.full((8, 8, 3), value, dtype=np.uint8),
    )


def _manifest(sink: ModuleType, **fields: Any) -> Any:
    manifest = sink.BlockManifest.model_validate(
        {
            "type": "roboflow_core/roboflow_vision_events@v1",
            "name": "vision_events",
            "event_type": "quality_check",
            "solution": "wheel-inspection",
            **fields,
        }
    )

    return manifest


def _local_event_kwargs(**overrides: Any) -> Dict[str, Any]:
    kwargs = {
        "event_store_url": "http://localhost:8001",
        "input_image": None,
        "output_image": None,
        "prediction": None,
        "event_type": "custom",
        "solution": "s",
        "event_data": {},
        "custom_metadata": {},
    }
    kwargs.update(overrides)

    return kwargs


def test_manifest_extra_images_defaults_to_empty(sink: ModuleType) -> None:
    assert _manifest(sink).extra_images == {}


def test_manifest_accepts_image_selectors_keyed_by_label(sink: ModuleType) -> None:
    manifest = _manifest(sink, extra_images={"raw_crop": "$steps.cropping.crops"})

    assert manifest.extra_images == {"raw_crop": "$steps.cropping.crops"}


def test_manifest_rejects_non_selector_extra_image(sink: ModuleType) -> None:
    with pytest.raises(ValueError):
        _manifest(sink, extra_images={"raw_crop": "not a selector"})


def test_local_event_appends_each_extra_image_with_label_and_no_annotations(
    sink: ModuleType, sent_local: List[Tuple[str, dict]]
) -> None:
    frame, annotated, crop = _image(10), _image(20), _image(30)

    error_status, _, event_id = sink._execute_local_event(
        **_local_event_kwargs(
            input_image=frame,
            output_image=annotated,
            event_type="quality_check",
            event_data={"result": "fail"},
            extra_images={"raw_crop": crop},
        )
    )

    assert (error_status, event_id) == (False, "evt-1")
    payload = sent_local[0][1]
    assert payload["images"] == [
        {
            "base64Image": annotated.base64_image,
            "inputBase64Image": frame.base64_image,
            "label": "workflow",
        },
        {"base64Image": crop.base64_image, "label": "raw_crop"},
    ]
    assert base64.b64decode(payload["images"][1]["base64Image"])[:2] == b"\xff\xd8"
    assert payload["displayImagePosition"] == 0


def test_local_event_skips_extra_image_that_resolved_to_none(
    sink: ModuleType, sent_local: List[Tuple[str, dict]]
) -> None:
    crop = _image(30)

    sink._execute_local_event(
        **_local_event_kwargs(
            input_image=_image(10),
            extra_images={"raw_crop": crop, "missing": None},
        )
    )

    labels = [entry["label"] for entry in sent_local[0][1]["images"]]
    assert labels == ["workflow", "raw_crop"]


def test_local_event_with_only_extra_images_displays_the_first(
    sink: ModuleType, sent_local: List[Tuple[str, dict]]
) -> None:
    crop = _image(30)

    sink._execute_local_event(**_local_event_kwargs(extra_images={"raw_crop": crop}))

    payload = sent_local[0][1]
    assert payload["images"] == [
        {"base64Image": crop.base64_image, "label": "raw_crop"}
    ]
    assert payload["displayImagePosition"] == 0


@pytest.mark.parametrize("extra_images", [None, {}, {"missing": None}])
def test_local_event_without_extra_images_is_unchanged(
    sink: ModuleType,
    sent_local: List[Tuple[str, dict]],
    extra_images: Any,
) -> None:
    frame = _image(10)

    sink._execute_local_event(**_local_event_kwargs(input_image=frame))
    sink._execute_local_event(
        **_local_event_kwargs(input_image=frame, extra_images=extra_images)
    )

    baseline, with_field = sent_local[0][1], sent_local[1][1]
    baseline.pop("inference_timestamp")
    with_field.pop("inference_timestamp")
    assert with_field == baseline


def _record_cloud_calls(
    sink: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Tuple[List[Tuple[WorkflowImageData, int]], List[dict]]:
    uploads: List[Tuple[WorkflowImageData, int]] = []
    events: List[dict] = []
    source_ids: Dict[int, str] = {}

    def fake_upload(
        api_base_url: str,
        api_key: str,
        image_data: WorkflowImageData,
        jpeg_quality: int = 85,
        platform_client: Any = None,
    ) -> Tuple[str, str]:
        uploads.append((image_data, jpeg_quality))
        # One id per distinct image, numbered in first-upload order.
        source_id = source_ids.setdefault(id(image_data), f"src-{len(source_ids) + 1}")
        return source_id, ""

    def fake_send_event(
        api_base_url: str, api_key: str, payload: dict, platform_client: Any = None
    ) -> Tuple[bool, str]:
        events.append(payload)
        return False, "ok"

    monkeypatch.setattr(sink, "_upload_image", fake_upload)
    monkeypatch.setattr(sink, "_send_event", fake_send_event)

    return uploads, events


def _cloud_event_kwargs(**overrides: Any) -> Dict[str, Any]:
    kwargs = {
        "api_base_url": "https://api.roboflow.com",
        "api_key": "key",
        "platform_client": OFFLINE_PLATFORM_CLIENT,
        "input_image": None,
        "output_image": None,
        "prediction": None,
        "event_type": "quality_check",
        "solution": "wheel-inspection",
        "event_data": {"result": "pass"},
        "custom_metadata": {},
    }
    kwargs.update(overrides)

    return kwargs


def test_cloud_event_uploads_each_extra_image_unannotated_at_input_quality(
    sink: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads, events = _record_cloud_calls(sink, monkeypatch)
    annotated, crop = _image(20), _image(30)

    error_status, _, event_id = sink._execute_vision_event(
        **_cloud_event_kwargs(
            output_image=annotated,
            extra_images={"raw_crop": crop, "missing": None},
        )
    )

    assert error_status is False
    assert event_id == events[0]["eventId"]
    assert uploads == [(annotated, 85), (crop, 95)]
    assert events[0]["images"] == [
        {"sourceId": "src-1", "label": "workflow"},
        {"sourceId": "src-2", "label": "raw_crop"},
    ]


@pytest.mark.parametrize("extra_images", [None, {}, {"missing": None}])
def test_cloud_event_without_extra_images_is_unchanged(
    sink: ModuleType, monkeypatch: pytest.MonkeyPatch, extra_images: Any
) -> None:
    uploads, events = _record_cloud_calls(sink, monkeypatch)
    frame = _image(10)

    sink._execute_vision_event(**_cloud_event_kwargs(input_image=frame))
    sink._execute_vision_event(
        **_cloud_event_kwargs(input_image=frame, extra_images=extra_images)
    )

    assert len(uploads) == 2
    baseline, with_field = events
    assert with_field["images"] == baseline["images"]
    assert {k for k in with_field if k not in ("eventId", "timestamp")} == {
        k for k in baseline if k not in ("eventId", "timestamp")
    }


def test_block_run_passes_extra_images_to_the_event_store(
    sink: ModuleType, sent_local: List[Tuple[str, dict]]
) -> None:
    block = sink.RoboflowVisionEventsBlockV1(
        api_key=None, background_tasks=None, thread_pool_executor=None
    )
    crop = _image(30)

    result = block.run(
        input_image=_image(10),
        output_image=_image(20),
        predictions=None,
        event_type="quality_check",
        solution="wheel-inspection",
        custom_metadata={},
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
        write_to_event_store=True,
        qc_result="pass",
        extra_images={"raw_crop": crop},
    )

    assert result["event_id"] == "evt-1"
    assert sent_local[0][1]["images"][1] == {
        "base64Image": crop.base64_image,
        "label": "raw_crop",
    }


def test_block_run_passes_extra_images_to_the_cloud(
    sink: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads, events = _record_cloud_calls(sink, monkeypatch)
    block = sink.RoboflowVisionEventsBlockV1(
        api_key="key", background_tasks=None, thread_pool_executor=None
    )
    crop = _image(30)

    result = block.run(
        input_image=None,
        output_image=None,
        predictions=None,
        event_type="quality_check",
        solution="wheel-inspection",
        custom_metadata={},
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
        qc_result="pass",
        extra_images={"raw_crop": crop},
    )

    assert result["error_status"] is False
    assert uploads == [(crop, 95)]
    assert events[0]["images"] == [{"sourceId": "src-1", "label": "raw_crop"}]
