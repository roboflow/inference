import errno
import json
import os
import re
import tarfile
import time
from unittest.mock import MagicMock
from uuid import UUID

import cv2
import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.sinks.roboflow import vision_events_bundle
from inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1 import (
    _convert_predictions_to_annotations,
)
from inference.core.workflows.core_steps.sinks.roboflow.vision_events_bundle.v1 import (
    BUNDLE_FORMAT_VERSION,
    MAX_ANNOTATIONS_PER_LIST,
    MAX_BUNDLE_SIZE_BYTES,
    BlockManifest,
    VisionEventBundleSinkBlockV1,
    _cap_annotation_lists,
)
from inference.core.workflows.execution_engine.constants import PREDICTION_TYPE_KEY
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

BUNDLE_FILE_NAME_PATTERN = re.compile(
    r"^event_\d{8}T\d{6}_\d{6}_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.tar\.gz$"
)


def files_in(directory) -> list:
    return sorted(p.name for p in directory.iterdir() if not p.name.startswith("."))


def _make_workflow_image(width: int = 100, height: int = 100) -> WorkflowImageData:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=image,
    )


def _make_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[10, 20, 50, 60]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={
            "class_name": np.array(["cat"]),
            PREDICTION_TYPE_KEY: np.array(["object-detection"]),
        },
    )


def _make_block(
    tmp_path,
    allow_access_to_file_system: bool = True,
    allowed_write_directory=None,
    disable_sinks: bool = False,
    background_tasks=None,
    thread_pool_executor=None,
) -> VisionEventBundleSinkBlockV1:
    return VisionEventBundleSinkBlockV1(
        background_tasks=background_tasks,
        thread_pool_executor=thread_pool_executor,
        allow_access_to_file_system=allow_access_to_file_system,
        allowed_write_directory=allowed_write_directory,
        disable_sinks=disable_sinks,
    )


def _run_block(block: VisionEventBundleSinkBlockV1, target_directory: str, **overrides):
    kwargs = {
        "target_directory": target_directory,
        "input_image": None,
        "output_image": None,
        "predictions": None,
        "event_type": "quality_check",
        "custom_metadata": {},
        "fire_and_forget": False,
        "disable_sink": False,
        "solution": None,
        "cooldown_seconds": 0,
        "qc_result": "pass",
    }
    kwargs.update(overrides)
    return block.run(**kwargs)


def _read_bundle(bundle_path: str):
    with tarfile.open(bundle_path, mode="r:gz") as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    payload = json.loads(members["payload.json"])
    return payload, members


# === Manifest validation ===


def test_manifest_parsing_valid() -> None:
    raw_manifest = {
        "type": "roboflow_core/vision_event_bundle@v1",
        "name": "test_step",
        "event_type": "quality_check",
        "target_directory": "/data/bundles",
    }
    manifest = BlockManifest.model_validate(raw_manifest)
    assert manifest.event_type == "quality_check"
    assert manifest.target_directory == "/data/bundles"
    assert manifest.solution is None
    assert manifest.fire_and_forget is True
    assert manifest.disable_sink is False


def test_manifest_parsing_missing_target_directory() -> None:
    raw_manifest = {
        "type": "roboflow_core/vision_event_bundle@v1",
        "name": "test_step",
        "event_type": "quality_check",
    }
    with pytest.raises(Exception):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_parsing_wrong_type() -> None:
    raw_manifest = {
        "type": "roboflow_core/roboflow_vision_events@v1",
        "name": "test_step",
        "event_type": "quality_check",
        "target_directory": "/data/bundles",
    }
    with pytest.raises(Exception):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_accepts_optional_solution() -> None:
    raw_manifest = {
        "type": "roboflow_core/vision_event_bundle@v1",
        "name": "test_step",
        "event_type": "quality_check",
        "target_directory": "/data/bundles",
        "solution": "my-use-case",
    }
    manifest = BlockManifest.model_validate(raw_manifest)
    assert manifest.solution == "my-use-case"


def test_manifest_defaults_file_name_to_none() -> None:
    raw_manifest = {
        "type": "roboflow_core/vision_event_bundle@v1",
        "name": "test_step",
        "event_type": "quality_check",
        "target_directory": "/data/bundles",
    }
    manifest = BlockManifest.model_validate(raw_manifest)
    assert manifest.file_name is None


@pytest.mark.parametrize(
    "file_name", ["person_batch7", "person_batch7.tar.gz", "cam-01.2026", "a"]
)
def test_manifest_accepts_valid_file_name(file_name: str) -> None:
    raw_manifest = {
        "type": "roboflow_core/vision_event_bundle@v1",
        "name": "test_step",
        "event_type": "quality_check",
        "target_directory": "/data/bundles",
        "file_name": file_name,
    }
    manifest = BlockManifest.model_validate(raw_manifest)
    assert manifest.file_name == file_name


@pytest.mark.parametrize(
    "file_name",
    [
        "",
        "../escape",
        "nested/bundle",
        ".hidden",
        "-leading-dash",
        "with space",
        "unicode_\u00e9",
        "trailing_newline\n",
        "embedded\nnewline",
        "x" * 201,
    ],
)
def test_manifest_rejects_unsafe_file_name(file_name: str) -> None:
    raw_manifest = {
        "type": "roboflow_core/vision_event_bundle@v1",
        "name": "test_step",
        "event_type": "quality_check",
        "target_directory": "/data/bundles",
        "file_name": file_name,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_accepts_selector_as_file_name() -> None:
    raw_manifest = {
        "type": "roboflow_core/vision_event_bundle@v1",
        "name": "test_step",
        "event_type": "quality_check",
        "target_directory": "/data/bundles",
        "file_name": "$steps.compose.output",
    }
    manifest = BlockManifest.model_validate(raw_manifest)
    assert manifest.file_name == "$steps.compose.output"


# === Happy path (sync) ===


def test_sync_write_with_both_images_and_predictions(tmp_path) -> None:
    block = _make_block(tmp_path)
    detections = _make_detections()

    result = _run_block(
        block,
        str(tmp_path),
        input_image=_make_workflow_image(),
        output_image=_make_workflow_image(),
        predictions=detections,
        event_type="quality_check",
        custom_metadata={"camera_id": "cam_01"},
    )

    assert result["error_status"] is False
    assert result["throttling_status"] is False
    assert result["message"] == "Vision event bundle written successfully"

    bundle_files = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert len(bundle_files) == 1
    bundle_file = bundle_files[0]
    assert BUNDLE_FILE_NAME_PATTERN.match(bundle_file.name)
    assert str(bundle_file) == result["bundle_path"]

    payload, members = _read_bundle(str(bundle_file))
    assert payload["bundleFormatVersion"] == BUNDLE_FORMAT_VERSION
    assert payload["eventId"] == result["event_id"]
    assert UUID(payload["eventId"])
    assert payload["eventId"] in bundle_file.name
    assert payload["eventType"] == "quality_check"
    assert payload["timestamp"].endswith("+00:00")
    assert "useCaseId" not in payload
    assert "solution" not in payload
    assert payload["eventData"] == {"result": "pass"}
    assert payload["customMetadata"] == {"camera_id": "cam_01"}
    assert payload["displayImagePosition"] == 0

    assert len(payload["images"]) == 1
    image_entry = payload["images"][0]
    assert image_entry["label"] == "workflow"
    assert "sourceId" not in image_entry
    assert "inputSourceId" not in image_entry
    assert image_entry["file"].startswith("images/")
    assert image_entry["file"].endswith(".jpg")
    assert image_entry["inputFile"].startswith("images/")
    assert image_entry["inputFile"].endswith(".jpg")
    assert image_entry["file"] != image_entry["inputFile"]

    expected_annotations = _convert_predictions_to_annotations(detections)
    assert image_entry["objectDetections"] == expected_annotations["objectDetections"]

    assert set(members.keys()) == {
        "payload.json",
        image_entry["file"],
        image_entry["inputFile"],
    }
    for member_name in (image_entry["file"], image_entry["inputFile"]):
        decoded = cv2.imdecode(
            np.frombuffer(members[member_name], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape == (100, 100, 3)


def test_sync_write_with_solution_set(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), solution="my-use-case")

    assert result["error_status"] is False
    payload, _ = _read_bundle(result["bundle_path"])
    assert payload["useCaseId"] == "my-use-case"


def test_sync_write_output_image_only(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), output_image=_make_workflow_image())

    payload, members = _read_bundle(result["bundle_path"])
    image_entry = payload["images"][0]
    assert "file" in image_entry
    assert "inputFile" not in image_entry
    assert set(members.keys()) == {"payload.json", image_entry["file"]}


def test_sync_write_input_image_only(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), input_image=_make_workflow_image())

    payload, members = _read_bundle(result["bundle_path"])
    image_entry = payload["images"][0]
    assert "inputFile" in image_entry
    assert "file" not in image_entry
    assert set(members.keys()) == {"payload.json", image_entry["inputFile"]}


def test_sync_write_no_images(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path))

    payload, members = _read_bundle(result["bundle_path"])
    assert payload["images"] == []
    assert "displayImagePosition" not in payload
    assert set(members.keys()) == {"payload.json"}


def test_sync_write_event_data_per_event_type(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(
        block,
        str(tmp_path),
        event_type="safety_alert",
        qc_result=None,
        alert_type="no_hardhat",
        severity="high",
        alert_description="Worker without hardhat",
        external_id="ext-1",
    )

    payload, _ = _read_bundle(result["bundle_path"])
    assert payload["eventData"] == {
        "alertType": "no_hardhat",
        "severity": "high",
        "description": "Worker without hardhat",
        "externalId": "ext-1",
    }


def test_sync_write_empty_event_data_omitted(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), qc_result=None)

    payload, _ = _read_bundle(result["bundle_path"])
    assert "eventData" not in payload
    assert "customMetadata" not in payload


def test_target_directory_created_if_missing(tmp_path) -> None:
    block = _make_block(tmp_path)
    nested = tmp_path / "nested" / "bundles"

    result = _run_block(block, str(nested), output_image=_make_workflow_image())

    assert result["error_status"] is False
    assert nested.is_dir()
    payload, _ = _read_bundle(result["bundle_path"])
    assert payload["bundleFormatVersion"] == BUNDLE_FORMAT_VERSION


# === Security ===


def test_run_raises_when_file_system_access_forbidden(tmp_path) -> None:
    block = _make_block(tmp_path, allow_access_to_file_system=False)

    with pytest.raises(RuntimeError):
        _run_block(block, str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_run_raises_when_target_outside_allowed_directory(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    block = _make_block(tmp_path, allowed_write_directory=str(allowed))

    with pytest.raises(ValueError):
        _run_block(block, str(outside))


def test_run_raises_on_escape_via_parent_traversal(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    block = _make_block(tmp_path, allowed_write_directory=str(allowed))

    with pytest.raises(ValueError):
        _run_block(block, str(allowed / ".." / "outside"))


def test_run_succeeds_inside_allowed_directory(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    block = _make_block(tmp_path, allowed_write_directory=str(allowed))

    result = _run_block(block, str(allowed / "bundles"))

    assert result["error_status"] is False


# === Dispatch modes ===


def test_fire_and_forget_uses_background_tasks(tmp_path) -> None:
    background_tasks = MagicMock()
    block = _make_block(tmp_path, background_tasks=background_tasks)

    result = _run_block(block, str(tmp_path), fire_and_forget=True)

    background_tasks.add_task.assert_called_once()
    assert result["error_status"] is False
    assert result["event_id"] == ""
    assert result["bundle_path"] == ""
    assert result["message"] == "Vision event bundle written in background task"
    assert list(tmp_path.iterdir()) == []


def test_fire_and_forget_uses_thread_pool_executor(tmp_path) -> None:
    thread_pool_executor = MagicMock()
    block = _make_block(tmp_path, thread_pool_executor=thread_pool_executor)

    result = _run_block(block, str(tmp_path), fire_and_forget=True)

    thread_pool_executor.submit.assert_called_once()
    assert result["error_status"] is False
    assert result["bundle_path"] == ""
    assert list(tmp_path.iterdir()) == []


def test_disable_sink_returns_without_writing(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), disable_sink=True)

    assert result["error_status"] is False
    assert result["event_id"] == ""
    assert result["bundle_path"] == ""
    assert "disabled" in result["message"]
    assert list(tmp_path.iterdir()) == []


def test_disable_sinks_execution_policy_returns_without_writing(tmp_path) -> None:
    block = _make_block(tmp_path, disable_sinks=True)

    result = _run_block(block, str(tmp_path))

    assert result["error_status"] is False
    assert "disabled" in result["message"]
    assert list(tmp_path.iterdir()) == []


# === Cooldown ===


def test_cooldown_throttles_second_event(tmp_path) -> None:
    block = _make_block(tmp_path)

    first = _run_block(block, str(tmp_path), cooldown_seconds=100)
    second = _run_block(block, str(tmp_path), cooldown_seconds=100)

    assert first["error_status"] is False
    assert first["throttling_status"] is False
    assert second["error_status"] is False
    assert second["throttling_status"] is True
    assert second["event_id"] == ""
    assert second["bundle_path"] == ""
    bundle_files = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert len(bundle_files) == 1


def test_zero_cooldown_allows_consecutive_events(tmp_path) -> None:
    block = _make_block(tmp_path)

    first = _run_block(block, str(tmp_path), cooldown_seconds=0)
    time.sleep(0.001)
    second = _run_block(block, str(tmp_path), cooldown_seconds=0)

    assert first["throttling_status"] is False
    assert second["throttling_status"] is False
    bundle_files = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert len(bundle_files) == 2


# === Atomicity and failure semantics ===


def test_failed_write_leaves_no_partial_bundle(tmp_path, monkeypatch) -> None:
    block = _make_block(tmp_path)

    def _explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(vision_events_bundle.v1, "_publish_bundle", _explode)

    result = _run_block(block, str(tmp_path), output_image=_make_workflow_image())

    assert result["error_status"] is True
    assert "OSError" in result["message"]
    assert result["event_id"] == ""
    assert result["bundle_path"] == ""
    non_dotfiles = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert non_dotfiles == []


def test_temp_files_are_dot_prefixed(tmp_path, monkeypatch) -> None:
    # a file mover matching *.tar.gz must never see a partially written bundle,
    # so whatever the publish writes first has to be a dotfile
    observed_temp_names = []
    real_link = os.link

    def _spy(source, destination, **kwargs):
        observed_temp_names.append(os.path.basename(source))
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(vision_events_bundle.v1.os, "link", _spy)

    result = _run_block(block=_make_block(tmp_path), target_directory=str(tmp_path))

    assert result["error_status"] is False
    assert len(observed_temp_names) == 1
    assert observed_temp_names[0].startswith(".")
    # and nothing dot-prefixed survives the publish
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


# === P1 regression: 25 MiB bundle size limit ===


def test_oversized_bundle_returns_error_not_writes(tmp_path, monkeypatch) -> None:
    block = _make_block(tmp_path)
    oversized = b"x" * (MAX_BUNDLE_SIZE_BYTES + 1)
    monkeypatch.setattr(
        vision_events_bundle.v1, "_build_tar_bytes", lambda **_: oversized
    )

    result = _run_block(block, str(tmp_path), output_image=_make_workflow_image())

    assert result["error_status"] is True
    assert "25 MiB" in result["message"] or "limit" in result["message"].lower()
    assert result["event_id"] == ""
    assert result["bundle_path"] == ""
    non_dotfiles = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert non_dotfiles == []


def test_bundle_at_exact_size_limit_succeeds(tmp_path, monkeypatch) -> None:
    block = _make_block(tmp_path)
    at_limit = b"x" * MAX_BUNDLE_SIZE_BYTES
    monkeypatch.setattr(
        vision_events_bundle.v1, "_build_tar_bytes", lambda **_: at_limit
    )

    result = _run_block(block, str(tmp_path))

    assert result["error_status"] is False


def test_bundle_one_byte_over_limit_returns_error(tmp_path, monkeypatch) -> None:
    block = _make_block(tmp_path)
    over_limit = b"x" * (MAX_BUNDLE_SIZE_BYTES + 1)
    monkeypatch.setattr(
        vision_events_bundle.v1, "_build_tar_bytes", lambda **_: over_limit
    )

    result = _run_block(block, str(tmp_path))

    assert result["error_status"] is True
    non_dotfiles = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert non_dotfiles == []


# === P1 regression: annotation list capping at 1000 ===


def test_cap_annotation_lists_truncates_over_limit() -> None:
    annotations = {
        "objectDetections": [{"class": "cat"}] * (MAX_ANNOTATIONS_PER_LIST + 50),
        "classifications": [{"class": "dog"}] * (MAX_ANNOTATIONS_PER_LIST + 1),
        "instanceSegmentations": [{"class": "bird"}] * MAX_ANNOTATIONS_PER_LIST,
        "keypoints": [{"class": "person"}] * 5,
    }

    result = _cap_annotation_lists(annotations)

    assert len(result["objectDetections"]) == MAX_ANNOTATIONS_PER_LIST
    assert len(result["classifications"]) == MAX_ANNOTATIONS_PER_LIST
    assert len(result["instanceSegmentations"]) == MAX_ANNOTATIONS_PER_LIST
    assert len(result["keypoints"]) == 5


def test_cap_annotation_lists_preserves_under_limit() -> None:
    annotations = {
        "objectDetections": [{"class": "cat"}] * 10,
    }

    result = _cap_annotation_lists(annotations)

    assert result["objectDetections"] == annotations["objectDetections"]


def test_cap_annotation_lists_passthrough_unknown_keys() -> None:
    annotations = {"someOtherKey": "value", "objectDetections": [{"class": "x"}] * 3}

    result = _cap_annotation_lists(annotations)

    assert result["someOtherKey"] == "value"
    assert len(result["objectDetections"]) == 3


def test_annotations_capped_in_written_bundle(tmp_path) -> None:
    block = _make_block(tmp_path)
    n = MAX_ANNOTATIONS_PER_LIST + 5
    detections = sv.Detections(
        xyxy=np.tile(np.array([[10, 20, 50, 60]], dtype=float), (n, 1)),
        confidence=np.ones(n) * 0.9,
        class_id=np.zeros(n, dtype=int),
        data={
            "class_name": np.array(["cat"] * n),
            PREDICTION_TYPE_KEY: np.array(["object-detection"] * n),
        },
    )

    result = _run_block(
        block,
        str(tmp_path),
        input_image=_make_workflow_image(),
        predictions=detections,
    )

    assert result["error_status"] is False
    payload, _ = _read_bundle(result["bundle_path"])
    image_entry = payload["images"][0]
    assert len(image_entry["objectDetections"]) == MAX_ANNOTATIONS_PER_LIST


# === P1 regression: strict JSON serialization (allow_nan=False) ===


def test_nan_in_payload_returns_error_not_writes(tmp_path, monkeypatch) -> None:
    block = _make_block(tmp_path)

    original_build_payload = vision_events_bundle.v1._build_bundle_payload

    def _inject_nan(**kwargs):
        payload = original_build_payload(**kwargs)
        payload["_nan_field"] = float("nan")
        return payload

    monkeypatch.setattr(vision_events_bundle.v1, "_build_bundle_payload", _inject_nan)

    result = _run_block(block, str(tmp_path))

    assert result["error_status"] is True
    assert result["event_id"] == ""
    assert result["bundle_path"] == ""
    non_dotfiles = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert non_dotfiles == []


def test_infinity_in_payload_returns_error_not_writes(tmp_path, monkeypatch) -> None:
    block = _make_block(tmp_path)

    original_build_payload = vision_events_bundle.v1._build_bundle_payload

    def _inject_inf(**kwargs):
        payload = original_build_payload(**kwargs)
        payload["_inf_field"] = float("inf")
        return payload

    monkeypatch.setattr(vision_events_bundle.v1, "_build_bundle_payload", _inject_inf)

    result = _run_block(block, str(tmp_path))

    assert result["error_status"] is True
    non_dotfiles = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert non_dotfiles == []


def test_valid_float_in_payload_is_serialized_correctly(tmp_path) -> None:
    block = _make_block(tmp_path)
    detections = _make_detections()

    result = _run_block(
        block, str(tmp_path), input_image=_make_workflow_image(), predictions=detections
    )

    assert result["error_status"] is False
    payload, _ = _read_bundle(result["bundle_path"])
    assert payload["images"][0]["objectDetections"][0]["confidence"] == pytest.approx(
        0.9
    )


# === Custom file name ===


def test_default_name_used_when_file_name_unset(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path))

    assert result["error_status"] is False
    assert BUNDLE_FILE_NAME_PATTERN.match(os.path.basename(result["bundle_path"]))


def test_custom_file_name_replaces_generated_name(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), file_name="person_batch7")

    assert result["error_status"] is False
    # nothing of the generated name survives - no timestamp, no event id
    assert os.path.basename(result["bundle_path"]) == "person_batch7.tar.gz"
    assert files_in(tmp_path) == ["person_batch7.tar.gz"]


def test_custom_file_name_keeps_existing_suffix(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), file_name="person_batch7.tar.gz")

    assert os.path.basename(result["bundle_path"]) == "person_batch7.tar.gz"


def test_custom_file_name_bundle_contents_are_valid(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(
        block,
        str(tmp_path),
        file_name="custom_name",
        input_image=_make_workflow_image(),
    )

    payload, members = _read_bundle(result["bundle_path"])
    assert payload["bundleFormatVersion"] == BUNDLE_FORMAT_VERSION
    # the event id still lives in the payload even though the name omits it
    assert payload["eventId"]
    assert any(m.startswith("images/") for m in members)


def test_two_blocks_route_by_directory_and_name(tmp_path) -> None:
    # each block owns its own directory and file name, so a downstream file
    # mover can route purely by folder and name
    person_directory = tmp_path / "person"
    event_directory = tmp_path / "event"

    person_result = _run_block(
        _make_block(tmp_path), str(person_directory), file_name="person_001"
    )
    event_result = _run_block(_make_block(tmp_path), str(event_directory))

    assert person_result["error_status"] is False
    assert event_result["error_status"] is False
    assert files_in(person_directory) == ["person_001.tar.gz"]
    assert BUNDLE_FILE_NAME_PATTERN.match(files_in(event_directory)[0])


def test_repeated_custom_file_name_raises_and_preserves_original(tmp_path) -> None:
    block = _make_block(tmp_path)
    first = _run_block(block, str(tmp_path), file_name="collides")
    original_bytes = (tmp_path / "collides.tar.gz").read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        _run_block(block, str(tmp_path), file_name="collides")

    assert first["error_status"] is False
    assert files_in(tmp_path) == ["collides.tar.gz"]
    assert (tmp_path / "collides.tar.gz").read_bytes() == original_bytes


def test_collision_raises_before_background_dispatch(tmp_path) -> None:
    # fire_and_forget returns before the write happens, so the guard has to run
    # in the foreground - otherwise a clobbering write is reported as success
    (tmp_path / "collides.tar.gz").write_bytes(b"existing bundle")
    background_tasks = MagicMock()
    block = _make_block(tmp_path, background_tasks=background_tasks)

    with pytest.raises(ValueError, match="already exists"):
        _run_block(block, str(tmp_path), file_name="collides", fire_and_forget=True)

    background_tasks.add_task.assert_not_called()
    assert (tmp_path / "collides.tar.gz").read_bytes() == b"existing bundle"


def test_concurrent_writes_of_one_name_never_overwrite(tmp_path) -> None:
    # a plain exists() check lets both threads through and the later write
    # clobbers the earlier one; the reservation makes exactly one win
    from concurrent.futures import ThreadPoolExecutor

    def attempt(_):
        try:
            result = _run_block(
                _make_block(tmp_path), str(tmp_path), file_name="shared"
            )
        except ValueError:
            # lost the foreground check
            return "rejected"
        # lost the publish race: reported through error_status, not an exception
        return "rejected" if result["error_status"] else "written"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))

    assert outcomes.count("written") == 1
    assert outcomes.count("rejected") == 7
    assert files_in(tmp_path) == ["shared.tar.gz"]
    # the single winner's bundle is intact, not a half-written loser's
    payload, _ = _read_bundle(str(tmp_path / "shared.tar.gz"))
    assert payload["bundleFormatVersion"] == BUNDLE_FORMAT_VERSION


def test_interrupted_write_leaves_no_marker_blocking_the_name(
    tmp_path, monkeypatch
) -> None:
    # a durable reservation would survive the failure and poison the name for
    # every later event; only dot-prefixed temporaries may be left behind
    monkeypatch.setattr(
        vision_events_bundle.v1,
        "_publish_bundle",
        MagicMock(side_effect=OSError("disk on fire")),
    )
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), file_name="retryable")

    assert result["error_status"] is True
    assert files_in(tmp_path) == []

    monkeypatch.undo()
    retry = _run_block(block, str(tmp_path), file_name="retryable")
    assert retry["error_status"] is False
    assert files_in(tmp_path) == ["retryable.tar.gz"]


def test_publish_never_replaces_an_existing_bundle(tmp_path) -> None:
    # the foreground check is skipped here on purpose, so this exercises the
    # publish itself - the guarantee that survives a lost race
    from inference.core.workflows.core_steps.sinks.roboflow.vision_events_bundle.v1 import (
        _publish_bundle,
    )

    target = tmp_path / "taken.tar.gz"
    target.write_bytes(b"original")

    with pytest.raises(ValueError, match="already exists"):
        _publish_bundle(target_path=str(target), content=b"replacement")

    assert target.read_bytes() == b"original"
    # the temporary file is cleaned up rather than left for a file mover
    assert files_in(tmp_path) == ["taken.tar.gz"]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_publish_falls_back_when_hard_links_are_unavailable(
    tmp_path, monkeypatch
) -> None:
    # FAT-family removable media is how bundles leave an air-gapped network, so
    # the block must still write there rather than failing outright
    monkeypatch.setattr(
        vision_events_bundle.v1.os,
        "link",
        MagicMock(side_effect=OSError(errno.EOPNOTSUPP, "not supported")),
    )
    result = _run_block(_make_block(tmp_path), str(tmp_path), file_name="fallback")

    assert result["error_status"] is False
    assert files_in(tmp_path) == ["fallback.tar.gz"]


def test_fallback_still_refuses_an_existing_bundle(tmp_path, monkeypatch) -> None:
    from inference.core.workflows.core_steps.sinks.roboflow.vision_events_bundle.v1 import (
        _publish_bundle,
    )

    monkeypatch.setattr(
        vision_events_bundle.v1.os,
        "link",
        MagicMock(side_effect=OSError(errno.EOPNOTSUPP, "not supported")),
    )
    target = tmp_path / "taken.tar.gz"
    target.write_bytes(b"original")

    with pytest.raises(ValueError, match="already exists"):
        _publish_bundle(target_path=str(target), content=b"replacement")

    assert target.read_bytes() == b"original"


def test_publish_does_not_fall_back_on_a_real_write_error(
    tmp_path, monkeypatch
) -> None:
    # only "this filesystem cannot do hard links" may reach the fallback; a
    # genuine failure must surface instead of being papered over by a rename
    replace = MagicMock()
    monkeypatch.setattr(
        vision_events_bundle.v1.os,
        "link",
        MagicMock(side_effect=OSError(errno.EIO, "disk on fire")),
    )
    monkeypatch.setattr(vision_events_bundle.v1.os, "replace", replace)

    result = _run_block(_make_block(tmp_path), str(tmp_path), file_name="broken")

    assert result["error_status"] is True
    replace.assert_not_called()
    assert files_in(tmp_path) == []


def test_default_names_do_not_collide_across_events(tmp_path) -> None:
    block = _make_block(tmp_path)

    for _ in range(5):
        assert _run_block(block, str(tmp_path))["error_status"] is False

    assert len(files_in(tmp_path)) == 5


@pytest.mark.parametrize(
    "file_name", ["../escape", "nested/bundle", ".hidden", "", "x" * 201]
)
def test_run_rejects_unsafe_file_name_resolved_from_selector(
    tmp_path, file_name: str
) -> None:
    # selector-resolved values bypass manifest validation, so run() must guard
    block = _make_block(tmp_path)

    with pytest.raises(ValueError):
        _run_block(block, str(tmp_path), file_name=file_name)

    assert files_in(tmp_path) == []


def test_disabled_sink_does_not_validate_file_name(tmp_path) -> None:
    block = _make_block(tmp_path)

    result = _run_block(block, str(tmp_path), disable_sink=True, file_name="../escape")

    assert result["error_status"] is False
    assert result["bundle_path"] == ""


# === Tensor-native sibling smoke coverage ===

_TENSOR_ONLY = pytest.mark.skipif(
    not __import__(
        "inference.core.env", fromlist=["ENABLE_TENSOR_DATA_REPRESENTATION"]
    ).ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


@_TENSOR_ONLY
def test_sync_write_with_native_detections_tensor_native(tmp_path) -> None:
    # end-to-end write through the v1_tensor sibling with a native
    # inference_models.Detections carrier - mirrors
    # test_sync_write_with_both_images_and_predictions
    import torch

    from inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1_tensor import (
        _convert_predictions_to_annotations as tensor_convert_predictions,
    )
    from inference.core.workflows.core_steps.sinks.roboflow.vision_events_bundle.v1_tensor import (
        VisionEventBundleSinkBlockV1 as TensorVisionEventBundleSinkBlockV1,
    )
    from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY
    from inference_models.models.base.object_detection import Detections

    block = TensorVisionEventBundleSinkBlockV1(
        background_tasks=None,
        thread_pool_executor=None,
        allow_access_to_file_system=True,
        allowed_write_directory=None,
        disable_sinks=False,
    )
    detections = Detections(
        xyxy=torch.tensor([[10, 20, 50, 60]], dtype=torch.float32),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "cat"}},
        bboxes_metadata=[{"detection_id": "d0"}],
    )

    result = block.run(
        target_directory=str(tmp_path),
        input_image=_make_workflow_image(),
        output_image=_make_workflow_image(),
        predictions=detections,
        event_type="quality_check",
        custom_metadata={"camera_id": "cam_01"},
        fire_and_forget=False,
        disable_sink=False,
        solution=None,
        cooldown_seconds=0,
        qc_result="pass",
    )

    assert result["error_status"] is False
    assert result["message"] == "Vision event bundle written successfully"
    payload, members = _read_bundle(result["bundle_path"])
    assert payload["bundleFormatVersion"] == BUNDLE_FORMAT_VERSION
    assert payload["eventType"] == "quality_check"
    image_entry = payload["images"][0]
    expected_annotations = tensor_convert_predictions(detections)
    assert image_entry["objectDetections"] == expected_annotations["objectDetections"]
    assert image_entry["objectDetections"][0]["class"] == "cat"
    assert image_entry["objectDetections"][0]["confidence"] == pytest.approx(0.9)
