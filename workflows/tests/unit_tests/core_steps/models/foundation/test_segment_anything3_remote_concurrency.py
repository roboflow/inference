"""Exercise SAM3 blocks through the real SDK, replacing only HTTP transport."""

import importlib
import json
import weakref
from contextvars import ContextVar
from threading import Barrier, Event, Lock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from requests import Response
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

from inference_sdk.http.errors import HTTPCallErrorError
from inference_sdk.http.utils import executors

VARIANTS = ["v1", "v1_tensor", "v2", "v2_tensor", "v3", "v3_tensor"]
REQUEST_CONTEXT = ContextVar("sam3_test_request_context", default=None)


def _block(monkeypatch, variant, concurrency):
    module = importlib.import_module(
        "roboflow_workflows.core_steps.models.foundation.segment_anything3." + variant
    )
    monkeypatch.setattr(module, "SAM3_EXEC_MODE", "local")
    monkeypatch.setattr(module, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
    monkeypatch.setattr(module, "HOSTED_CORE_MODEL_URL", "http://sam3.invalid")
    monkeypatch.setattr(
        module, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", concurrency
    )
    version = variant.split("_")[0][-1]
    block = getattr(module, f"SegmentAnything3BlockV{version}")(
        model_manager=MagicMock(),
        api_key="test-key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    kwargs = (
        {"threshold": 0.3}
        if version == "1"
        else {
            "confidence": 0.3,
            "per_class_confidence": [0.4],
            "apply_nms": True,
            "nms_iou_threshold": 0.7,
        }
    )
    if version == "3":
        kwargs.update(class_mapping={"box": "product"}, output_format="rle")
    return block, kwargs


def _images(count):
    return [
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id=f"frame-{i}"),
            numpy_image=np.full((16, 16, 3), i, dtype=np.uint8),
        )
        for i in range(count)
    ]


def _response(index, fmt="polygon", status=200):
    if fmt == "rle":
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:6, index + 1 : index + 4] = 1
        masks = mask_utils.encode(np.asfortranarray(mask))
        masks["counts"] = masks["counts"].decode("ascii")
    else:
        masks = [[[index + 1, 2], [index + 3, 2], [index + 3, 5], [index + 1, 5]]]
    payload = {
        "prompt_results": [
            {
                "prompt_index": 0,
                "predictions": [{"masks": masks, "confidence": 0.9, "format": fmt}],
            }
        ],
        "time": 0.01,
    }
    response = Response()
    response.status_code = status
    response.url = "http://sam3.invalid/sam3/concept_segment"
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(
        payload if status == 200 else {"detail": "failed"}
    ).encode()
    return response


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("concurrency", [1, 2, 3, 8])
def test_remote_requests_overlap_within_limit_and_keep_frame_order(
    monkeypatch, variant, concurrency
):
    block, kwargs = _block(monkeypatch, variant, concurrency)
    images = _images(6)
    indices = {image.base64_image: i for i, image in enumerate(images)}
    barriers = [
        Barrier(min(concurrency, len(images) - start))
        for start in range(0, len(images), concurrency)
    ]
    release = [Event() for _ in images]
    for start in range(0, len(images), concurrency):
        release[min(start + concurrency, len(images)) - 1].set()
    lock = Lock()
    active = peak = 0
    completed, requests, contexts = [], [], []

    def transport(request_data, request_method):
        nonlocal active, peak
        # A SAM3 request must contain one image, even for a workflow batch.
        assert isinstance(request_data.payload["image"], dict)
        idx = indices[request_data.payload["image"]["value"]]
        with lock:
            active += 1
            peak = max(peak, active)
            requests.append(request_data)
            contexts.append(REQUEST_CONTEXT.get())
        try:
            barriers[idx // concurrency].wait(timeout=5)
            assert release[idx].wait(timeout=5)
            with lock:
                completed.append(idx)
            if idx % concurrency:
                release[idx - 1].set()
            return _response(idx, request_data.payload["format"])
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(executors, "make_request", transport)
    token = REQUEST_CONTEXT.set("preview-context")
    try:
        result = block.run(
            images=images, model_id="sam3/sam3_final", class_names=["box"], **kwargs
        )
    finally:
        REQUEST_CONTEXT.reset(token)

    assert len(requests) == len(result) == len(images)
    assert peak == min(concurrency, len(images))
    assert active == 0
    assert contexts == ["preview-context"] * len(images)
    expected_completion = [
        idx
        for start in range(0, len(images), concurrency)
        for idx in reversed(range(start, min(start + concurrency, len(images))))
    ]
    assert completed == expected_completion
    for idx, item in enumerate(result):
        detections = item["predictions"]
        assert float(detections.xyxy[0, 0]) == idx + 1
        if "tensor" not in variant:
            assert detections.data["parent_id"][0] == f"frame-{idx}"
            assert detections.data["class_name"][0] == (
                "product" if variant == "v3" else "box"
            )
    for request in requests:
        assert request.payload["model_id"] == "sam3/sam3_final"
        assert request.payload["prompts"][0]["text"] == "box"
        if not variant.startswith("v1"):
            assert request.payload["prompts"][0]["output_prob_thresh"] == 0.4
            assert request.payload["nms_iou_threshold"] == 0.7


def test_cpu_v3_polygon_output_keeps_class_mapping_and_masks(monkeypatch):
    block, kwargs = _block(monkeypatch, "v3", 2)
    kwargs["output_format"] = "polygons"
    requests = []

    def transport(request_data, request_method):
        requests.append(request_data)
        return _response(0, request_data.payload["format"])

    monkeypatch.setattr(executors, "make_request", transport)
    result = block.run(
        images=_images(2), model_id="sam3/sam3_final", class_names=["box"], **kwargs
    )

    assert len(result) == len(requests) == 2
    assert all(r.payload["format"] == "polygon" for r in requests)
    for idx, item in enumerate(result):
        detections = item["predictions"]
        assert detections.mask.shape == (1, 16, 16)
        assert detections.data["class_name"].tolist() == ["product"]
        assert detections.data["parent_id"].tolist() == [f"frame-{idx}"]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("count", [0, 1])
def test_remote_empty_and_single_image_batches(monkeypatch, variant, count):
    block, kwargs = _block(monkeypatch, variant, 8)
    calls = []

    def transport(request_data, request_method):
        calls.append(request_data)
        return _response(0, request_data.payload["format"])

    monkeypatch.setattr(executors, "make_request", transport)
    result = block.run(
        images=_images(count), model_id="sam3/sam3_final", class_names=["box"], **kwargs
    )
    assert len(result) == len(calls) == count


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("failed_group", [0, 1])
def test_remote_http_failure_stops_later_groups(
    monkeypatch, variant, concurrency, failed_group
):
    block, kwargs = _block(monkeypatch, variant, concurrency)
    images = _images(7)
    first = images[failed_group * concurrency].base64_image
    requested = []

    def transport(request_data, request_method):
        requested.append(request_data.payload["image"]["value"])
        status = 400 if request_data.payload["image"]["value"] == first else 200
        return _response(0, request_data.payload["format"], status=status)

    monkeypatch.setattr(executors, "make_request", transport)
    with pytest.raises(HTTPCallErrorError):
        block.run(
            images=images, model_id="sam3/sam3_final", class_names=["box"], **kwargs
        )

    expected_count = (failed_group + 1) * concurrency
    assert len(requested) == expected_count
    assert set(requested) == {image.base64_image for image in images[:expected_count]}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("concurrency", [1, 2])
def test_remote_releases_large_polygon_responses_before_next_group(
    monkeypatch, variant, concurrency
):
    block, kwargs = _block(monkeypatch, variant, concurrency)
    if variant == "v3":
        kwargs["output_format"] = "polygons"
    images = _images(7)
    indices = {image.base64_image: i for i, image in enumerate(images)}
    http_refs, parsed_refs = [], []
    lock = Lock()

    class TrackedPayload(dict):
        pass

    class TrackedResponse(Response):
        def json(self, **kwargs):
            payload = TrackedPayload(super().json(**kwargs))
            parsed_refs.append(weakref.ref(payload))
            return payload

    def transport(request_data, request_method):
        idx = indices[request_data.payload["image"]["value"]]
        with lock:
            if idx >= concurrency:
                # No raw or parsed responses from previous groups remain alive.
                previous_count = (idx // concurrency) * concurrency
                assert all(ref() is None for ref in http_refs[:previous_count])
                assert all(ref() is None for ref in parsed_refs[:previous_count])
            response = TrackedResponse()
            response.status_code = 200
            payload = _response(idx).json()
            # A detailed polygon stresses both the HTTP body and decoded lists.
            angles = np.linspace(0, 2 * np.pi, 8192, endpoint=False)
            polygon = np.column_stack(
                (8 + 6 * np.cos(angles), 8 + 6 * np.sin(angles))
            ).tolist()
            payload["prompt_results"][0]["predictions"][0]["masks"] = [polygon]
            response._content = json.dumps(payload).encode()
            http_refs.append(weakref.ref(response))
            return response

    monkeypatch.setattr(executors, "make_request", transport)
    result = block.run(
        images=images, model_id="sam3/sam3_final", class_names=["box"], **kwargs
    )

    assert len(result) == len(http_refs) == len(parsed_refs) == len(images)
    assert all(ref() is None for ref in http_refs + parsed_refs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_remote_response_count_mismatch_fails_instead_of_dropping_frames(
    monkeypatch, variant
):
    block, kwargs = _block(monkeypatch, variant, 2)
    monkeypatch.setattr(executors, "make_request", lambda **_: _response(0))
    from inference_sdk import InferenceHTTPClient

    monkeypatch.setattr(
        InferenceHTTPClient, "sam3_concept_segment", lambda *_, **__: {}
    )
    with pytest.raises(ValueError, match="1 responses for 2 images"):
        block.run(
            images=_images(2), model_id="sam3/sam3_final", class_names=["box"], **kwargs
        )
