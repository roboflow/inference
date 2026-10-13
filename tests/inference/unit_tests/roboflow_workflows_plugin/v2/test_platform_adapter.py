"""Loopback wire checks of ``RoboflowPlatformAdapter`` and V1 parity checks.

The real adapter calls the real ``inference.core.roboflow_api`` functions;
``API_BASE_URL`` points at an ephemeral HTTP server on ``127.0.0.1``. These
tests prove request shape, key handling, response parsing and failure mapping
of this host code. They do not prove how the Roboflow cloud answers: the
fixture's responses are invented, as is every credential::

    adapter ─▶ roboflow_api ─▶ requests ─▶ http://127.0.0.1:<ephemeral>/...
                                              └─ records method, path, query,
                                                 headers and JSON body

Proxy variables are removed, so requests go straight to the loopback server.
"""

import base64
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.utils.images import downscale_image_keeping_aspect_ratio
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.roboflow_platform import visual_search
from roboflow_workflows.v2.blocks.roboflow_platform.custom_metadata import (
    RoboflowCustomMetadataBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError
from roboflow_workflows.v2.engine.compilation import compile_workflow

import inference.core.roboflow_api as roboflow_api
from inference.core import exceptions
from inference.core.utils import image_utils, url_utils
from inference.roboflow_workflows_plugin.integrations.visual_search import (
    helpers as v1_helpers,
)
from inference.roboflow_workflows_plugin.integrations.visual_search_classifier import (
    classification_annotations as v1_annotations,
)
from inference.roboflow_workflows_plugin.integrations.visual_search_classifier import (
    v1 as v1_classifier,
)
from inference.roboflow_workflows_plugin.v2.platform_adapter import (
    RoboflowPlatformAdapter,
)

API_KEY = "rftestKEYr1loopback93d2"
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
V1_DEFAULT_SEARCH_FIELDS = [
    "id",
    "name",
    "filename",
    "url",
    "user_metadata",
    "tags",
    "width",
    "height",
    "aspectRatio",
]

Route = Tuple[int, bytes, str]


class LoopbackPlatform:
    """An HTTP server on 127.0.0.1 answering from a route table."""

    def __init__(self) -> None:
        self.routes: Dict[Tuple[str, str], Route] = {}
        self.requests: List[Dict[str, Any]] = []
        platform = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                platform._answer(self, body=None)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                platform._answer(self, body=self.rfile.read(length))

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def route(
        self, method: str, path: str, status: int = 200, body: Any = None
    ) -> None:
        if isinstance(body, bytes):
            self.routes[(method, path)] = (status, body, "image/png")
        else:
            payload = json.dumps({} if body is None else body).encode()
            self.routes[(method, path)] = (status, payload, "application/json")

    def paths(self) -> List[str]:
        return [f"{request['method']} {request['path']}" for request in self.requests]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _answer(
        self, handler: BaseHTTPRequestHandler, *, body: Optional[bytes]
    ) -> None:
        url = urlsplit(handler.path)
        self.requests.append(
            {
                "method": handler.command,
                "path": url.path,
                "query": {
                    key: values[0] for key, values in parse_qs(url.query).items()
                },
                "headers": dict(handler.headers.items()),
                "json": json.loads(body) if body else None,
            }
        )
        status, payload, content_type = self.routes.get(
            (handler.command, url.path), (404, b"{}", "application/json")
        )
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)


@pytest.fixture
def loopback(monkeypatch: pytest.MonkeyPatch) -> Iterator[LoopbackPlatform]:
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    platform = LoopbackPlatform()
    monkeypatch.setattr(roboflow_api, "API_BASE_URL", platform.base_url)
    monkeypatch.setattr(
        roboflow_api, "API_PROXY_BASE_URL", f"{platform.base_url}/proxy"
    )
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", None)
    monkeypatch.setattr(url_utils, "SECURE_GATEWAY", None)
    roboflow_api.get_roboflow_workspace.cache_clear()
    yield platform
    roboflow_api.get_roboflow_workspace.cache_clear()
    platform.close()


def _key() -> Secret:
    return Secret(API_KEY, name="ROBOFLOW_API_KEY")


def _png(bgr: np.ndarray) -> bytes:
    _, encoded = cv2.imencode(".png", bgr)

    return encoded.tobytes()


def _assert_clean(error: PlatformRequestError) -> None:
    assert API_KEY not in f"{error} {error!r} {error.error_type}"
    assert error.__cause__ is None
    assert error.__context__ is None


def _qualified(error_class: type) -> str:
    return f"{error_class.__module__}.{error_class.__qualname__}"


def _closed_loopback_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    return port


# --- request shape and response parsing --------------------------------------


def test_workspace_lookup_sends_the_key_as_v1_and_parses_the_workspace(
    loopback: LoopbackPlatform,
) -> None:
    # given
    loopback.route("GET", "/", body={"workspace": "loop-ws"})

    # when
    workspace = RoboflowPlatformAdapter().get_workspace(api_key=_key())

    # then
    assert workspace == "loop-ws"
    (request,) = loopback.requests
    assert request["query"] == {"api_key": API_KEY, "nocache": "true"}
    assert request["headers"][roboflow_api.ROBOFLOW_INFERENCE_VERSION_HEADER]


def test_custom_metadata_posts_the_v1_body(loopback: LoopbackPlatform) -> None:
    # given
    loopback.route("POST", "/loop-ws/inference-stats/metadata")
    inference_ids = ("inf-1", "inf-2")

    # when
    RoboflowPlatformAdapter().add_custom_metadata(
        api_key=_key(),
        workspace_id="loop-ws",
        inference_ids=inference_ids,
        field_name="location",
        field_value="toronto",
    )

    # then
    (request,) = loopback.requests
    assert request["query"] == {"api_key": API_KEY, "nocache": "true"}
    assert request["json"] == {
        "data": [
            {
                "inference_ids": ["inf-1", "inf-2"],
                "field_name": "location",
                "field_value": "toronto",
            }
        ]
    }
    assert inference_ids == ("inf-1", "inf-2")


@pytest.mark.parametrize(
    ("fields", "expected_fields"),
    [(None, V1_DEFAULT_SEARCH_FIELDS), (("id", "score"), ["id", "score"])],
    ids=["default", "explicit"],
)
def test_search_posts_the_v1_payload_and_returns_the_json(
    loopback: LoopbackPlatform, fields: Optional[tuple], expected_fields: list
) -> None:
    # given
    loopback.route("POST", "/loop-ws/refs/search", body={"results": [{"id": "a"}]})

    # when
    response = RoboflowPlatformAdapter().search_project_images(
        api_key=_key(),
        workspace="loop-ws",
        project="refs",
        image_base64="QUJD",
        limit=3,
        fields=fields,
    )

    # then
    assert response == {"results": [{"id": "a"}]}
    (request,) = loopback.requests
    assert request["query"] == {"api_key": API_KEY}
    assert request["json"] == {
        "image_base64": "QUJD",
        "limit": 3,
        "fields": expected_fields,
    }


def test_extra_headers_apply_to_every_request_without_accumulating(
    loopback: LoopbackPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    extra_headers = json.dumps({"X-Extra": "e", "X-Allow-Chunked": "false"})
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", extra_headers)
    loopback.route("POST", "/loop-ws/refs/search", body={"results": []})
    adapter = RoboflowPlatformAdapter()
    arguments = dict(workspace="loop-ws", project="refs", image_base64="QUJD", limit=1)

    # when
    adapter.search_project_images(api_key=_key(), **arguments)
    adapter.search_project_images(api_key=_key(), **arguments)

    # then
    first, second = (request["headers"] for request in loopback.requests)
    assert first == second
    assert first["X-Extra"] == "e"
    assert first["X-Allow-Chunked"] == "true"
    assert roboflow_api.ROBOFLOW_API_EXTRA_HEADERS == extra_headers


# --- failure semantics and key safety -----------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "error_class", "text"),
    [
        (401, {}, "RoboflowAPINotAuthorizedError", "check API key"),
        (403, {}, "RoboflowAPIForbiddenError", "required scopes"),
        (404, {}, "RoboflowAPINotNotFoundError", ""),
        (500, {}, "RoboflowAPIUnsuccessfulRequestError", "response code: 500"),
        (200, b"not json", "MalformedRoboflowAPIResponseError", "Could not decode"),
    ],
    ids=["401", "403", "404", "500", "malformed"],
)
def test_search_failures_map_to_clean_platform_errors_without_retry(
    loopback: LoopbackPlatform, status: int, body: Any, error_class: str, text: str
) -> None:
    # given
    loopback.route("POST", "/loop-ws/refs/search", status=status, body=body)

    # when
    with pytest.raises(PlatformRequestError) as raised:
        RoboflowPlatformAdapter().search_project_images(
            api_key=_key(),
            workspace="loop-ws",
            project="refs",
            image_base64="A",
            limit=1,
        )

    # then
    expected_class = getattr(exceptions, error_class)
    assert raised.value.error_type == _qualified(expected_class)
    assert str(raised.value).startswith("project image search failed:")
    assert API_KEY not in str(raised.value)
    _assert_clean(raised.value)
    assert loopback.paths() == ["POST /loop-ws/refs/search"]


def test_a_connection_failure_is_cleaned_of_the_key_its_chain_carried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        roboflow_api, "API_BASE_URL", f"http://127.0.0.1:{_closed_loopback_port()}"
    )
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(url_utils, "SECURE_GATEWAY", None)
    arguments = dict(workspace="w", project="p", image_base64="A", limit=1)

    # when
    with pytest.raises(Exception) as original:
        roboflow_api.search_project_images_at_roboflow(api_key=API_KEY, **arguments)
    with pytest.raises(PlatformRequestError) as raised:
        RoboflowPlatformAdapter().search_project_images(api_key=_key(), **arguments)

    # then
    chain, error = [], original.value
    while error is not None:
        chain.append(str(error))
        error = error.__cause__ or error.__context__
    assert API_KEY in "\n".join(chain)
    assert raised.value.error_type.endswith("RoboflowAPIConnectionError")
    _assert_clean(raised.value)


def test_offline_mode_keeps_v1_semantics_and_sends_nothing(
    loopback: LoopbackPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", True)
    adapter = RoboflowPlatformAdapter()

    # when
    added = adapter.add_custom_metadata(
        api_key=_key(),
        workspace_id="w",
        inference_ids=["i"],
        field_name="f",
        field_value="v",
    )
    with pytest.raises(
        PlatformRequestError, match="project image search failed: connection failed"
    ) as raised:
        adapter.search_project_images(
            api_key=_key(), workspace="w", project="p", image_base64="A", limit=1
        )

    # then
    assert added is None
    _assert_clean(raised.value)
    assert loopback.requests == []


def test_an_unavailable_key_is_refused_before_any_request(
    loopback: LoopbackPlatform,
) -> None:
    # when
    with pytest.raises(SecretUnavailableError):
        RoboflowPlatformAdapter().get_workspace(api_key=Secret.unavailable("RF"))

    # then
    assert loopback.requests == []


def test_candidate_images_obey_the_server_url_guards(
    loopback: LoopbackPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255
    loopback.route("GET", "/img.png", body=_png(bgr))
    url = f"{loopback.base_url}/img.png"
    adapter = RoboflowPlatformAdapter()

    # when
    with pytest.raises(PlatformRequestError) as refused:
        adapter.load_candidate_image(url)
    requests_while_guarded = list(loopback.requests)
    monkeypatch.setattr(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", True)
    monkeypatch.setattr(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
    rgb = adapter.load_candidate_image(url)

    # then
    assert refused.value.error_type.endswith("InputImageLoadError")
    assert requests_while_guarded == []
    assert rgb.shape == (2, 3, 3)
    assert rgb[0, 0].tolist() == [0, 0, 255]


# --- blocks through the compiler, adapter and loopback server -----------------


def _tagging_plan():
    catalogue = create_catalogue().with_blocks(
        [
            RoboflowCustomMetadataBlock,
            visual_search.VisualSearchBlock,
            visual_search.VisualSearchClassifierBlock,
        ]
    )
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "v2/visual_search_classifier@v1",
                "name": "classify",
                "image": "$inputs.image",
                "target_project": "refs",
            },
            {
                "type": "v2/roboflow_custom_metadata@v1",
                "name": "tag",
                "predictions": "$steps.classify.predictions",
                "inference_id": "$steps.classify.inference_id",
                "field_name": "source",
                "field_value": "search",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.classify.predictions",
            },
            {
                "type": "JsonField",
                "name": "inference_id",
                "selector": "$steps.classify.inference_id",
            },
            {"type": "JsonField", "name": "tag", "selector": "$steps.tag.*"},
        ],
    }
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


def test_compiled_workflow_searches_classifies_and_tags_over_loopback(
    loopback: LoopbackPlatform, caplog: pytest.LogCaptureFixture
) -> None:
    # given
    loopback.route("GET", "/", body={"workspace": "loop-ws"})
    loopback.route(
        "POST",
        "/loop-ws/refs/search",
        body={"results": [{"id": "r1", "score": 1.2, "labels": ["apple"]}]},
    )
    loopback.route("POST", "/loop-ws/inference-stats/metadata")
    adapter = RoboflowPlatformAdapter()
    session = _tagging_plan().create_session(
        resources={
            "visual_search_port": adapter,
            "custom_metadata_port": adapter,
            "api_key": _key(),
        }
    )
    image = ImageData.from_tensor(torch.full((3, 8, 8), 40, dtype=torch.uint8))

    # when
    # INFO, not DEBUG: urllib3's DEBUG records print request URLs, and V1's
    # roboflow_api sends the key as a query parameter (an inherited limit).
    with caplog.at_level(logging.INFO):
        (row,) = session.run({"image": [image]}).rows(serialize=True)

    # then
    assert loopback.paths() == [
        "GET /",
        "POST /loop-ws/refs/search",
        "POST /loop-ws/inference-stats/metadata",
    ]
    search, tag = loopback.requests[1], loopback.requests[2]
    assert search["json"]["fields"] == list(visual_search.CLASSIFIER_FIELDS)
    assert tag["json"]["data"][0]["inference_ids"] == [row["inference_id"]]
    assert row["tag"] == {
        "error_status": False,
        "message": "Custom metadata upload was successful",
    }
    assert API_KEY not in json.dumps(row, default=str) + caplog.text


def test_compiled_workflow_reports_a_rejected_tag_without_the_key(
    loopback: LoopbackPlatform, caplog: pytest.LogCaptureFixture
) -> None:
    # given
    loopback.route("GET", "/", body={"workspace": "loop-ws"})
    loopback.route(
        "POST",
        "/loop-ws/refs/search",
        body={"results": [{"id": "r1", "labels": ["a"]}]},
    )
    loopback.route("POST", "/loop-ws/inference-stats/metadata", status=403)
    adapter = RoboflowPlatformAdapter()
    session = _tagging_plan().create_session(
        resources={
            "visual_search_port": adapter,
            "custom_metadata_port": adapter,
            "api_key": API_KEY,
        }
    )
    image = ImageData.from_tensor(torch.full((3, 8, 8), 40, dtype=torch.uint8))

    # when
    with caplog.at_level(logging.INFO):
        (row,) = session.run({"image": [image]}).rows(serialize=True)

    # then
    assert row["tag"]["error_status"] is True
    forbidden = _qualified(exceptions.RoboflowAPIForbiddenError)
    assert f"Error type: <class '{forbidden}'>" in row["tag"]["message"]
    assert API_KEY not in json.dumps(row, default=str) + caplog.text


# --- latest V1 parity of pure logic ---------------------------------------------

PARSER_CORPUS = [
    {"classification": {"class": "cat", "class_id": 4}},
    {
        "classification": {
            "predicted_classes": ["a", "b"],
            "predictions": {"b": {"class_id": 0}},
        }
    },
    {"classification": {"predicted_classes": ["a", "a", 3, ""]}},
    {"classification": "cat", "labels": ["x", {"class": "y", "class_id": 0}, "x"]},
    {"labels": [{"class": "p", "class_id": "2"}, {"class": "q", "class_id": 2}]},
    {"labels": [], "annotations": [{"classes": ["m", "n"]}, "bad", {"classes": "z"}]},
    {"annotations": [{"classes": [None, "k"]}]},
    {"labels": "cat"},
    {},
]


@pytest.mark.parametrize("candidate", PARSER_CORPUS)
def test_classification_parser_matches_latest_v1(candidate: Dict[str, Any]) -> None:
    # when
    v2 = visual_search.parse_visual_search_classification(candidate)
    v1 = v1_annotations.parse_visual_search_classification(candidate=candidate)

    # then
    assert v2 == v1


def test_parser_differs_from_v1_only_for_a_class_without_id() -> None:
    # given
    candidate = {"classification": {"class": "cat"}}

    # when
    v2 = visual_search.parse_visual_search_classification(candidate)
    v1 = v1_annotations.parse_visual_search_classification(candidate=candidate)

    # then
    assert v1["classes"] == [{"class": "cat", "class_id": -1}]
    assert v2["classes"] == [{"class": "cat", "class_id": 0}]


@pytest.mark.parametrize(
    "candidate",
    [
        {
            "id": "a",
            "url": "u",
            "name": "n",
            "user_metadata": {"k": 1},
            "tags": ["t"],
            "width": 1,
            "height": 2,
            "aspectRatio": 0.5,
            "score": "1.5",
            "labels": ["c"],
        },
        {"filename": "f.jpg", "score": float("inf"), "classification": {"class": "c"}},
        {"name": "only-name", "user_metadata": None, "tags": None, "score": None},
    ],
)
def test_candidate_formatting_and_scores_match_latest_v1(
    candidate: Dict[str, Any],
) -> None:
    # when / then
    assert visual_search._format_candidate(candidate) == (
        v1_helpers.format_visual_search_candidate(candidate=candidate)
    )
    assert visual_search._format_classifier_candidate(candidate) == (
        v1_classifier._format_visual_search_classifier_candidate(candidate=candidate)
    )
    for score in (candidate.get("score"), -1.0, 0.5, 3.0, "x"):
        assert visual_search._confidence(score) == (
            v1_classifier._normalise_visual_search_confidence(score=score)
        )


@pytest.mark.parametrize("size_hw", [(200, 400), (400, 200), (300, 300), (50, 60)])
def test_query_downscale_and_jpeg_match_latest_v1(size_hw: Tuple[int, int]) -> None:
    # given
    generator = torch.Generator().manual_seed(size_hw[0])
    pixels = torch.randint(
        0, 256, (3, *size_hw), dtype=torch.uint8, generator=generator
    )
    image = ImageData.from_tensor(pixels)
    bgr = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy()[:, :, ::-1])
    v1_query = downscale_image_keeping_aspect_ratio(image=bgr, desired_size=(224, 224))
    _, v1_jpeg = cv2.imencode(".jpg", v1_query, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # when
    v2_query = visual_search._query_base64(image, max_side=224)

    # then
    assert base64.b64decode(v2_query) == v1_jpeg.tobytes()
