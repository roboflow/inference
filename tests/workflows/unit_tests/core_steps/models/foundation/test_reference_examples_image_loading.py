"""Reference examples through an engine set up like the HTTP API's, with the server's codec.

A block cannot see `prevent_local_images_loading`, and the codec reads local files by default, so
only a run through both shows that an example image cannot reach the disk. Engines of one process
share the examples they fetched, which only runs through several engines show.
"""

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List
from unittest import mock

import cv2
import numpy as np
import pytest
from roboflow_workflows.core_steps.common.vlm_reference_examples import (
    REFERENCE_PARTS_CACHE_TTL_SECONDS,
    SHARED_REFERENCE_PARTS_CACHE,
)

from inference.core.interfaces.workflows_image_codec import bind_image_codec
from inference.core.utils import image_utils
from inference.core.workflows.errors import RuntimeInputError, WorkflowSyntaxError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.prototypes.image_codec import reset_image_codec

OPENAI = "roboflow_core/open_ai@v7"
GEMINI = "roboflow_core/google_gemini@v6"
MODEL_VERSIONS = {OPENAI: "gpt-6.1-terra", GEMINI: "gemini-4-argon"}
REQUESTS_SEAMS = {
    OPENAI: "roboflow_workflows.core_steps.models.foundation.openai.v7.execute_openai_requests",
    GEMINI: "roboflow_workflows.core_steps.models.foundation.google_gemini.v6.execute_gemini_requests",
}
BLOCKS = pytest.mark.parametrize("block", [GEMINI, OPENAI])
ANNOTATIONS = [{"x": 100, "y": 50, "width": 60, "height": 40, "class": "cat"}]
URL = "https://example.com/labeled.jpg"


@pytest.fixture(autouse=True)
def _reset_codec():
    reset_image_codec()
    yield
    reset_image_codec()


@pytest.fixture(autouse=True)
def _empty_reference_parts_cache():
    SHARED_REFERENCE_PARTS_CACHE.clear()
    yield
    SHARED_REFERENCE_PARTS_CACHE.clear()


def _engine(block: str, reference_examples: Any) -> ExecutionEngine:
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "reference_examples"},
        ],
        "steps": [
            {
                "type": block,
                "name": "vlm",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["cat"],
                "model_version": MODEL_VERSIONS[block],
                "api_key": "provider-key",
                "reference_examples": reference_examples,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "output", "selector": "$steps.vlm.output"}
        ],
    }
    init_parameters = {
        "workflows_core.api_key": "roboflow-key",
        "workflows_core.background_tasks": None,
    }
    bind_image_codec(init_parameters)
    engine = ExecutionEngine.init(
        workflow_definition=definition,
        init_parameters=init_parameters,
        prevent_local_images_loading=True,
    )
    return engine


def _target_image() -> dict:
    _, encoded = cv2.imencode(".png", np.full((100, 200, 3), 128, dtype=np.uint8))
    value = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {"type": "base64", "value": value}


def _reference_images(block: str, prompt: dict) -> List[str]:
    if block == OPENAI:
        images = [
            part["image_url"].split(",", 1)[1]
            for part in prompt["input"][0]["content"]
            if part.get("type") == "input_image"
        ]
    else:
        images = [
            part["inline_data"]["data"]
            for part in prompt["contents"]["parts"]
            if "inline_data" in part
        ]
    return images[:-1]


def _blue_of_first_reference(block: str, requests: mock.Mock, call: int) -> int:
    prompts = requests.call_args_list[call].kwargs
    prompt = (prompts.get("openai_prompts") or prompts.get("gemini_prompts"))[0]
    encoded = base64.b64decode(_reference_images(block, prompt)[0])
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    return int(image[2, 2, 0])


@BLOCKS
@pytest.mark.parametrize("delivery", ["literal", "selector"])
@pytest.mark.parametrize(
    "as_image",
    [lambda path: path, lambda path: {"type": "file", "value": path}],
    ids=["bare path", "file dict"],
)
def test_reference_image_never_reads_a_local_file(
    tmp_path, block: str, delivery: str, as_image
) -> None:
    # given
    path = tmp_path / "server-file.png"
    cv2.imwrite(str(path), np.full((100, 200, 3), 255, dtype=np.uint8))
    examples = [{"image": as_image(str(path)), "annotations": ANNOTATIONS}]
    literal = delivery == "literal"
    runtime = {} if literal else {"reference_examples": examples}

    # when
    with mock.patch.object(
        image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", True
    ), mock.patch(REQUESTS_SEAMS[block]) as requests:
        engine = _engine(block, examples if literal else "$inputs.reference_examples")
        with pytest.raises(RuntimeInputError, match="reference_examples"):
            engine.run(runtime_parameters={"image": _target_image(), **runtime})

    # then
    requests.assert_not_called()


@BLOCKS
def test_selector_inside_a_literal_example_is_refused_when_compiling(
    block: str,
) -> None:
    # given
    examples = [{"image": "$inputs.reference_image", "annotations": ANNOTATIONS}]

    # when
    with pytest.raises(WorkflowSyntaxError) as error:
        _engine(block, examples)

    # then
    assert "selectors are not resolved inside an example" in str(
        error.value.inner_error
    )


@BLOCKS
def test_reference_image_fetched_once_for_all_engines_until_it_expires(
    block: str,
) -> None:
    # given
    examples = [{"image": {"type": "url", "value": URL}, "annotations": ANNOTATIONS}]
    images = [
        np.full((100, 200, 3), 30, dtype=np.uint8),
        np.full((100, 200, 3), 220, dtype=np.uint8),
    ]
    parameters = {"image": _target_image()}
    now = [0.0]

    # when
    with mock.patch.object(
        image_utils, "load_image_from_url", side_effect=images
    ) as fetch, mock.patch(
        REQUESTS_SEAMS[block], return_value=[("[]", 1, 1)]
    ) as requests, mock.patch.object(
        SHARED_REFERENCE_PARTS_CACHE, "_clock", new=lambda: now[0]
    ):
        first_job = _engine(block, examples)
        first_job.run(runtime_parameters=parameters)
        first_job.run(runtime_parameters=parameters)
        second_job = _engine(block, examples)
        second_job.run(runtime_parameters=parameters)
        now[0] = REFERENCE_PARTS_CACHE_TTL_SECONDS
        first_job.run(runtime_parameters=parameters)

    # then
    assert fetch.call_count == 2
    assert abs(_blue_of_first_reference(block, requests, 0) - 30) < 8
    assert abs(_blue_of_first_reference(block, requests, 1) - 30) < 8
    assert abs(_blue_of_first_reference(block, requests, 2) - 30) < 8
    assert abs(_blue_of_first_reference(block, requests, 3) - 220) < 8


@BLOCKS
def test_engines_running_at_once_fetch_a_reference_image_once(block: str) -> None:
    # given
    examples = [{"image": {"type": "url", "value": URL}, "annotations": ANNOTATIONS}]
    engines = [_engine(block, examples) for _ in range(4)]
    parameters = {"image": _target_image()}
    start = threading.Barrier(len(engines))

    def fetch_slowly(*args: Any, **kwargs: Any) -> np.ndarray:
        # Long enough for the other engines to ask for the examples meanwhile.
        time.sleep(0.2)
        return np.full((100, 200, 3), 30, dtype=np.uint8)

    def run(engine: ExecutionEngine) -> None:
        start.wait(timeout=10)
        engine.run(runtime_parameters=parameters)

    # when
    with mock.patch.object(
        image_utils, "load_image_from_url", side_effect=fetch_slowly
    ) as fetch, mock.patch(
        REQUESTS_SEAMS[block], return_value=[("[]", 1, 1)]
    ) as requests, ThreadPoolExecutor(
        max_workers=len(engines)
    ) as pool:
        list(pool.map(run, engines))

    # then
    assert fetch.call_count == 1
    assert len(requests.call_args_list) == len(engines)
    for call in range(len(engines)):
        assert abs(_blue_of_first_reference(block, requests, call) - 30) < 8


@BLOCKS
def test_engine_retries_expired_download_without_waiting_for_stalled_engine(
    block: str,
) -> None:
    started, release = threading.Event(), threading.Event()
    request_lock = threading.Lock()
    fetched = []
    now = [0.0]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with request_lock:
                fetched.append(self.path)
                number = len(fetched)
            if number == 1:
                started.set()
                assert release.wait(timeout=15)
            color = 30 if number == 1 else 220
            _, png = cv2.imencode(".png", np.full((100, 200, 3), color, dtype=np.uint8))
            payload = png.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    examples = [
        {
            "image": {
                "type": "url",
                "value": f"http://127.0.0.1:{server.server_port}/reference.png",
            },
            "annotations": ANNOTATIONS,
        }
    ]
    parameters = {"image": _target_image()}

    try:
        # Only the destination policy is relaxed for the local test server. The
        # actual HTTP transport, image codec, rendering and engine all execute.
        with mock.patch.multiple(
            image_utils,
            OFFLINE_MODE=False,
            ALLOW_URL_INPUT=True,
            ALLOW_NON_HTTPS_URL_INPUT=True,
            ALLOW_URL_INPUT_WITHOUT_FQDN=True,
            ALLOW_URL_TO_NON_GLOBAL_ADDRESSES=True,
            WHITELISTED_DESTINATIONS_FOR_URL_INPUT=None,
            BLACKLISTED_DESTINATIONS_FOR_URL_INPUT=None,
            VALIDATE_IMAGE_URL_REDIRECTS=True,
        ), mock.patch.object(
            SHARED_REFERENCE_PARTS_CACHE, "_clock", new=lambda: now[0]
        ), mock.patch(
            REQUESTS_SEAMS[block], return_value=[("[]", 1, 1)]
        ) as requests, ThreadPoolExecutor(
            max_workers=2
        ) as pool:
            first_engine, second_engine = _engine(block, examples), _engine(
                block, examples
            )
            first = pool.submit(first_engine.run, runtime_parameters=parameters)
            try:
                assert started.wait(timeout=5)
                now[0] = 61.0
                second = pool.submit(second_engine.run, runtime_parameters=parameters)
                second.result(timeout=5)
                assert not first.done()
                assert len(fetched) == 2
                assert abs(_blue_of_first_reference(block, requests, 0) - 220) < 8
            finally:
                release.set()
            first.result(timeout=5)
            assert abs(_blue_of_first_reference(block, requests, 1) - 30) < 8
            _engine(block, examples).run(runtime_parameters=parameters)
            assert abs(_blue_of_first_reference(block, requests, 2) - 220) < 8
            assert len(fetched) == 2
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


@BLOCKS
def test_numpy_reference_image_runs_through_a_selector(block: str) -> None:
    # given
    pixels = np.full((100, 200, 3), 30, dtype=np.uint8)
    examples = [
        {"image": {"type": "numpy_object", "value": pixels}, "annotations": ANNOTATIONS}
    ]
    parameters = {"image": _target_image(), "reference_examples": examples}

    # when
    with mock.patch(REQUESTS_SEAMS[block], return_value=[("[]", 1, 1)]) as requests:
        engine = _engine(block, "$inputs.reference_examples")
        engine.run(runtime_parameters=parameters)
        pixels[:] = 220
        engine.run(runtime_parameters=parameters)

    # then
    assert abs(_blue_of_first_reference(block, requests, 0) - 30) < 8
    assert abs(_blue_of_first_reference(block, requests, 1) - 220) < 8
