"""Stability AI blocks in compiled workflows, over real HTTP to a loopback fake.

The real ``StabilityAIClient`` posts multipart requests to a fake Stability AI
service on ``127.0.0.1``; no request leaves the machine and the key is an
invented sentinel. A fake model stands in for segmentation weights::

    image ─▶ v2/instance_segmentation@v1 (fake model: red pixels)
          └▶ v2/stability_ai_inpainting@v1 ─▶ v2/stability_ai_outpainting@v1
    secret input ──────────┘                                  │
    loopback service: records requests, answers PNGs of the requested size
"""

import email.parser
import email.policy
import http.server
import json
import threading
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.generation.stability import (
    StabilityAIClient,
    StabilityAIImageGenerationBlock,
    StabilityAIInpaintingBlock,
    StabilityAIOutpaintingBlock,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.instance_segmentation import (
    InstanceSegmentationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import SECRET_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError

from inference_models.models.base.instance_segmentation import InstanceDetections

KEY = "sk-test-LOOPBACK-91c2"
RED = (255, 0, 0)
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy")
CATALOGUE = Catalogue(
    [
        InstanceSegmentationBlock,
        StabilityAIImageGenerationBlock,
        StabilityAIInpaintingBlock,
        StabilityAIOutpaintingBlock,
    ],
    kinds=[SECRET_KIND],
)


class FakeStabilityService(http.server.BaseHTTPRequestHandler):
    """Answer like Stability AI: a JPEG of the size the request implies."""

    requests: List[Dict[str, Any]] = []
    error: Any = None

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        body = self.rfile.read(int(self.headers["Content-Length"]))
        fields = _multipart(self.headers["Content-Type"], body)
        self.requests.append(
            {
                "path": self.path,
                "authorization": self.headers["authorization"],
                **fields,
            }
        )
        if self.error is not None:
            status, payload = self.error
            self._reply(status, "application/json", json.dumps(payload).encode())
            return

        self._reply(200, "image/jpeg", _answer(self.path, fields))

    def _reply(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


def _multipart(content_type: str, body: bytes) -> Dict[str, Any]:
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body
    )
    fields: Dict[str, Any] = {"data": {}, "files": {}}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if part.get_filename() is None:
            fields["data"][name] = payload.decode()
        else:
            fields["files"][name] = payload

    return fields


def _answer(path: str, fields: Dict[str, Any]) -> bytes:
    if "image" in fields["files"]:
        sent = _decode(fields["files"]["image"])
        height, width = sent.shape[:2]
    else:
        height, width = 32, 48
    data = fields["data"]
    height += int(data.get("up", 0)) + int(data.get("down", 0))
    width += int(data.get("left", 0)) + int(data.get("right", 0))
    blue_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255
    _, encoded = cv2.imencode(".jpg", blue_bgr)

    return encoded.tobytes()


def _decode(jpeg: bytes) -> np.ndarray:
    array = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    return array


class RedObjectModel:
    """Fake instance segmentation: one mask of the image's red pixels."""

    class_names: Sequence[str] = ("object",)

    def __init__(self, request: ModelRequest) -> None:
        self.request = request

    def pre_process(self, images: List[torch.Tensor], input_color_format: str):
        return list(images), [tuple(image.shape[1:]) for image in images]

    def forward(self, batch: Any) -> Any:
        return batch

    def post_process(
        self, images: List[torch.Tensor], sizes: list, **kwargs: Any
    ) -> List[InstanceDetections]:
        predictions = []
        for pixels in images:
            red = torch.tensor(RED, dtype=pixels.dtype)[:, None, None]
            mask = (pixels == red).all(dim=0)
            ys, xs = torch.nonzero(mask, as_tuple=True)
            box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            predictions.append(
                InstanceDetections(
                    xyxy=torch.tensor([box], dtype=torch.float32),
                    class_id=torch.zeros(1, dtype=torch.int64),
                    confidence=torch.ones(1),
                    mask=mask[None],
                )
            )

        return predictions


@pytest.fixture()
def service(monkeypatch: pytest.MonkeyPatch) -> Iterator[Tuple[str, type]]:
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    handler = type("Service", (FakeStabilityService,), {"requests": [], "error": None})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _scene(image_id: str = "street") -> ImageData:
    pixels = np.full((40, 60, 3), 90, dtype=np.uint8)
    pixels[10:30, 20:40] = RED
    image = ImageData.from_numpy_rgb(pixels, image_id=image_id)

    return image


def _session(base_url: str, steps: List[dict], outputs: Dict[str, str], **extra):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "api_key", "kind": ["secret"]},
            *extra.pop("inputs", []),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session(
        resources={
            "models": ModelCache(RedObjectModel),
            "stability_ai_client": StabilityAIClient(base_url=base_url, timeout=10),
        }
    )

    return session


EDIT_STEPS = [
    {
        "type": "v2/instance_segmentation@v1",
        "name": "seg",
        "image": "$inputs.image",
        "model_id": "red-objects",
    },
    {
        "type": "v2/stability_ai_inpainting@v1",
        "name": "inpaint",
        "image": "$inputs.image",
        "segmentation_mask": "$steps.seg.segmentation",
        "prompt": "a blue box",
        "preset": "photographic",
        "api_key": "$inputs.api_key",
    },
    {
        "type": "v2/stability_ai_outpainting@v1",
        "name": "outpaint",
        "image": "$steps.inpaint.image",
        "left": 16,
        "up": 8,
        "api_key": "$inputs.api_key",
    },
]


def test_segment_inpaint_outpaint_over_http_keeps_geometry(service) -> None:
    # given
    base_url, handler = service
    session = _session(
        base_url,
        EDIT_STEPS,
        {"inpainted": "$steps.inpaint.image", "outpainted": "$steps.outpaint.image"},
    )
    image = _scene()

    # when
    result = session.run({"image": [image], "api_key": KEY})
    row = result.rows()[0]
    serialized = result.rows(serialize=True)[0]

    # then
    inpaint, outpaint = handler.requests
    assert inpaint["path"] == "/v2beta/stable-image/edit/inpaint"
    assert inpaint["authorization"] == f"Bearer {KEY}"
    assert inpaint["data"] == {
        "prompt": "a blue box",
        "output_format": "jpeg",
        "preset": "photographic",
    }
    mask = _decode(inpaint["files"]["mask"])
    assert mask[20, 30].min() >= 250 and mask[2, 2].max() <= 5
    assert outpaint["data"] == {
        "output_format": "jpeg",
        "creativity": "0.5",
        "left": "16",
        "up": "8",
    }
    inpainted, outpainted = row["inpainted"], row["outpainted"]
    assert inpainted.image_id == image.image_id
    assert inpainted.tensor_image[2, 20, 30] > 200  # the service's blue
    assert outpainted.size_hw == (48, 76)
    assert outpainted.root.frame_id == image.image_id
    assert outpainted.root.map_xy(16, 8) == (0.0, 0.0)
    assert serialized["outpainted"]["root"]["frame_id"] == image.image_id
    assert KEY not in json.dumps(serialized)


def test_a_service_error_fails_the_step_without_the_key(service) -> None:
    # given: the service echoes the key back, as a careless proxy might
    base_url, handler = service
    handler.error = (401, {"name": "unauthorized", "errors": [f"bad key {KEY}"]})
    session = _session(base_url, EDIT_STEPS, {"image": "$steps.inpaint.image"})

    # when
    with pytest.raises(StepExecutionError) as caught:
        session.run({"image": [_scene()], "api_key": KEY})

    # then
    message = str(caught.value)
    assert "HTTP 401" in message and "unauthorized" in message
    assert "bad key <secret input>" in message
    chain, error = [], caught.value
    while error is not None:
        chain.append(str(error))
        error = error.__cause__ or error.__context__
    assert KEY not in "\n".join(chain)
    assert len(handler.requests) == 1  # outpainting never ran


def test_generation_per_image_with_a_selected_model(service) -> None:
    # given
    base_url, handler = service
    steps = [
        {
            "type": "v2/stability_ai_image_gen@v1",
            "name": "generate",
            "prompt": "$inputs.prompt",
            "model": "$inputs.model",
            "image": "$inputs.image",
            "strength": 0.6,
            "api_key": "$inputs.api_key",
        }
    ]
    session = _session(
        base_url,
        steps,
        {"image": "$steps.generate.image"},
        inputs=[
            {"type": "WorkflowParameter", "name": "prompt", "kind": ["string"]},
            {"type": "WorkflowParameter", "name": "model", "kind": ["string"]},
        ],
    )
    images = [_scene("a"), _scene("b")]

    # when
    rows = session.run(
        {"image": images, "api_key": KEY, "prompt": "winter", "model": "sd3"}
    ).rows()

    # then
    assert [request["path"] for request in handler.requests] == [
        "/v2beta/stable-image/generate/sd3"
    ] * 2
    assert all(
        request["data"]
        == {"prompt": "winter", "output_format": "jpeg", "strength": "0.6"}
        for request in handler.requests
    )
    generated = [row["image"] for row in rows]
    assert [image.size_hw for image in generated] == [(40, 60), (40, 60)]
    assert len({image.image_id for image in generated}) == 2
    assert all(image.root.frame_id == image.image_id for image in generated)


def test_text_only_generation_sends_multipart_without_an_image(service) -> None:
    # given
    base_url, handler = service
    steps = [
        {
            "type": "v2/stability_ai_image_gen@v1",
            "name": "generate",
            "prompt": "a lighthouse",
            "negative_prompt": "people",
            "api_key": "$inputs.api_key",
        }
    ]
    session = _session(base_url, steps, {"image": "$steps.generate.image"})

    # when
    row = session.run({"image": [_scene()], "api_key": KEY}).rows()[0]

    # then
    request = handler.requests[0]
    assert request["path"] == "/v2beta/stable-image/generate/core"
    assert request["data"]["negative_prompt"] == "people"
    assert "image" not in request["files"]
    assert row["image"].size_hw == (32, 48)
