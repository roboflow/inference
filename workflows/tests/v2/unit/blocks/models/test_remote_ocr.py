"""Offline vendor fixtures plus real V2 compilation and downstream wiring.

The response shapes and OCR feature names come from the pinned V1
google_vision_ocr/v1_tensor.py. Detections is the actual inference_models
dataclass, not a replacement carrier. No model or remote service is invoked.
"""

import base64
import copy
import io
import json
import os
import socket
import traceback
from urllib.parse import quote, urlencode

import pytest
import torch
from PIL import Image
from pydantic import Field
from roboflow_workflows.v2.blocks.models.remote_ocr import (
    GoogleVisionOCRBlock,
    GoogleVisionOCRError,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.predictions import (
    NATIVE_KINDS,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Secret,
    SecretUnavailableError,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

FAKE_KEY = "fixture-key/with spaces+?&"
BASE = "http://127.0.0.1:9999/google"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Network is forbidden in Google OCR tests")

    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(socket.socket, "connect", reject)


def _image(image_id="input", *, channels=3, value=91):
    return ImageData.from_tensor(
        torch.full((channels, 12, 20), value, dtype=torch.uint8), image_id=image_id
    )


def _text_block(text="Hello world", *, confidence=None):
    block = {
        "boundingBox": {"vertices": [{}, {"x": 9}, {"x": 9, "y": 8}, {"y": 8}]},
        "paragraphs": [
            {
                "words": [
                    {"symbols": [{"text": c} for c in word]} for word in text.split()
                ]
            }
        ],
    }
    if confidence is not None:
        block["confidence"] = confidence

    return block


def _annotation(text="Hello world", *, confidence=None):
    return {
        "textAnnotations": [{"description": text, "locale": "en"}],
        "fullTextAnnotation": {
            "pages": [{"blocks": [_text_block(text, confidence=confidence)]}]
        },
    }


class _Response:
    def __init__(self, payload, *, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload

        return copy.deepcopy(self.payload)


class _Transport:
    def __init__(self, responses=None):
        self.responses = list(responses or [_Response({"responses": [_annotation()]})])
        self.calls = []

    def post(self, url, *, params, json, timeout):
        self.calls.append(
            {"url": url, "params": params, "json": json, "timeout": timeout}
        )
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response

        return response


def _block(transport=None):
    return GoogleVisionOCRBlock(
        google_vision_transport=transport or _Transport(), google_vision_base_url=BASE
    )


@pytest.mark.parametrize(
    "mode,feature",
    [
        ("text_detection", "TEXT_DETECTION"),
        ("ocr_text_detection", "DOCUMENT_TEXT_DETECTION"),
    ],
)
@pytest.mark.parametrize("hints", [None, [], ["en", "pl"]])
@pytest.mark.parametrize("channels", [1, 3])
def test_constructs_exact_vendor_request_and_lossless_image(
    mode, feature, hints, channels
):
    image = _image(channels=channels)
    original = image.tensor_image.clone()
    storage = image.tensor_image.data_ptr()
    transport = _Transport()
    (result,) = _block(transport).run(
        image=Batch([image]),
        api_key=Secret(FAKE_KEY, name="fixture"),
        ocr_type=mode,
        language_hints=hints,
        timeout=2.5,
    )
    (call,) = transport.calls
    assert call["url"] == BASE + "/v1/images:annotate"
    assert call["params"] == {"key": FAKE_KEY}
    assert call["timeout"] == 2.5
    (request,) = call["json"]["requests"]
    assert request["features"] == [{"type": feature}]
    assert ("imageContext" in request) == (hints is not None)
    if hints is not None:
        assert request["imageContext"] == {"languageHints": hints}
    encoded = base64.b64decode(request["image"]["content"], validate=True)
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    decoded = Image.open(io.BytesIO(encoded))
    assert decoded.mode == ("RGB" if channels == 3 else "L")
    assert decoded.size == (20, 12)
    assert decoded.getpixel((4, 5)) == ((91, 91, 91) if channels == 3 else 91)
    assert image.tensor_image.data_ptr() == storage
    assert torch.equal(image.tensor_image, original)
    assert result["text"] == "Hello world"


def test_preserves_text_block_confidence_geometry_frames_and_fresh_identity():
    annotation = _annotation(confidence=0.72)
    annotation["fullTextAnnotation"]["pages"].append(
        {"blocks": [_text_block("Second line")]}
    )
    transport = _Transport([_Response({"responses": [annotation]})])
    block = _block(transport)
    root = _image("root")
    crop = root.crop((3, 2, 17, 10), image_id="crop").resize((4, 7), image_id="resized")
    first = block.run(image=Batch([crop]), api_key=FAKE_KEY)[0]
    second = block.run(image=Batch([crop]), api_key=FAKE_KEY)[0]
    predictions = first["predictions"]
    assert set(first) == {"text", "language", "predictions"}
    assert first["language"] == "en"
    assert predictions.xyxy.tolist() == [[0, 0, 9, 8], [0, 0, 9, 8]]
    assert predictions.confidence.tolist() == pytest.approx([0.72, 1.0])
    assert predictions.class_id.tolist() == [0, 0]
    assert predictions.class_id.dtype == torch.int64
    assert predictions.xyxy.device == crop.tensor_image.device
    assert predictions.bboxes_metadata == [
        {"class": "Hello world", "text": "Hello world"},
        {"class": "Second line", "text": "Second line"},
    ]
    assert predictions.image_metadata["class_names"] == {0: "text-region"}
    assert predictions.image_metadata["prediction_type"] == "ocr"
    assert frame_of(predictions) == FrameRef.from_image(crop)
    assert (
        row_record(predictions).namespace != row_record(second["predictions"]).namespace
    )
    assert len(transport.calls) == 2  # no prediction cache


@pytest.mark.parametrize(
    "annotation",
    [
        {},
        {"textAnnotations": []},
        {
            "textAnnotations": [],
            "fullTextAnnotation": {"pages": [{"blocks": [_text_block()]}]},
        },
    ],
)
def test_empty_annotations_keep_empty_native_carrier_and_parent(annotation):
    image = _image().crop((1, 1, 10, 9), image_id="crop")
    result = _block(_Transport([_Response({"responses": [annotation]})])).run(
        image=Batch([image]), api_key=FAKE_KEY
    )[0]
    assert (result["text"], result["language"]) == ("", "")
    predictions = result["predictions"]
    assert predictions.xyxy.shape == (0, 4)
    assert predictions.class_id.dtype == torch.int64
    assert predictions.confidence.dtype == torch.float32
    assert predictions.bboxes_metadata is None
    assert frame_of(predictions) == FrameRef.from_image(image)
    assert row_record(predictions).rows == 0
    assert OBJECT_DETECTION_PREDICTION_KIND.validate(predictions)


def test_text_only_annotation_uses_empty_locale_and_no_fabricated_boxes():
    result = _block(
        _Transport(
            [
                _Response(
                    {"responses": [{"textAnnotations": [{"description": "text only"}]}]}
                )
            ]
        )
    ).run(image=Batch([_image()]), api_key=FAKE_KEY)[0]
    assert (result["text"], result["language"]) == ("text only", "")
    assert len(result["predictions"]) == 0


@pytest.mark.parametrize(
    "field,bad",
    [
        ("ocr_type", "unknown"),
        ("ocr_type", []),
        ("language_hints", "en"),
        ("language_hints", ["en", 2]),
        ("language_hints", [""]),
        ("language_hints", [" "]),
        ("timeout", 0),
        ("timeout", -1),
        ("timeout", True),
        ("timeout", float("inf")),
        ("timeout", float("nan")),
        ("api_key", ""),
        ("api_key", "rf_key:account"),
        ("api_key", "rf_key:user:123"),
        ("api_key", 123),
        ("api_key", None),
    ],
)
def test_invalid_second_row_fails_before_any_invoke(field, bad):
    transport = _Transport()
    settings = {
        "api_key": FAKE_KEY,
        "ocr_type": "text_detection",
        "language_hints": None,
        "timeout": 1,
    }
    settings[field] = Batch([settings[field], bad])
    with pytest.raises((ValueError, TypeError)):
        _block(transport).run(image=Batch([_image("a"), _image("b")]), **settings)
    assert transport.calls == []


def test_unavailable_secret_and_misaligned_batch_fail_before_transport():
    transport = _Transport()
    for key in [Batch([FAKE_KEY]), Batch([FAKE_KEY, FAKE_KEY], indices=[(1,), (2,)])]:
        with pytest.raises(ValueError, match="align"):
            _block(transport).run(image=Batch([_image("a"), _image("b")]), api_key=key)
    with pytest.raises(SecretUnavailableError):
        _block(transport).run(
            image=Batch([_image("a"), _image("b")]),
            api_key=Batch([FAKE_KEY, Secret.unavailable("recorded")]),
        )
    assert transport.calls == []


@pytest.mark.parametrize(
    "response,message",
    [
        (TimeoutError(FAKE_KEY), "timed out"),
        (RuntimeError(quote(FAKE_KEY)), "transport"),
        (_Response({"error": {"message": FAKE_KEY}}, status=403), "HTTP status"),
        (_Response(ValueError(FAKE_KEY)), "JSON"),
        (_Response({"error": {"message": FAKE_KEY}}), "envelope"),
        (_Response({"responses": [{"error": {"message": FAKE_KEY}}]}), "image error"),
        (_Response({"responses": []}), "envelope"),
        (_Response({"responses": [{}, {}]}), "envelope"),
        (_Response({"responses": [None]}), "envelope"),
        (_Response([]), "envelope"),
    ],
)
def test_sanitizes_errors_without_exception_context_or_retries(response, message):
    transport = _Transport([response])
    with pytest.raises(GoogleVisionOCRError, match=message) as captured:
        _block(transport).run(
            image=Batch([_image()]), api_key=Secret(FAKE_KEY, name="fixture")
        )
    rendered = "".join(traceback.format_exception(captured.value))
    assert FAKE_KEY not in rendered
    assert quote(FAKE_KEY) not in rendered
    assert urlencode({"key": FAKE_KEY}) not in rendered
    assert captured.value.__context__ is None
    assert captured.value.__cause__ is None
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x["textAnnotations"][0].update(description=123),
        lambda x: x["fullTextAnnotation"]["pages"][0]["blocks"][0].update(
            confidence=float("nan")
        ),
        lambda x: x["fullTextAnnotation"]["pages"][0]["blocks"][0].update(
            confidence=1.2
        ),
        lambda x: x["fullTextAnnotation"]["pages"][0]["blocks"][0][
            "boundingBox"
        ].update(vertices=[]),
        lambda x: x["fullTextAnnotation"]["pages"][0]["blocks"][0]["boundingBox"][
            "vertices"
        ][0].update(x=float("inf")),
        lambda x: x["fullTextAnnotation"]["pages"][0]["blocks"][0]["boundingBox"][
            "vertices"
        ][1].update(x=1e100),
        lambda x: x["fullTextAnnotation"]["pages"][0]["blocks"][0]["boundingBox"][
            "vertices"
        ][2].update(y=-1e100),
        lambda x: x["fullTextAnnotation"]["pages"][0]["blocks"][0]["boundingBox"][
            "vertices"
        ][2].update(y=10**400),
    ],
)
def test_malformed_annotations_fail_without_partial_fabricated_predictions(mutate):
    annotation = _annotation()
    mutate(annotation)
    with pytest.raises(GoogleVisionOCRError, match="annotation") as captured:
        _block(_Transport([_Response({"responses": [annotation]})])).run(
            image=Batch([_image()]), api_key=FAKE_KEY
        )
    assert captured.value.__context__ is None


def test_largest_float32_representable_coordinates_are_kept():
    largest = float(torch.finfo(torch.float32).max)
    annotation = _annotation()
    vertices = annotation["fullTextAnnotation"]["pages"][0]["blocks"][0]["boundingBox"][
        "vertices"
    ]
    vertices[1]["x"] = largest
    vertices[3]["y"] = -largest
    result = _block(_Transport([_Response({"responses": [annotation]})])).run(
        image=Batch([_image()]), api_key=FAKE_KEY
    )[0]
    assert result["predictions"].xyxy.tolist() == [[0, -largest, largest, 8]]


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "ftp://localhost",
        "http://user:fixture@localhost",
        "http://localhost?key=fixture",
        "http://localhost#fixture",
        "http://localhost:bad",
        " http://localhost",
        "http://localhost\n",
    ],
)
def test_rejects_bad_explicit_endpoints_without_echoing_them(url):
    with pytest.raises(ValueError) as captured:
        GoogleVisionOCRBlock(
            google_vision_transport=_Transport(), google_vision_base_url=url
        )
    assert "fixture" not in str(captured.value)


class _ImageForward(Block):
    type = "test/google_image_forward@v1"
    outputs = {"image": Output(IMAGE_KIND, source="image")}
    seen = []

    class Params(BlockParams):
        image: Ref(IMAGE_KIND, batch="always") = Field(
            description="Forwarded images."
        )  # noqa: F821

    def run(self, image):
        self.seen.extend(image.content)
        return [{"image": member} for member in image.content]


class _Downstream(Block):
    type = "test/google_downstream@v1"
    outputs = {"summary": Output(STRING_KIND, source="predictions")}
    seen = []

    class Params(BlockParams):
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND, batch="always") = Field(
            description="Native OCR rows."
        )  # noqa: F821
        text: Ref(STRING_KIND, batch="always") = Field(
            description="Recognized text."
        )  # noqa: F821

    def run(self, predictions, text):
        self.seen.extend(predictions.content)
        return [
            {"summary": f"{value}:{len(prediction)}"}
            for prediction, value in zip(predictions.content, text.content)
        ]


def _plan(*, mode="run", dynamic=False, coordinates="parent", settings=None):
    inputs = [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}]
    step = {
        "type": GoogleVisionOCRBlock.type,
        "name": "ocr",
        "image": "$steps.forward.image",
        "api_key": FAKE_KEY,
    }
    if dynamic:
        for field, kind in [
            ("api_key", "secret"),
            ("ocr_type", "string"),
            ("language_hints", "list_of_values"),
            ("timeout", "float"),
        ]:
            inputs.append({"type": "WorkflowBatchInput", "name": field, "kind": [kind]})
            step[field] = f"$inputs.{field}"
    step.update(settings or {})
    catalogue = Catalogue(
        [_ImageForward, GoogleVisionOCRBlock, _Downstream],
        kinds=[*NATIVE_KINDS, SECRET_KIND],
    )
    return compile_workflow(
        {
            "version": "2.0",
            "inputs": inputs,
            "steps": [
                {
                    "type": _ImageForward.type,
                    "name": "forward",
                    "image": "$inputs.image",
                },
                step,
                {
                    "type": _Downstream.type,
                    "name": "downstream",
                    "predictions": "$steps.ocr.predictions",
                    "text": "$steps.ocr.text",
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": name,
                    "selector": selector,
                    "coordinates_system": coordinates,
                }
                for name, selector in {
                    "text": "$steps.ocr.text",
                    "language": "$steps.ocr.language",
                    "predictions": "$steps.ocr.predictions",
                    "summary": "$steps.downstream.summary",
                }.items()
            ],
        },
        catalogue=catalogue,
        options=CompileOptions(block_execution=mode),
    )


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_compiled_workflow_preserves_order_wiring_and_reuses_only_host_transport(
    mode, dynamic
):
    transport = _Transport(
        [
            _Response({"responses": [_annotation("first")]}),
            _Response({"responses": [{}]}),
        ]
    )
    plan = _plan(mode=mode, dynamic=dynamic)
    assert transport.calls == []
    session = plan.create_session(
        {"google_vision_transport": transport, "google_vision_base_url": BASE}
    )
    images = [_image("a"), _image("b", value=0)]
    _ImageForward.seen.clear()
    _Downstream.seen.clear()
    inputs = {"image": images}
    if dynamic:
        inputs.update(
            api_key=[FAKE_KEY, FAKE_KEY],
            ocr_type=["text_detection", "ocr_text_detection"],
            language_hints=[["en"], []],
            timeout=[1.5, 2.5],
        )
    try:
        rows = session.run(inputs).rows()
        again = session.run(inputs).rows()
    finally:
        session.close()
    assert [r["summary"] for r in rows] == ["first:1", ":0"]
    assert [frame_of(r["predictions"]).image_id for r in rows] == ["a", "b"]
    assert _ImageForward.seen[0] is images[0]
    assert _ImageForward.seen[0].tensor_image is images[0].tensor_image
    assert _Downstream.seen[0] is rows[0]["predictions"]
    assert len(transport.calls) == 4
    assert (
        row_record(rows[0]["predictions"]).namespace
        != row_record(again[0]["predictions"]).namespace
    )
    if dynamic:
        assert [
            c["json"]["requests"][0]["features"][0]["type"] for c in transport.calls[:2]
        ] == ["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"]
        assert [c["timeout"] for c in transport.calls[:2]] == [1.5, 2.5]


def test_compiled_root_coordinates_restore_crop_resize_geometry():
    transport = _Transport()
    crop = (
        _image("root")
        .crop((3, 2, 17, 10), image_id="crop")
        .resize((4, 7), image_id="resized")
    )
    session = _plan(coordinates="root").create_session(
        {"google_vision_transport": transport, "google_vision_base_url": BASE}
    )
    try:
        (result,) = session.run({"image": [crop]}).rows()
    finally:
        session.close()
    assert result["predictions"].xyxy.tolist() == [[3, 2, 21, 18]]
    assert frame_of(result["predictions"]).image_id == "root"


@pytest.mark.parametrize(
    "field,kind,bad",
    [
        ("ocr_type", "string", "invalid"),
        ("language_hints", "list_of_values", [1]),
        ("timeout", "float", 0),
        ("api_key", "secret", "rf_key:account"),
    ],
)
def test_compiled_invalid_selected_parameters_fail_before_any_invoke(field, kind, bad):
    transport = _Transport()
    session = _plan(dynamic=True).create_session(
        {"google_vision_transport": transport, "google_vision_base_url": BASE}
    )
    inputs = {
        "image": [_image("a"), _image("b")],
        "api_key": [FAKE_KEY, FAKE_KEY],
        "ocr_type": ["text_detection"] * 2,
        "language_hints": [[], []],
        "timeout": [1, 1],
    }
    inputs[field][1] = bad
    try:
        with pytest.raises((StepExecutionError, ValueError)):
            session.run(inputs)
    finally:
        session.close()
    assert transport.calls == []


def test_empty_direct_and_compiled_batches_do_not_invoke():
    transport = _Transport()
    assert _block(transport).run(image=Batch([]), api_key=FAKE_KEY) == []
    session = _plan().create_session(
        {"google_vision_transport": transport, "google_vision_base_url": BASE}
    )
    try:
        assert session.run({"image": []}).rows() == []
    finally:
        session.close()
    assert transport.calls == []


@pytest.mark.skipif(
    os.environ.get("GOOGLE_VISION_CHECK_REQUESTS") != "1",
    reason="opt-in real requests package check; no remote Google service exercised",
)
def test_installed_requests_session_prepares_real_request_with_injected_adapter():
    requests = pytest.importorskip("requests")
    prepared = []

    class Adapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            prepared.append((request, kwargs))
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({"responses": [_annotation()]}).encode()
            return response

        def close(self):
            pass

    transport = requests.Session()
    transport.trust_env = False
    transport.mount("http://", Adapter())
    try:
        result = _block(transport).run(image=Batch([_image()]), api_key=FAKE_KEY)[0]
    finally:
        transport.close()
    request, kwargs = prepared[0]
    assert request.method == "POST"
    assert request.url == BASE + "/v1/images:annotate?" + urlencode({"key": FAKE_KEY})
    assert request.headers["Content-Type"] == "application/json"
    assert kwargs["timeout"] == 30
    assert json.loads(request.body)["requests"][0]["features"] == [
        {"type": "TEXT_DETECTION"}
    ]
    assert result["text"] == "Hello world"


@pytest.mark.parametrize(
    "settings",
    [
        {"ocr_type": "invalid"},
        {"language_hints": [1]},
        {"api_key": None},
        {"api_key": 12},
        {"not_a_parameter": True},
        {"timeout": True},
        {"timeout": "1.0"},
        {"timeout": 0},
        {"timeout": float("inf")},
    ],
)
def test_malformed_literal_definitions_fail_compilation(settings):
    with pytest.raises(ParamsValidationError):
        _plan(settings=settings)


def test_real_source_drives_ocr_and_downstream_until_end_of_stream():
    from roboflow_workflows.v2.engine.sources import Emission, Source, SourceOutput

    class FixtureImages(Source):
        type = "test/google_fixture_images@v1"
        outputs = {"image": SourceOutput(IMAGE_KIND)}

        def __init__(self, *, fixture_images):
            self.images = fixture_images

        def open(self):
            self.remaining = iter(self.images)

        def read(self):
            image = next(self.remaining, None)
            return None if image is None else Emission({"image": image})

    transport = _Transport(
        [
            _Response({"responses": [_annotation("source text")]}),
            _Response({"responses": [{}]}),
        ]
    )
    catalogue = Catalogue(
        [GoogleVisionOCRBlock, _Downstream],
        sources=[FixtureImages],
        kinds=[*NATIVE_KINDS, SECRET_KIND],
    )
    plan = compile_workflow(
        {
            "version": "2.0",
            "inputs": [],
            "sources": [{"type": FixtureImages.type, "name": "frames"}],
            "steps": [
                {
                    "type": GoogleVisionOCRBlock.type,
                    "name": "ocr",
                    "image": "$sources.frames.image",
                    "api_key": FAKE_KEY,
                },
                {
                    "type": _Downstream.type,
                    "name": "downstream",
                    "predictions": "$steps.ocr.predictions",
                    "text": "$steps.ocr.text",
                },
            ],
            "outputs": [
                {
                    "type": "OutputGroup",
                    "name": "results",
                    "anchor": "$sources.frames.image",
                    "outputs": [
                        {
                            "type": "JsonField",
                            "name": "summary",
                            "selector": "$steps.downstream.summary",
                        },
                        {
                            "type": "JsonField",
                            "name": "predictions",
                            "selector": "$steps.ocr.predictions",
                        },
                    ],
                }
            ],
        },
        catalogue=catalogue,
    )
    images = [_image("source-a"), _image("source-b")]
    session = plan.create_session(
        {
            "fixture_images": images,
            "google_vision_transport": transport,
            "google_vision_base_url": BASE,
        }
    )
    delivered = []
    run = session.start(handlers={"results": delivered.append})
    try:
        assert run.wait(10)
    finally:
        if not run.done:
            run.cancel()
            run.wait(10)
        session.close()
    assert [pulse.outputs.data["summary"] for pulse in delivered] == [
        "source text:1",
        ":0",
    ]
    assert [
        frame_of(pulse.outputs.data["predictions"]).image_id for pulse in delivered
    ] == ["source-a", "source-b"]
    assert len(transport.calls) == 2
