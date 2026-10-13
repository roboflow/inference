"""Unit tests of ``v2/visual_search@v1`` and ``v2/visual_search_classifier@v1``.

Fake ports prove wiring and V1 output rules only; the host adapter and its
HTTP requests are tested in the inference package. Every credential is
invented.
"""

import base64
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.blocks.roboflow_platform.visual_search import (
    CLASSIFIER_FIELDS,
    VisualSearchBlock,
    VisualSearchClassifierBlock,
    parse_visual_search_classification,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef, classification_parent
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError
from roboflow_workflows.v2.engine.reactions.testing import block_call

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)

API_KEY = "rf-test-KEY-r1-search-0002"
CANDIDATE = {
    "id": "img-1",
    "url": "https://images.test/img-1.jpg",
    "name": "one.jpg",
    "user_metadata": {"line": "a"},
    "tags": ["ok"],
    "width": 4,
    "height": 2,
    "aspectRatio": 2.0,
}
CANDIDATE_PIXELS = np.full((2, 4, 3), (10, 20, 30), dtype=np.uint8)


class FakeSearchPort:
    """Records calls and answers with fixed results; never reveals the key."""

    def __init__(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        *,
        error: Optional[Exception] = None,
        response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.response = {"results": results or []} if response is None else response
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def get_workspace(self, *, api_key: Secret) -> str:
        self.calls.append({"operation": "get_workspace", "api_key": api_key})
        return "key-workspace"

    def search_project_images(self, **arguments: Any) -> Dict[str, Any]:
        self.calls.append({"operation": "search", **arguments})
        if self.error is not None:
            raise self.error
        return self.response

    def load_candidate_image(self, url: str) -> np.ndarray:
        self.calls.append({"operation": "load", "url": url})
        return CANDIDATE_PIXELS.copy()

    def operations(self) -> List[str]:
        return [call["operation"] for call in self.calls]


def _image(height: int = 6, width: int = 8, channels: int = 3) -> ImageData:
    generator = torch.Generator().manual_seed(height * 1000 + width)
    pixels = torch.randint(
        0, 256, (channels, height, width), dtype=torch.uint8, generator=generator
    )
    image = ImageData.from_tensor(pixels, image_id="query")

    return image


def _decoded_query(call: Dict[str, Any]) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(call["image_base64"]), dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

    return decoded


def _search(port: FakeSearchPort, *, wanted=None, **params: Any) -> Dict[str, Any]:
    block = VisualSearchBlock(visual_search_port=port, api_key=API_KEY)
    arguments = {
        "image": _image(),
        "target_project": "refs",
        "top_k": 3,
        "workspace": "ws",
        **params,
    }
    with block_call(block, wanted=wanted):
        result = block.run(**arguments)

    return result


def _classify(port: FakeSearchPort, *, wanted=None, **params: Any) -> Dict[str, Any]:
    block = VisualSearchClassifierBlock(visual_search_port=port, api_key=API_KEY)
    arguments = {
        "image": _image(),
        "target_project": "refs",
        "top_k": 1,
        "workspace": "ws",
        "max_image_size": 224,
        **params,
    }
    with block_call(block, wanted=wanted):
        result = block.run(**arguments)

    return result


def test_search_returns_formatted_candidates_and_the_best_image() -> None:
    # given
    port = FakeSearchPort([CANDIDATE, {"id": "img-2", "filename": "two.png"}])

    # when
    result = _search(port)

    # then
    best = {
        "image_id": "img-1",
        "image_url": "https://images.test/img-1.jpg",
        "name": "one.jpg",
        "filename": "one.jpg",
        "metadata": {"line": "a"},
        "tags": ["ok"],
        "width": 4,
        "height": 2,
        "aspect_ratio": 2.0,
    }
    image = result.pop("best_candidate_image")
    assert result == {
        "candidate_found": True,
        "best_candidate": best,
        "candidates": [
            best,
            {
                "image_id": "img-2",
                "image_url": None,
                "name": "two.png",
                "filename": "two.png",
                "metadata": {},
                "tags": [],
                "width": None,
                "height": None,
                "aspect_ratio": None,
            },
        ],
        "error_status": False,
        "message": "Visual search completed.",
        "best_candidate_metadata": {"line": "a"},
        "best_candidate_tags": ["ok"],
    }
    assert image.image_id == "img-1"
    assert torch.equal(
        image.tensor_image, torch.from_numpy(CANDIDATE_PIXELS).permute(2, 0, 1)
    )
    search = port.calls[0]
    assert (search["workspace"], search["project"], search["limit"]) == (
        "ws",
        "refs",
        3,
    )
    assert "fields" not in search
    assert isinstance(search["api_key"], Secret)
    assert port.operations() == ["search", "load"]


def test_query_is_the_v1_jpeg_of_the_bgr_pixels() -> None:
    # given
    port = FakeSearchPort()
    image = _image(height=5, width=7)
    bgr = image.tensor_image.permute(1, 2, 0).numpy()[:, :, ::-1]
    _, expected = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # when
    _search(port, image=image)

    # then
    sent = base64.b64decode(port.calls[0]["image_base64"])
    assert sent == expected.tobytes()


def test_grayscale_query_is_sent_as_a_grayscale_jpeg() -> None:
    # given
    port = FakeSearchPort()

    # when
    _search(port, image=_image(height=4, width=4, channels=1))

    # then
    assert _decoded_query(port.calls[0]).shape == (4, 4)


def test_no_candidates_gives_v1_empty_outputs() -> None:
    # given
    port = FakeSearchPort([])

    # when
    result = _search(port)

    # then
    assert result == {
        "candidate_found": False,
        "best_candidate": {},
        "candidates": [],
        "error_status": False,
        "message": "No visually similar images found.",
        "best_candidate_image": None,
        "best_candidate_metadata": {},
        "best_candidate_tags": [],
    }


def test_an_undemanded_candidate_image_is_not_downloaded() -> None:
    # given
    port = FakeSearchPort([CANDIDATE])
    wanted = set(spec_of(VisualSearchBlock).outputs) - {"best_candidate_image"}

    # when
    result = _search(port, wanted=wanted)

    # then
    assert result["best_candidate_image"] is None
    assert port.operations() == ["search"]


def test_a_candidate_without_url_has_no_image() -> None:
    # given
    port = FakeSearchPort([{"id": "img-3"}])

    # when
    result = _search(port)

    # then
    assert result["candidate_found"] is True
    assert result["best_candidate_image"] is None
    assert port.operations() == ["search"]


@pytest.mark.parametrize(
    "port",
    [
        FakeSearchPort(error=PlatformRequestError("boom 500", error_type="x.Error")),
        FakeSearchPort(response={"results": ["not-a-dict"]}),
    ],
    ids=["request", "malformed"],
)
def test_search_failures_are_reported_in_outputs(port: FakeSearchPort) -> None:
    # when
    result = _search(port)

    # then
    assert result["error_status"] is True
    assert result["message"].startswith("Visual search failed: ")
    assert result["candidate_found"] is False
    assert result["best_candidate_image"] is None


@pytest.mark.parametrize(
    "block_class", [VisualSearchBlock, VisualSearchClassifierBlock]
)
def test_missing_port_key_or_replayed_key_fail_the_step(block_class: type) -> None:
    # given
    params = {
        "image": _image(),
        "target_project": "refs",
        "top_k": 1,
        "workspace": "ws",
    }
    if block_class is VisualSearchClassifierBlock:
        params["max_image_size"] = 224
    port = FakeSearchPort()

    # when / then
    with block_call(block_class):
        with pytest.raises(PlatformUnavailableError, match="'visual_search_port'"):
            block_class(api_key=API_KEY).run(**params)
        with pytest.raises(ValueError, match="without a Roboflow API key"):
            block_class(visual_search_port=port).run(**params)
        with pytest.raises(SecretUnavailableError):
            block_class(
                visual_search_port=port, api_key=Secret.unavailable("RF_KEY")
            ).run(**params)
    assert port.calls == []


def test_classifier_builds_a_single_label_prediction_from_the_best_candidate() -> None:
    # given
    candidate = {
        **CANDIDATE,
        "score": "1.64",
        "classification": {"class": "cat", "class_id": 2},
    }
    port = FakeSearchPort([candidate])

    # when
    result = _classify(port)

    # then
    prediction = result["predictions"]
    assert isinstance(prediction, ClassificationPrediction)
    assert prediction.class_id.tolist() == [2]
    assert prediction.confidence.dtype == torch.float64
    assert prediction.confidence.tolist() == [[0.0, 0.0, 0.82]]
    metadata = prediction.images_metadata[0]
    assert metadata["class_names"] == {0: "0", 1: "1", 2: "cat"}
    assert metadata["prediction_type"] == "classification"
    assert metadata["inference_id"] == result["inference_id"]
    assert result["visual_search_score"] == 1.64
    assert result["best_candidate"]["score"] == 1.64
    assert result["best_candidate"]["classification"] == candidate["classification"]
    assert (result["class_found"], result["error_status"]) == (True, False)
    assert result["message"] == "Visual search classification completed."
    assert result["best_candidate_image"].image_id == "img-1"
    assert port.calls[0]["fields"] == CLASSIFIER_FIELDS
    assert port.operations() == ["search", "load"]


def test_classifier_builds_a_multi_label_prediction_from_labels() -> None:
    # given
    labels = ["dog", {"class": "cat", "class_id": 3}, "dog"]
    port = FakeSearchPort([{**CANDIDATE, "score": 4.0, "labels": labels}])

    # when
    result = _classify(port)

    # then
    prediction = result["predictions"]
    assert isinstance(prediction, MultiLabelClassificationPrediction)
    assert prediction.class_ids.tolist() == [0, 3]
    assert prediction.confidence.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert prediction.image_metadata["class_names"] == {
        0: "dog",
        1: "1",
        2: "2",
        3: "cat",
    }


def test_classifier_resolves_the_workspace_only_without_one() -> None:
    # given
    port = FakeSearchPort([])

    # when
    _classify(port, workspace=None)
    _classify(port, workspace="ws")

    # then
    assert port.operations() == ["get_workspace", "search", "search"]
    assert port.calls[1]["workspace"] == "key-workspace"


def test_classifier_reports_a_candidate_without_classification() -> None:
    # given
    port = FakeSearchPort([{**CANDIDATE, "score": float("nan")}])

    # when
    result = _classify(port)

    # then
    assert result["predictions"] is None
    assert (result["candidate_found"], result["class_found"]) == (True, False)
    assert result["error_status"] is True
    assert result["visual_search_score"] is None
    assert result["message"] == (
        "Best visual search candidate does not include classification labels "
        "or annotations."
    )


def test_classifier_without_score_has_zero_confidence() -> None:
    # given
    port = FakeSearchPort([{**CANDIDATE, "annotations": [{"classes": ["cat"]}]}])

    # when
    result = _classify(port)

    # then
    assert result["predictions"].confidence.tolist() == [[0.0]]
    assert result["visual_search_score"] is None


def test_classifier_failure_keeps_a_fresh_inference_id() -> None:
    # given
    port = FakeSearchPort(error=PlatformRequestError("refused", error_type="x.Error"))

    # when
    first = _classify(port)
    second = _classify(port)

    # then
    assert first["message"] == "Visual search classification failed: refused"
    assert first["error_status"] is True
    assert first["predictions"] is None
    assert first["inference_id"] != second["inference_id"]


@pytest.mark.parametrize(
    ("size_hw", "max_image_size", "expected_hw"),
    [
        ((200, 400), 224, (112, 224)),
        ((400, 200), 224, (224, 112)),
        ((50, 60), 224, (50, 60)),
    ],
)
def test_classifier_downscales_the_query_as_v1(
    size_hw: tuple, max_image_size: int, expected_hw: tuple
) -> None:
    # given
    port = FakeSearchPort()

    # when
    _classify(port, image=_image(*size_hw), max_image_size=max_image_size)

    # then
    assert _decoded_query(port.calls[0]).shape[:2] == expected_hw


def test_classification_of_a_crop_keeps_its_parent_row() -> None:
    # given
    row = RowRef(namespace="inf-parent", id=3)
    port = FakeSearchPort([{**CANDIDATE, "classification": {"class": "cat"}}])

    # when
    result = _classify(port, image=_image().with_parent_row(row))

    # then
    assert classification_parent(result["predictions"]) == row
    assert result["predictions"].class_id.tolist() == [0]


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ({"classification": {"class": "cat", "class_id": "x"}}, [("cat", 0)]),
        (
            {
                "classification": {
                    "predicted_classes": ["a", "b", "a"],
                    "predictions": {"a": {"class_id": 1}, "b": {"class_id": 1}},
                }
            },
            [("a", 1), ("b", 2)],
        ),
        ({"classification": {"predicted_classes": []}, "labels": ["x"]}, [("x", 0)]),
        (
            {"labels": [{"class": ""}, 5], "annotations": [{"classes": ["y"]}]},
            [("y", 0)],
        ),
        ({"labels": "cat"}, None),
        ({}, None),
    ],
)
def test_parser_follows_v1_source_order_and_id_rules(
    candidate: Dict[str, Any], expected: Optional[list]
) -> None:
    # when
    parsed = parse_visual_search_classification(candidate)

    # then
    if expected is None:
        assert parsed is None
    else:
        assert [
            (entry["class"], entry["class_id"]) for entry in parsed["classes"]
        ] == expected


@pytest.mark.parametrize(
    ("block_class", "field", "value"),
    [
        (VisualSearchBlock, "top_k", 0),
        (VisualSearchBlock, "top_k", 51),
        (VisualSearchClassifierBlock, "max_image_size", 0),
    ],
)
def test_out_of_range_literals_are_rejected(
    block_class: type, field: str, value: int
) -> None:
    # given
    step = {
        "image": "$inputs.image",
        "target_project": "refs",
        "workspace": "ws",
        field: value,
    }

    # when / then
    with pytest.raises(ParamsValidationError):
        spec_of(block_class).validate_params(step)


@pytest.mark.parametrize(
    "params", [{"top_k": 0}, {"top_k": 51}, {"max_image_size": 0}], ids=str
)
def test_out_of_range_runtime_values_fail_before_any_request(
    params: Dict[str, Any],
) -> None:
    # given
    port = FakeSearchPort([CANDIDATE])

    # when
    with pytest.raises(ValueError, match="must be in"):
        _classify(port, **params)

    # then
    assert port.calls == []


def test_workspace_is_required_only_for_plain_search() -> None:
    # given
    step = {"image": "$inputs.image", "target_project": "refs"}

    # when
    params = spec_of(VisualSearchClassifierBlock).validate_params(step)

    # then
    assert (params.workspace, params.top_k, params.max_image_size) == (None, 1, 224)
    with pytest.raises(ParamsValidationError):
        spec_of(VisualSearchBlock).validate_params(step)
