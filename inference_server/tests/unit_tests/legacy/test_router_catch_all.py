import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests.unit_tests.legacy.conftest import FakeGateway


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buf, format="JPEG")
    return buf.getvalue()


def _gw():
    det = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    return FakeGateway(
        predictions={("ds/1", "infer"): det},
        model_info={"ds/1": {"class_names": ["cat"]}},
    )


def test_catch_all_raw_body_base64(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = _gw()
    r = legacy_client(gw).post(
        "/ds/1?api_key=k&confidence=50&overlap=30",
        content=base64.b64encode(_jpeg()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 200 and r.json()["predictions"][0]["class"] == "cat"
    params = next(c for c in gw.calls if c[0] == "infer")[3]
    assert params["confidence"] == 0.5 and params["iou_threshold"] == 0.3


def test_catch_all_multipart_file(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    r = legacy_client(_gw()).post(
        "/ds/1?api_key=k", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}
    )
    assert r.status_code == 200


def test_catch_all_missing_content_type_is_400(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    r = legacy_client(_gw()).post("/ds/1?api_key=k")
    assert r.status_code == 400


def test_catch_all_does_not_shadow_two_segment_named_routes(legacy_client, fake_stat):
    c = legacy_client(_gw())
    assert c.get("/v2/server/health").status_code == 200
    assert c.get("/model/registry").status_code == 200
    assert c.post("/model/clear").status_code == 200
    assert c.get("/clear_cache").status_code == 200


IMAGE_URL = "https://example.com/a.jpg"
IMAGE_LOAD_PREFIX = "Could not load input image. Cause: "
MISSING_FILE_PART = {
    "message": f"{IMAGE_LOAD_PREFIX}Expected image to be send in part named 'file' "
    "of multipart/form-data request"
}
CONTENT_TYPE_MISSING = {"message": "Content-Type header not provided with request."}
CONTENT_TYPE_INVALID = {"message": "Invalid Content-Type header provided with request."}
EMPTY_PAYLOAD = {"message": f"{IMAGE_LOAD_PREFIX}Empty image payload."}
RAW_BYTES = {
    "message": f"{IMAGE_LOAD_PREFIX}Invalid base64 input: the image payload contains "
    "raw bytes instead of a base64-encoded string. Please base64-encode the image "
    "before sending."
}
FORM_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


@pytest.mark.parametrize(
    "query_image,request_kwargs,status,body,fetched",
    [
        pytest.param(False, {}, 400, CONTENT_TYPE_MISSING, [], id="no-header"),
        pytest.param(True, {}, 200, None, [[IMAGE_URL]], id="no-header-query-image"),
        pytest.param(
            False,
            {"files": {"other": ("a.jpg", _jpeg(), "image/jpeg")}},
            400,
            MISSING_FILE_PART,
            [],
            id="multipart-without-file",
        ),
        pytest.param(
            True,
            {"files": {"other": ("a.jpg", _jpeg(), "image/jpeg")}},
            400,
            MISSING_FILE_PART,
            [],
            id="multipart-without-file-query-image",
        ),
        pytest.param(
            False,
            {"files": {"file": ("a.jpg", _jpeg(), "image/jpeg")}},
            200,
            None,
            [],
            id="multipart-with-file",
        ),
        pytest.param(
            True,
            {"files": {"file": ("a.jpg", b"not an image", "image/jpeg")}},
            200,
            None,
            [[IMAGE_URL]],
            id="multipart-with-file-query-image",
        ),
        pytest.param(
            False,
            {
                "files": {"other": ("a.jpg", _jpeg(), "image/jpeg")},
                "data": {"file": "text"},
            },
            400,
            CONTENT_TYPE_INVALID,
            [],
            id="multipart-with-text-file-field",
        ),
        pytest.param(
            True,
            {
                "files": {"other": ("a.jpg", _jpeg(), "image/jpeg")},
                "data": {"file": "text"},
            },
            200,
            None,
            [[IMAGE_URL]],
            id="multipart-with-text-file-field-query-image",
        ),
        pytest.param(
            False,
            {"content": base64.b64encode(_jpeg()), "headers": FORM_HEADERS},
            200,
            None,
            [],
            id="body",
        ),
        pytest.param(
            True,
            {"content": b"not an image", "headers": FORM_HEADERS},
            200,
            None,
            [[IMAGE_URL]],
            id="body-query-image",
        ),
        pytest.param(
            False,
            {"content": b"", "headers": FORM_HEADERS},
            400,
            EMPTY_PAYLOAD,
            [],
            id="empty-body",
        ),
        pytest.param(
            False,
            {"content": _jpeg(), "headers": FORM_HEADERS},
            400,
            RAW_BYTES,
            [],
            id="raw-image-bytes-body",
        ),
    ],
)
def test_catch_all_image_source_follows_legacy_order(
    legacy_client,
    fake_stat,
    monkeypatch,
    query_image,
    request_kwargs,
    status,
    body,
    fetched,
):
    seen = []

    async def _fetch(urls, destination_policy=None):
        seen.append(urls)
        return [_jpeg() for _ in urls], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)
    fake_stat["ds/1"] = ("object-detection", "infer")
    path = "/ds/1?api_key=k"
    if query_image:
        path = f"{path}&image={IMAGE_URL}"

    response = legacy_client(_gw()).post(path, **request_kwargs)

    assert response.status_code == status, response.text
    if body is not None:
        assert response.json() == body
    assert seen == fetched


FORM_CONTENT_TYPE = {"Content-Type": "application/x-www-form-urlencoded"}


def test_catch_all_image_type_is_matched_in_any_letter_case(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _gw()

    response = legacy_client(gateway).post(
        "/ds/1?api_key=k&image_type=BASE64",
        content=base64.b64encode(_jpeg()),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 200, response.text
    assert [call[4] for call in gateway.calls if call[0] == "infer"] == [_jpeg()]


@pytest.mark.parametrize("image_type", ["file", "FILE"])
def test_catch_all_loads_local_file_named_in_the_body(
    legacy_client, fake_stat, tmp_path, monkeypatch, image_type
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _gw()
    client = legacy_client(gateway)
    monkeypatch.setattr(
        "inference_server.legacy.common.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM",
        True,
    )
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(_jpeg())

    response = client.post(
        f"/ds/1?api_key=k&image_type={image_type}",
        content=str(image_path).encode(),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 200, response.text
    assert response.json()["image"] == {"width": 8, "height": 6}
    assert [call[4] for call in gateway.calls if call[0] == "infer"] == [_jpeg()]


def test_catch_all_refuses_local_file_when_loading_is_disabled(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_gw())
    monkeypatch.setattr(
        "inference_server.legacy.common.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM",
        False,
    )
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(_jpeg())

    response = client.post(
        "/ds/1?api_key=k&image_type=file",
        content=str(image_path).encode(),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 400
    assert response.json() == {
        "message": f"{IMAGE_LOAD_PREFIX}Loading images from local filesystem is "
        "disabled."
    }


@pytest.mark.parametrize("target", ["missing.jpg", "notes.txt", "."])
def test_catch_all_refuses_unreadable_local_file_with_one_answer(
    legacy_client, fake_stat, tmp_path, monkeypatch, target
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_gw())
    monkeypatch.setattr(
        "inference_server.legacy.common.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM",
        True,
    )
    (tmp_path / "notes.txt").write_text("hello")

    response = client.post(
        "/ds/1?api_key=k&image_type=file",
        content=str(tmp_path / target).encode(),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 400
    assert response.json() == {
        "message": f"{IMAGE_LOAD_PREFIX}Could not load image from the local file."
    }


def _post_catch_all(legacy_client, fake_stat, task_type, prediction, info, query=""):
    fake_stat["ds/1"] = (task_type, "infer")
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): prediction},
        model_info={"ds/1": dict(info, actions={"infer": {}})},
    )
    response = legacy_client(gateway).post(
        f"/ds/1?api_key=k{query}",
        content=base64.b64encode(_jpeg()),
        headers=FORM_CONTENT_TYPE,
    )
    return response, gateway


def test_catch_all_form_encoded_base64_body_with_query_parameters(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _gw()

    response = legacy_client(gateway).post(
        "/ds/1?confidence=40&api_key=x",
        content=base64.b64encode(_jpeg()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["image"] == {"width": 8, "height": 6}
    assert [prediction["class"] for prediction in body["predictions"]] == ["cat"]
    params = next(call for call in gateway.calls if call[0] == "infer")[3]
    assert params["confidence"] == 0.4


def test_catch_all_dispatches_instance_segmentation(legacy_client, fake_stat):
    mask = np.zeros((1, 6, 8), dtype=bool)
    mask[0, 1:5, 1:5] = True
    prediction = SimpleNamespace(
        xyxy=np.array([[1, 1, 5, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        mask=mask,
    )

    response, _ = _post_catch_all(
        legacy_client,
        fake_stat,
        "instance-segmentation",
        prediction,
        {"class_names": ["cat"]},
    )

    assert response.status_code == 200, response.text
    first = response.json()["predictions"][0]
    assert first["class"] == "cat"
    assert len(first["points"]) >= 3


def test_catch_all_dispatches_keypoint_detection(legacy_client, fake_stat):
    keypoints = SimpleNamespace(
        xy=np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        class_id=np.array([0]),
        confidence=np.array([[0.9, 0.8]]),
    )
    detections = SimpleNamespace(
        xyxy=np.array([[0, 0, 7, 5]], dtype=float),
        confidence=np.array([0.8]),
        class_id=np.array([0]),
    )

    response, _ = _post_catch_all(
        legacy_client,
        fake_stat,
        "keypoint-detection",
        ([keypoints], [detections]),
        {"class_names": ["person"], "key_points_classes": [["nose", "eye"]]},
    )

    assert response.status_code == 200, response.text
    first = response.json()["predictions"][0]
    assert first["class"] == "person"
    assert [point["class"] for point in first["keypoints"]] == ["nose", "eye"]


def test_catch_all_dispatches_classification(legacy_client, fake_stat):
    prediction = SimpleNamespace(confidence=np.array([[0.1, 0.9]]))

    response, _ = _post_catch_all(
        legacy_client,
        fake_stat,
        "classification",
        prediction,
        {"class_names": ["cat", "dog"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["top"] == "dog"
    assert body["predictions"][0]["class"] == "dog"


def test_catch_all_dispatches_semantic_segmentation(legacy_client, fake_stat):
    prediction = SimpleNamespace(
        segmentation_map=np.array([[0, 1], [1, 0]]),
        confidence=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )

    response, _ = _post_catch_all(
        legacy_client,
        fake_stat,
        "semantic-segmentation",
        prediction,
        {"class_names": ["bg", "fg"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predictions"]["class_map"] == {"0": "bg", "1": "fg"}
    assert body["predictions"]["present_class_ids"] == [0, 1]


def test_catch_all_image_query_parameter_is_fetched_by_url(
    legacy_client, fake_stat, monkeypatch
):
    seen = []

    async def _fetch(urls, destination_policy=None):
        seen.append(list(urls))
        return [_jpeg() for _ in urls], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _gw()

    response = legacy_client(gateway).get(
        f"/ds/1?api_key=k&confidence=50&image={IMAGE_URL}"
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["image"] == {"width": 8, "height": 6}
    assert [prediction["class"] for prediction in body["predictions"]] == ["cat"]
    assert seen == [[IMAGE_URL]]
    assert [call[4] for call in gateway.calls if call[0] == "infer"] == [_jpeg()]


def test_catch_all_class_filter_is_not_applied_to_detections(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    detections = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5], [2, 2, 4, 4]], dtype=float),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 1]),
    )
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): detections},
        model_info={"ds/1": {"class_names": ["cat", "dog"]}},
    )

    response = legacy_client(gateway).post(
        "/ds/1?api_key=k&class_filter=cat",
        content=base64.b64encode(_jpeg()),
        headers=FORM_CONTENT_TYPE,
    )

    assert response.status_code == 200, response.text
    assert [prediction["class"] for prediction in response.json()["predictions"]] == [
        "cat",
        "dog",
    ]
