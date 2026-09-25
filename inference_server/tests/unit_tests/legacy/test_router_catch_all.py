import base64
import io
from types import SimpleNamespace

import numpy as np
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
