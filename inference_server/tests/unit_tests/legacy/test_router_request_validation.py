import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests.unit_tests.legacy.conftest import FakeGateway


def _image():
    buf = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buf, format="JPEG")
    return {"type": "base64", "value": base64.b64encode(buf.getvalue()).decode()}


CLIP_ACTIONS = {"clip/ViT-B-16": {"actions": {"embed_text": {}, "embed_images": {}}}}
PE_ACTIONS = {
    "perception-encoder/PE-Core-L14-336": {
        "actions": {"embed_text": {}, "embed_images": {}, "compare": {}}
    }
}
SAM_ACTIONS = {"sam/vit_h": {"actions": {"segment": {}, "embed": {}}}}
SAM2_ACTIONS = {"sam2/hiera_large": {"actions": {"segment": {}, "embed": {}}}}


def _compare(**overrides):
    body = {
        "subject": "a",
        "subject_type": "text",
        "prompt": "b",
        "prompt_type": "text",
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize(
    "path, body, model_info, message",
    [
        pytest.param(
            "/clip/embed_image",
            {"image": [_image(), _image()]},
            CLIP_ACTIONS,
            "The maximum number of images that can be embedded at once is 1",
            id="too-many-images",
        ),
        pytest.param(
            "/clip/compare",
            _compare(prompt=["a", "b"]),
            CLIP_ACTIONS,
            "The maximum number of prompts that can be compared at once is 1",
            id="too-many-prompts",
        ),
        pytest.param(
            "/clip/compare",
            _compare(subject_type="audio"),
            CLIP_ACTIONS,
            "subject_type must be either 'image' or 'text'",
            id="subject-type",
        ),
        pytest.param(
            "/clip/compare",
            _compare(prompt_type="audio"),
            CLIP_ACTIONS,
            "prompt_type must be either 'image' or 'text'",
            id="prompt-type",
        ),
        pytest.param(
            "/clip/embed_image",
            {"image": []},
            CLIP_ACTIONS,
            "At least one image is required",
            id="empty-images",
        ),
        pytest.param(
            "/clip/embed_text",
            {"text": []},
            CLIP_ACTIONS,
            "At least one text is required",
            id="empty-texts",
        ),
        pytest.param(
            "/clip/compare",
            _compare(prompt=[]),
            CLIP_ACTIONS,
            "At least one prompt is required",
            id="empty-prompt-list",
        ),
        pytest.param(
            "/clip/compare",
            _compare(prompt={}),
            CLIP_ACTIONS,
            "At least one prompt is required",
            id="empty-prompt-dict",
        ),
        pytest.param(
            "/perception_encoder/embed_text",
            {"text": []},
            PE_ACTIONS,
            "At least one text is required",
            id="pe-empty-texts",
        ),
        pytest.param(
            "/perception_encoder/embed_image",
            {"image": []},
            PE_ACTIONS,
            "At least one image is required",
            id="pe-empty-images",
        ),
        pytest.param(
            "/perception_encoder/compare",
            _compare(prompt=[]),
            PE_ACTIONS,
            "At least one prompt is required",
            id="pe-empty-prompt-list",
        ),
        pytest.param(
            "/sam/segment_image",
            {},
            SAM_ACTIONS,
            "Must provide either image, cached image_id, or embeddings",
            id="sam-no-image",
        ),
        pytest.param(
            "/sam/segment_image",
            {"image": _image(), "has_mask_input": True},
            SAM_ACTIONS,
            "Must provide either mask_input or cached image_id",
            id="sam-mask-input-without-image-id",
        ),
        pytest.param(
            "/sam/segment_image",
            {"image": _image(), "format": "rle"},
            SAM_ACTIONS,
            "Invalid format rle",
            id="sam-format",
        ),
        pytest.param(
            "/sam2/segment_image",
            {"image": _image(), "format": "png"},
            SAM2_ACTIONS,
            "Invalid format png",
            id="sam2-format",
        ),
    ],
)
def test_client_mistake_in_translation_answers_400(
    legacy_client, fake_stat, monkeypatch, path, body, model_info, message
):
    monkeypatch.setattr("inference_server.configuration.CLIP_MAX_BATCH_SIZE", 1)
    gw = FakeGateway(model_info=model_info)
    r = legacy_client(gw).post(path, json=body)
    assert r.status_code == 400, r.text
    assert r.json() == {"message": message}
    assert not [c for c in gw.calls if c[0] == "infer"]


def test_catch_all_invalid_query_value_answers_400_with_field_name(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("keypoint-detection", "infer")
    det = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    gw = FakeGateway(
        predictions={("ds/1", "infer"): det},
        model_info={"ds/1": {"class_names": ["cat"]}},
    )
    r = legacy_client(gw).post(
        "/ds/1?api_key=k&confidence=best",
        content=base64.b64encode(b"x"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 400, r.text
    assert "confidence" in r.json()["message"]
    assert not [c for c in gw.calls if c[0] == "infer"]
