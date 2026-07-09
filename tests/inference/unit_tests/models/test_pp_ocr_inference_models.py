from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from inference.core.entities.requests.pp_ocr import PPOCRInferenceRequest
from inference.core.entities.responses.ocr import OCRInferenceResponse

pytest.importorskip(
    "inference_models.models.pp_ocrv6",
    reason="Installed `inference_models` version does not ship PP-OCRv6.",
)

from inference.models.pp_ocr import pp_ocr_inference_models


class _FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


def _build_adapter(model_id: str):
    det_sentinel = MagicMock(name="det_model")
    rec_sentinel = MagicMock(name="rec_model")

    def _auto_model(name_or_path, *args, **kwargs):
        return det_sentinel if name_or_path.startswith("pp-ocrv6-det") else rec_sentinel

    with patch.object(
        pp_ocr_inference_models.AutoModel,
        "from_pretrained",
        side_effect=_auto_model,
    ) as auto_model, patch.object(
        pp_ocr_inference_models, "PPOCRv6Pipeline"
    ) as pipeline_cls, patch.object(
        pp_ocr_inference_models,
        "get_extra_weights_provider_headers",
        return_value=None,
    ):
        adapter = pp_ocr_inference_models.InferenceModelsPPOCRAdapter(
            model_id=model_id, api_key="test-key"
        )
    return (
        adapter,
        auto_model,
        pipeline_cls,
        det_sentinel,
        rec_sentinel,
    )


def test_adapter_full_mode_loads_both_models() -> None:
    _, auto_model, pipeline_cls, det, rec = _build_adapter("pp_ocr/medium-small")

    loaded = [call.args[0] for call in auto_model.call_args_list]
    assert loaded == ["pp-ocrv6-det/medium", "pp-ocrv6-rec/small"]
    assert pipeline_cls.call_args.kwargs == {"det_model": det, "rec_model": rec}


def test_adapter_bare_model_id_defaults_small_small() -> None:
    _, auto_model, pipeline_cls, det, rec = _build_adapter("pp_ocr")

    loaded = [call.args[0] for call in auto_model.call_args_list]
    assert loaded == ["pp-ocrv6-det/small", "pp-ocrv6-rec/small"]
    assert pipeline_cls.call_args.kwargs == {"det_model": det, "rec_model": rec}


def test_adapter_legacy_single_token_expands_to_both() -> None:
    _, auto_model, pipeline_cls, det, rec = _build_adapter("pp_ocr/small")

    loaded = [call.args[0] for call in auto_model.call_args_list]
    assert loaded == ["pp-ocrv6-det/small", "pp-ocrv6-rec/small"]
    assert pipeline_cls.call_args.kwargs == {"det_model": det, "rec_model": rec}


def test_adapter_detect_only_skips_recognition() -> None:
    _, auto_model, pipeline_cls, det, _ = _build_adapter("pp_ocr/small-none")

    loaded = [call.args[0] for call in auto_model.call_args_list]
    assert loaded == ["pp-ocrv6-det/small"]
    assert pipeline_cls.call_args.kwargs == {"det_model": det, "rec_model": None}


def test_adapter_recognize_only_skips_detection() -> None:
    _, auto_model, pipeline_cls, _, rec = _build_adapter("pp_ocr/none-medium")

    loaded = [call.args[0] for call in auto_model.call_args_list]
    assert loaded == ["pp-ocrv6-rec/medium"]
    assert pipeline_cls.call_args.kwargs == {"det_model": None, "rec_model": rec}


def test_infer_from_request_builds_response_full_mode() -> None:
    adapter, *_ = _build_adapter("pp_ocr/small-small")
    pipeline_result = SimpleNamespace(
        text="hello\nworld",
        line_texts=["hello", "world"],
        detections=SimpleNamespace(
            xyxy=_FakeArray([[0.0, 0.0, 10.0, 4.0], [2.0, 6.0, 12.0, 10.0]]),
            confidence=_FakeArray([0.9, 0.8]),
        ),
    )
    image_metadata = pp_ocr_inference_models.InferenceResponseImage(width=20, height=12)
    with patch.object(adapter, "infer", return_value=(pipeline_result, image_metadata)):
        response = adapter.infer_from_request(
            PPOCRInferenceRequest(
                image={"type": "url", "value": "https://some/image.jpg"}
            )
        )

    assert isinstance(response, OCRInferenceResponse)
    assert response.result == "hello\nworld"
    assert response.image == image_metadata
    assert len(response.predictions) == 2
    first = response.predictions[0]
    assert first.class_name == "hello"
    assert first.x == 5.0
    assert first.y == 2.0
    assert first.width == 10.0
    assert first.height == 4.0
    assert first.confidence == 0.9


def test_infer_from_request_detect_only_empty_result() -> None:
    adapter, *_ = _build_adapter("pp_ocr/small-none")
    pipeline_result = SimpleNamespace(
        text="",
        line_texts=[],
        detections=SimpleNamespace(
            xyxy=_FakeArray([[0.0, 0.0, 10.0, 4.0], [2.0, 6.0, 12.0, 10.0]]),
            confidence=_FakeArray([0.9, 0.8]),
        ),
    )
    image_metadata = pp_ocr_inference_models.InferenceResponseImage(width=20, height=12)
    with patch.object(adapter, "infer", return_value=(pipeline_result, image_metadata)):
        response = adapter.infer_from_request(
            PPOCRInferenceRequest(
                image={"type": "url", "value": "https://some/image.jpg"},
                text_recognition="none",
            )
        )

    assert response.result == ""
    assert len(response.predictions) == 2
    assert all(pred.class_name == "" for pred in response.predictions)


def test_infer_from_request_recognize_only_empty_predictions() -> None:
    adapter, *_ = _build_adapter("pp_ocr/none-small")
    pipeline_result = SimpleNamespace(
        text="a single line",
        line_texts=["a single line"],
        detections=None,
    )
    image_metadata = pp_ocr_inference_models.InferenceResponseImage(width=20, height=12)
    with patch.object(adapter, "infer", return_value=(pipeline_result, image_metadata)):
        response = adapter.infer_from_request(
            PPOCRInferenceRequest(
                image={"type": "url", "value": "https://some/image.jpg"},
                text_detection="none",
            )
        )

    assert response.result == "a single line"
    assert response.predictions == []


def test_infer_from_request_handles_list_of_images() -> None:
    adapter, *_ = _build_adapter("pp_ocr/small-small")
    pipeline_result = SimpleNamespace(
        text="hi",
        line_texts=["hi"],
        detections=SimpleNamespace(
            xyxy=_FakeArray([[0.0, 0.0, 4.0, 2.0]]),
            confidence=_FakeArray([0.7]),
        ),
    )
    image_metadata = pp_ocr_inference_models.InferenceResponseImage(width=8, height=4)
    with patch.object(adapter, "infer", return_value=(pipeline_result, image_metadata)):
        response = adapter.infer_from_request(
            PPOCRInferenceRequest(
                image=[
                    {"type": "url", "value": "https://some/image-1.jpg"},
                    {"type": "url", "value": "https://some/image-2.jpg"},
                ]
            )
        )

    assert isinstance(response, list)
    assert len(response) == 2
    assert all(isinstance(item, OCRInferenceResponse) for item in response)
