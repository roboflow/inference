import base64
import re

import cv2
import numpy as np
import pytest
from aioresponses import CallbackResult, aioresponses
from PIL import Image
from requests_mock.mocker import Mocker

from inference_sdk.http.client import InferenceHTTPClient
from inference_sdk.http.entities import InferenceConfiguration


def _prediction_for_size(width: int, height: int, *, prediction_kind: str) -> dict:
    prediction = {
        "x": width / 2,
        "y": height / 2,
        "width": width,
        "height": height,
        "confidence": 0.9,
        "class": "object",
    }
    if prediction_kind in {"points", "keypoints"}:
        prediction[prediction_kind] = [
            {"x": 0.0, "y": 0.0},
            {"x": float(width), "y": float(height)},
        ]

    response = {
        "image": {"width": width, "height": height},
        "predictions": [prediction],
    }
    if prediction_kind == "mask_grid":
        prediction.update(x=1.0, y=1.0, width=2, height=2)
        response["original_image"] = response.pop("image")
        response["mask_metadata"] = {
            "coordinate_system": "mask_grid",
            "width": 2,
            "height": 2,
            "scale_x": width / 2,
            "scale_y": height / 2,
        }

    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("height,width", [(1, 4096), (4096, 1), (1, 1280), (1280, 1)])
@pytest.mark.parametrize("use_pillow", [False, True])
@pytest.mark.parametrize(
    "prediction_kind", ["bbox", "points", "keypoints", "mask_grid"]
)
@pytest.mark.parametrize("use_async", [False, True])
async def test_thin_image_predictions_keep_original_coordinates(
    requests_mock: Mocker,
    height: int,
    width: int,
    use_pillow: bool,
    prediction_kind: str,
    use_async: bool,
) -> None:
    pixels = np.full((height, width, 3), 128, dtype=np.uint8)
    image = Image.fromarray(pixels) if use_pillow else pixels
    client = InferenceHTTPClient(api_url="http://thin-image.test", api_key="test-key")
    client.configure(
        inference_configuration=InferenceConfiguration(
            default_max_input_size=640, client_downsizing_disabled=False
        )
    )

    def _server_response(encoded_image):
        # Return predictions in the *actual uploaded* image's coordinate system.
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded_image), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        uploaded_height, uploaded_width = decoded.shape[:2]
        response = _prediction_for_size(
            uploaded_width, uploaded_height, prediction_kind=prediction_kind
        )

        return response

    if use_async:

        def _async_response(url, **kwargs):
            response = CallbackResult(payload=_server_response(kwargs["data"]))

            return response

        with aioresponses() as mocked_http:
            mocked_http.post(
                re.compile(r"http://thin-image\.test/project/1\?.*"),
                callback=_async_response,
            )
            result = await client.infer_from_api_v0_async(image, model_id="project/1")
    else:
        requests_mock.post(
            "http://thin-image.test/project/1",
            json=lambda request, context: _server_response(request.body),
        )
        result = client.infer_from_api_v0(image, model_id="project/1")

    assert result == _prediction_for_size(
        width, height, prediction_kind=prediction_kind
    )
