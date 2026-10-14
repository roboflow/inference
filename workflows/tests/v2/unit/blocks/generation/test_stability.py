"""Stability AI blocks called directly, with an in-process fake client.

No request leaves the process. The fake records each request and answers
with a PNG of one known color, so channel order and size are checkable::

    block.run(...) ─▶ FakeClient.post(path, api_key, data, files) ─▶ PNG bytes
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import pydantic
import pytest
import torch
from roboflow_workflows.v2.blocks.generation._shared.stability import ERROR_DETAIL_LIMIT, GENERATION_ENDPOINTS, INPAINTING_ENDPOINT, OUTPAINTING_ENDPOINT, StabilityAIClient, StabilityAIError, edited_image
from roboflow_workflows.v2.blocks.generation.stability_ai_image_gen.v1 import StabilityAIImageGenerationBlock
from roboflow_workflows.v2.blocks.generation.stability_ai_inpainting.v1 import StabilityAIInpaintingBlock
from roboflow_workflows.v2.blocks.generation.stability_ai_outpainting.v1 import StabilityAIOutpaintingBlock
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError
from roboflow_workflows.v2.engine.declaration import spec_of

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

KEY = "sk-test-STABILITY-0b7e"
RESULT_RGB = (200, 10, 30)
HEIGHT, WIDTH = 48, 64


class FakeClient:
    """Records requests; answers with a solid PNG or raises ``error``."""

    def __init__(
        self, *, size_hw: Tuple[int, int] = (HEIGHT, WIDTH), error: Any = None
    ) -> None:
        self.size_hw = size_hw
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def post(
        self, path: str, *, api_key: str, data: Dict[str, Any], files: Dict[str, Any]
    ) -> bytes:
        self.calls.append(
            {"path": path, "api_key": api_key, "data": dict(data), "files": files}
        )
        if self.error is not None:
            raise self.error

        rgb = np.empty((*self.size_hw, 3), dtype=np.uint8)
        rgb[:] = RESULT_RGB
        _, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        return png.tobytes()


def _image(image_id: str = "frame") -> ImageData:
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[:, :, 1] = 120
    image = ImageData.from_numpy_rgb(pixels, image_id=image_id)

    return image


def _masks() -> np.ndarray:
    masks = np.zeros((2, HEIGHT, WIDTH), dtype=bool)
    masks[0, 4:20, 4:20] = True
    masks[1, 24:44, 36:60] = True

    return masks


def _segmentation(image: ImageData, masks: Optional[np.ndarray] = None) -> Any:
    masks = _masks() if masks is None else masks
    boxes = [_box_of(mask) for mask in masks]
    predictions = InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(len(masks), dtype=torch.int64),
        confidence=torch.full((len(masks),), 0.9),
        mask=torch.from_numpy(masks),
        image_metadata=FrameRef.from_image(image).prediction_metadata(),
    )

    return predictions


def _box_of(mask: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    return box


def _rle(dense: InstanceDetections, image: ImageData) -> InstanceDetections:
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
        for mask in dense.mask
    ]
    predictions = InstanceDetections(
        xyxy=dense.xyxy,
        class_id=dense.class_id,
        confidence=dense.confidence,
        mask=InstancesRLEMasks.from_coco_rle_masks(image.size_hw, encoded),
        image_metadata=dense.image_metadata,
    )

    return predictions


def _view(dense: InstanceDetections, image: ImageData) -> SegmentationView:
    # Full-resolution logits: the reference masks equal the dense ones.
    height, width = image.size_hw
    view = SegmentationView(
        detections=Detections(
            xyxy=dense.xyxy,
            class_id=dense.class_id,
            confidence=dense.confidence,
            image_metadata=dense.image_metadata,
        ),
        scores=dense.mask.float() * 16.0 - 8.0,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry(
            grid_size_hw=(height, width),
            inference_size_hw=(height, width),
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(1.0, 1.0),
            crop_size_hw=(height, width),
            original_size_hw=(height, width),
            static_crop_xywh=(0, 0, width, height),
        ),
        take_ownership=True,
    )

    return view


def _decoded(jpeg: bytes) -> np.ndarray:
    array = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    return array


def _inpaint(client: FakeClient, image: ImageData, predictions: Any, **params: Any):
    arguments = dict(
        image=image,
        segmentation_mask=predictions,
        prompt="a carriage",
        negative_prompt=None,
        invert_segmentation_mask=False,
        preset=None,
        seed=None,
        api_key=KEY,
    )
    arguments.update(params)
    result = StabilityAIInpaintingBlock(stability_ai_client=client).run(**arguments)

    return result


def _outpaint(client: FakeClient, image: ImageData, **params: Any):
    arguments = dict(
        image=image,
        creativity=0.5,
        left=None,
        right=None,
        up=None,
        down=None,
        prompt=None,
        preset=None,
        seed=None,
        api_key=KEY,
    )
    arguments.update(params)
    result = StabilityAIOutpaintingBlock(stability_ai_client=client).run(**arguments)

    return result


def _generate(client: FakeClient, **params: Any):
    arguments = dict(
        prompt="a red bicycle",
        negative_prompt=None,
        model="core",
        image=None,
        strength=0.3,
        api_key=KEY,
    )
    arguments.update(params)
    result = StabilityAIImageGenerationBlock(stability_ai_client=client).run(
        **arguments
    )

    return result


def test_text_generation_sends_v1_fields_and_returns_a_new_rgb_root() -> None:
    # given
    client = FakeClient(size_hw=(30, 40))

    # when
    generated = _generate(client)["image"]

    # then
    assert client.calls == [
        {
            "path": GENERATION_ENDPOINTS["core"],
            "api_key": KEY,
            "data": {"prompt": "a red bicycle", "output_format": "jpeg"},
            "files": {"none": ""},
        }
    ]
    assert generated.image_id.startswith("stability-")
    assert generated.parent.frame_id == generated.image_id
    assert generated.root.frame_id == generated.image_id
    assert generated.size_hw == (30, 40)
    assert generated.tensor_image[:, 0, 0].tolist() == list(RESULT_RGB)


def test_variation_sends_the_image_strength_and_negative_prompt() -> None:
    # given
    client = FakeClient()
    image = _image()

    # when
    generated = _generate(
        client, image=image, strength=0.7, negative_prompt="text", model="ultra"
    )["image"]

    # then
    call = client.calls[0]
    assert call["path"] == GENERATION_ENDPOINTS["ultra"]
    assert call["data"] == {
        "prompt": "a red bicycle",
        "output_format": "jpeg",
        "strength": 0.7,
        "negative_prompt": "text",
    }
    sent_bgr = _decoded(call["files"]["image"])
    assert np.abs(sent_bgr[:, :, 1].astype(int) - 120).max() <= 2
    assert sent_bgr[:, :, [0, 2]].max() <= 2
    assert generated.root.frame_id == generated.image_id != image.image_id


@pytest.mark.parametrize(
    "params, message",
    [
        ({"model": "sdxl"}, "model must be one of"),
        ({"strength": 1.5}, "strength must be in"),
    ],
)
def test_generation_rejects_bad_runtime_values_before_sending(
    params: Dict[str, Any], message: str
) -> None:
    # given
    client = FakeClient()

    # when
    with pytest.raises(ValueError, match=message):
        _generate(client, **params)

    # then
    assert client.calls == []


@pytest.mark.parametrize(
    "block, params",
    [
        (StabilityAIImageGenerationBlock, {"prompt": "p", "model": "sdxl"}),
        (StabilityAIImageGenerationBlock, {"prompt": "p", "strength": -0.1}),
        (StabilityAIOutpaintingBlock, {"image": "$inputs.image", "left": 2001}),
        (StabilityAIOutpaintingBlock, {"image": "$inputs.image", "creativity": 2}),
        (StabilityAIInpaintingBlock, {"preset": "watercolor"}),
        (StabilityAIInpaintingBlock, {"seed": -1}),
    ],
)
def test_out_of_range_literals_fail_validation(block: type, params: dict) -> None:
    # given
    base = {
        "image": "$inputs.image",
        "segmentation_mask": "$steps.seg.predictions",
        "prompt": "p",
    }
    fields = spec_of(block).fields
    values = {name: value for name, value in base.items() if name in fields}
    values.update(params, api_key=KEY)

    # when / then
    with pytest.raises(pydantic.ValidationError):
        block.Params(**values)


def test_a_secret_key_is_revealed_only_for_the_request() -> None:
    # given
    client = FakeClient()
    secret = Secret(KEY, name="STABILITY_KEY")

    # when
    _generate(client, api_key=secret)

    # then
    assert client.calls[0]["api_key"] == KEY


def test_a_failure_message_is_scrubbed_and_keeps_no_exception_holding_the_key() -> None:
    # given
    client = FakeClient(error=ConnectionError(f"refused for Bearer {KEY}"))

    # when
    with pytest.raises(StabilityAIError) as caught:
        _generate(client, api_key=Secret(KEY, name="STABILITY_KEY"))

    # then
    message = str(caught.value)
    assert "ConnectionError: refused for Bearer <secret STABILITY_KEY>" in message
    assert KEY not in message
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_a_replayed_secret_fails_without_sending() -> None:
    # given
    client = FakeClient()

    # when
    with pytest.raises(SecretUnavailableError):
        _generate(client, api_key=Secret.unavailable("STABILITY_KEY"))

    # then
    assert client.calls == []


@pytest.mark.parametrize("payload", [b"", b"not an image", None])
def test_a_response_that_is_not_an_image_raises(payload: Any) -> None:
    # given
    class Answer(FakeClient):
        def post(self, path: str, **kwargs: Any) -> Any:
            return payload

    # when / then
    with pytest.raises(StabilityAIError, match="image"):
        _generate(Answer())


def test_inpainting_sends_the_blurred_union_of_masks_and_keeps_geometry() -> None:
    # given
    client = FakeClient()
    image = _image()

    # when
    inpainted = _inpaint(
        client,
        image,
        _segmentation(image),
        negative_prompt="blurry",
        preset="anime",
        seed=7,
    )["image"]

    # then
    call = client.calls[0]
    assert call["path"] == INPAINTING_ENDPOINT
    assert call["data"] == {
        "prompt": "a carriage",
        "output_format": "jpeg",
        "negative_prompt": "blurry",
        "preset": "anime",
        "seed": 7,
    }
    mask = _decoded(call["files"]["mask"])
    assert mask.shape == (HEIGHT, WIDTH, 3)
    assert mask[12, 12].min() >= 250  # inside the first mask
    assert mask[34, 48].min() >= 250  # inside the second mask
    assert mask[40, 8].max() <= 5  # outside both, beyond the blur
    assert 20 < mask[12, 20, 0] < 235  # the blurred edge
    assert inpainted.image_id == image.image_id
    assert inpainted.parent == image.parent and inpainted.root == image.root
    assert inpainted.tensor_image[:, 5, 5].tolist() == list(RESULT_RGB)


def test_inverting_repaints_the_background_and_seed_zero_is_not_sent() -> None:
    # given
    client = FakeClient()
    image = _image()

    # when
    _inpaint(client, image, _segmentation(image), invert_segmentation_mask=True, seed=0)

    # then
    mask = _decoded(client.calls[0]["files"]["mask"])
    assert mask[12, 12].max() <= 5
    assert mask[40, 8].min() >= 250
    assert "seed" not in client.calls[0]["data"]


@pytest.mark.parametrize("representation", ["rle", "view"])
def test_inpainting_reads_rle_and_view_masks_like_dense_ones(
    representation: str,
) -> None:
    # given
    image = _image()
    dense = _segmentation(image)
    other = _rle(dense, image) if representation == "rle" else _view(dense, image)
    dense_client, other_client = FakeClient(), FakeClient()

    # when
    _inpaint(dense_client, image, dense)
    _inpaint(other_client, image, other)

    # then
    assert (
        other_client.calls[0]["files"]["mask"] == dense_client.calls[0]["files"]["mask"]
    )


def test_empty_predictions_still_send_an_all_black_mask_as_v1() -> None:
    # given
    client = FakeClient()
    image = _image()
    empty = _segmentation(image, masks=np.zeros((0, HEIGHT, WIDTH), dtype=bool))

    # when
    _inpaint(client, image, empty)

    # then
    assert _decoded(client.calls[0]["files"]["mask"]).max() <= 2


def test_masks_of_another_image_are_rejected_before_sending() -> None:
    # given
    client = FakeClient()
    image = _image("frame")
    foreign = _segmentation(_image("other-frame"))

    # when
    with pytest.raises(ValueError, match="cannot be placed"):
        _inpaint(client, image, foreign)

    # then
    assert client.calls == []


def test_an_inpainted_result_of_another_size_maps_back_with_its_scale() -> None:
    # given
    client = FakeClient(size_hw=(HEIGHT * 2, WIDTH * 2))
    image = _image()

    # when
    inpainted = _inpaint(client, image, _segmentation(image))["image"]

    # then
    assert inpainted.image_id != image.image_id
    assert inpainted.parent.frame_id == image.image_id
    assert inpainted.parent.map_xy(WIDTH * 2, HEIGHT * 2) == (WIDTH, HEIGHT)


def test_outpainting_without_sides_returns_the_input_and_sends_nothing() -> None:
    # given
    client = FakeClient()
    image = _image()

    # when
    result = _outpaint(client, image, left=0)

    # then
    assert result["image"] is image
    assert client.calls == []


def test_outpainting_places_the_input_at_left_and_up_of_the_result() -> None:
    # given
    client = FakeClient(size_hw=(HEIGHT + 10, WIDTH + 30))
    image = _image()

    # when
    outpainted = _outpaint(
        client, image, left=20, right=10, up=10, prompt="sand", preset="enhance"
    )["image"]

    # then
    call = client.calls[0]
    assert call["path"] == OUTPAINTING_ENDPOINT
    assert call["data"] == {
        "output_format": "jpeg",
        "creativity": 0.5,
        "left": 20,
        "right": 10,
        "up": 10,
        "preset": "enhance",
        "prompt": "sand",
    }
    assert outpainted.size_hw == (HEIGHT + 10, WIDTH + 30)
    assert outpainted.parent.frame_id == image.image_id
    assert outpainted.parent.map_xy(20, 10) == (0.0, 0.0)
    assert outpainted.root.map_xy(20 + WIDTH, 10 + HEIGHT) == (WIDTH, HEIGHT)


def test_an_outpainted_result_of_another_size_keeps_exact_coordinates() -> None:
    # given: the service answered at half the requested size
    image = _image()
    pixels = torch.zeros((3, (HEIGHT + 20) // 2, (WIDTH + 40) // 2), dtype=torch.uint8)

    # when
    outpainted = edited_image(image, pixels, padding=(40, 20, 0, 0))

    # then
    assert outpainted.parent.map_xy(20, 10) == (0.0, 0.0)
    assert outpainted.parent.map_xy(pixels.shape[2], pixels.shape[1]) == (
        WIDTH,
        HEIGHT,
    )


@pytest.mark.parametrize(
    "params, message",
    [
        ({"left": -5}, "left must be in"),
        ({"down": 2001}, "down must be in"),
        ({"left": 10, "creativity": 1.2}, "creativity must be in"),
        ({"left": 10, "seed": 4294967295}, "seed must be in"),
    ],
)
def test_outpainting_rejects_runtime_values_v1_clamped(
    params: Dict[str, Any], message: str
) -> None:
    # given
    client = FakeClient()

    # when
    with pytest.raises(ValueError, match=message):
        _outpaint(client, _image(), **params)

    # then
    assert client.calls == []


def test_workload_hooks_name_the_service_and_operations() -> None:
    # given
    generation = StabilityAIImageGenerationBlock.Params(
        prompt="p", model="$inputs.model", api_key=KEY
    )

    # when
    resources = StabilityAIImageGenerationBlock.discover_dependent_resources(generation)
    operations = StabilityAIInpaintingBlock.discover_work_operations(None)

    # then
    assert [(item.resource_type, item.identifier) for item in resources] == [
        ("third_party_model", "$inputs.model")
    ]
    assert resources[0].details["provider"] == "stability_ai"
    assert [operation.value for operation in operations] == [
        "model_inference",
        "external_request",
        "image_encoding",
        "image_composition",
    ]


@pytest.mark.parametrize(
    "arguments", [{"base_url": "api.stability.ai"}, {"timeout": 0}]
)
def test_the_client_rejects_a_bad_base_url_or_timeout(arguments: dict) -> None:
    # when / then
    with pytest.raises(ValueError):
        StabilityAIClient(**arguments)


def test_the_default_client_targets_the_v1_host() -> None:
    # given
    resource = spec_of(StabilityAIOutpaintingBlock).resources[0]

    # then
    assert resource.name == "stability_ai_client"
    assert resource.default.base_url == "https://api.stability.ai"


@pytest.mark.parametrize("adapter", ["generate", "inpaint", "outpaint"])
@pytest.mark.parametrize("error_path", ["json-string", "json-object", "text", "raised"])
@pytest.mark.parametrize("key_offset", [0, 480])
def test_provider_error_redaction_before_cutoff(
    monkeypatch: pytest.MonkeyPatch, adapter: str, error_path: str, key_offset: int
) -> None:
    import requests

    # A full key at offset 480 used to become a visible prefix at the cutoff.
    key = "sk-redaction-synthetic-0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    body = (
        "diagnostic: " + "x" * max(0, key_offset - 12) + key + " trailing detail" * 80
    )
    calls = []

    class Response:
        status_code = 401
        text = body

        def json(self):
            if error_path == "text":
                raise ValueError("not JSON")
            return {"error": body} if error_path == "json-object" else body

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(requests, "post", post)
    client = (
        FakeClient(error=ConnectionError(body))
        if error_path == "raised"
        else StabilityAIClient()
    )
    image = _image()
    secret = Secret(key, name="STABILITY_KEY")
    with pytest.raises(StabilityAIError) as caught:
        if adapter == "generate":
            _generate(client, api_key=secret)
        elif adapter == "inpaint":
            _inpaint(client, image, _segmentation(image), api_key=secret)
        else:
            _outpaint(client, image, left=1, api_key=secret)

    error = caught.value
    message = str(error)
    prefix = ": Stability AI request failed: "
    assert "diagnostic:" in message
    assert key not in message and key[:3] not in message
    assert len(message.split(prefix, 1)[1]) <= ERROR_DETAIL_LIMIT
    assert error.__cause__ is None and error.__context__ is None
    if error_path == "raised":
        assert "ConnectionError:" in message and len(client.calls) == 1
        if key_offset == 0:
            assert "<secret STABILITY_KEY>" in message
    else:
        assert "HTTP 401" in message and len(calls) == 1
        assert calls[0][1]["headers"]["authorization"] == f"Bearer {key}"
        if key_offset == 0:
            assert "<secret api_key>" in message


@pytest.mark.parametrize("json_body", [False, True])
@pytest.mark.parametrize("key_length", [60, 600])
def test_transport_redacts_complete_keys_before_bounding_detail(
    monkeypatch: pytest.MonkeyPatch, json_body: bool, key_length: int
) -> None:
    import requests

    key = "sk-direct-synthetic-" + "K" * key_length
    body = "diagnostic: " + "x" * 468 + key + " tail" * 200

    class Response:
        status_code = 401
        text = body

        def json(self):
            if json_body:
                return body
            raise ValueError("not JSON")

    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response())
    with pytest.raises(StabilityAIError) as caught:
        StabilityAIClient().post("/test", api_key=key, data={}, files={})

    message = str(caught.value)
    assert message.startswith("HTTP 401 from /test: diagnostic:")
    assert key not in message and key[:3] not in message
    assert len(message.removeprefix("HTTP 401 from /test: ")) <= ERROR_DETAIL_LIMIT
    assert caught.value.__cause__ is None and caught.value.__context__ is None
