"""Protocol-neutral remote chat helpers: keys, scrubbing, concurrency, images."""

import base64
import threading
import time

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.language.chat import (
    ChatRequestError,
    check_reasoning_effort,
    check_task_fields,
    encode_image,
    is_managed_key,
    reveal_keys,
    run_concurrently,
    scrubbed_call,
    third_party_model,
    token_count,
    token_outputs,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

PROVIDER_KEY = "sk-test-PROVIDER-91c2"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-44d1"


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        ("rf_key:account", True),
        ("rf_key:user:abc", True),
        ("rf_key:other", False),
        ("sk-123", False),
        (None, False),
    ],
)
def test_is_managed_key(api_key, expected) -> None:
    assert is_managed_key(api_key) is expected


def test_reveal_keys_wraps_plain_strings_so_they_are_scrubbed() -> None:
    keys = reveal_keys(PROVIDER_KEY, roboflow_api_key=Secret(ROBOFLOW_KEY, name="RF"))

    assert keys.provider_key == PROVIDER_KEY
    assert keys.roboflow_key == ROBOFLOW_KEY
    assert not keys.managed
    assert keys.scrub(f"bad {PROVIDER_KEY} and {ROBOFLOW_KEY}") == (
        "bad <secret api_key> and <secret RF>"
    )
    assert PROVIDER_KEY not in repr(keys)


def test_reveal_keys_accepts_missing_roboflow_key_and_managed_reference() -> None:
    keys = reveal_keys("rf_key:account", roboflow_api_key=None)

    assert keys.managed
    assert keys.roboflow_key is None


def test_reveal_keys_refuses_a_replayed_secret() -> None:
    with pytest.raises(SecretUnavailableError):
        reveal_keys(Secret.unavailable("OPENAI_KEY"), roboflow_api_key=None)


def test_scrubbed_call_returns_the_request_result() -> None:
    keys = reveal_keys(PROVIDER_KEY, roboflow_api_key=None)

    assert scrubbed_call(lambda: 42, keys=keys, provider="OpenAI") == 42


def test_scrubbed_call_replaces_any_error_and_chains_nothing() -> None:
    keys = reveal_keys(PROVIDER_KEY, roboflow_api_key=None)

    def request() -> None:
        try:
            raise KeyError(PROVIDER_KEY)
        except KeyError:
            raise ConnectionError(f"401 for Bearer {PROVIDER_KEY}")

    with pytest.raises(ChatRequestError) as caught:
        scrubbed_call(request, keys=keys, provider="OpenAI")

    assert str(caught.value) == "OpenAI request failed: 401 for Bearer <secret api_key>"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_scrubbed_call_keeps_own_chat_errors_without_prefix() -> None:
    keys = reveal_keys(PROVIDER_KEY, roboflow_api_key=None)

    def request() -> None:
        raise ChatRequestError(f"OpenAI API request was cancelled ({PROVIDER_KEY})")

    with pytest.raises(ChatRequestError) as caught:
        scrubbed_call(request, keys=keys, provider="OpenAI")

    assert str(caught.value) == "OpenAI API request was cancelled (<secret api_key>)"
    assert caught.value.__context__ is None


def test_run_concurrently_keeps_call_order_and_respects_the_limit() -> None:
    active = []
    peak = []
    lock = threading.Lock()

    def call(index: int, delay: float):
        def run() -> int:
            with lock:
                active.append(index)
                peak.append(len(active))
            time.sleep(delay)
            with lock:
                active.remove(index)
            return index

        return run

    calls = [call(index, delay) for index, delay in enumerate([0.05, 0.0, 0.03, 0.0])]

    results = run_concurrently(calls, max_concurrent_requests=2)

    assert results == [0, 1, 2, 3]
    assert max(peak) <= 2


def test_run_concurrently_cancels_unstarted_calls_after_a_failure() -> None:
    started = []

    def call(index: int):
        def run() -> int:
            started.append(index)
            if index == 1:
                raise ValueError("second failed")
            return index

        return run

    with pytest.raises(ValueError, match="second failed"):
        run_concurrently([call(index) for index in range(4)], max_concurrent_requests=1)

    assert started == [0, 1]


def test_run_concurrently_runs_one_call_in_the_calling_thread() -> None:
    caller = threading.get_ident()

    results = run_concurrently([threading.get_ident], max_concurrent_requests=None)

    assert results == [caller]
    assert run_concurrently([], max_concurrent_requests=None) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_run_concurrently_rejects_a_limit_below_one(limit) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_concurrently([lambda: 1, lambda: 2], max_concurrent_requests=limit)


@pytest.mark.parametrize(
    ("task_type", "fields", "missing"),
    [
        ("unconstrained", {}, "prompt"),
        ("visual-question-answering", {}, "prompt"),
        ("classification", {}, "classes"),
        ("instance-segmentation", {}, "classes"),
        ("structured-answering", {}, "output_structure"),
    ],
)
def test_check_task_fields_names_the_missing_field(task_type, fields, missing) -> None:
    arguments = {"prompt": None, "classes": None, "output_structure": None, **fields}

    with pytest.raises(ValueError, match=f"`{missing}` parameter required"):
        check_task_fields(task_type, **arguments)


def test_check_task_fields_accepts_complete_configuration() -> None:
    check_task_fields("ocr", prompt=None, classes=None, output_structure=None)
    check_task_fields(
        "object-detection", prompt=None, classes=["a"], output_structure=None
    )


def test_check_reasoning_effort() -> None:
    levels = {"gpt-5.1": ["none", "low"], "gpt-4o": []}

    check_reasoning_effort("gpt-5.1", "low", levels_by_model=levels)
    check_reasoning_effort("future-model", "max", levels_by_model=levels)
    check_reasoning_effort("$inputs.model", "max", levels_by_model=levels)
    check_reasoning_effort("gpt-4o", None, levels_by_model=levels)
    with pytest.raises(
        ValueError, match=r"supports reasoning_effort values \['none', 'low'\]"
    ):
        check_reasoning_effort("gpt-5.1", "max", levels_by_model=levels)
    with pytest.raises(ValueError, match="does not support configurable"):
        check_reasoning_effort("gpt-4o", "low", levels_by_model=levels)


def _decoded(encoded) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(encoded.base64), dtype=np.uint8)
    pixels = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    return pixels


def test_encode_image_png_is_lossless_and_in_bgr_file_order() -> None:
    pixels = torch.randint(0, 256, (3, 5, 7), dtype=torch.uint8)
    image = ImageData.from_tensor(pixels)

    encoded = encode_image(image, image_format="png")

    assert encoded.media_type == "image/png"
    assert encoded.size_wh == (7, 5)
    assert encoded.data_url.startswith("data:image/png;base64,")
    decoded = _decoded(encoded)
    assert np.array_equal(decoded[:, :, ::-1], pixels.permute(1, 2, 0).numpy())


def test_encode_image_jpeg_keeps_colour_and_resizes_to_upload_size() -> None:
    pixels = torch.zeros((3, 40, 80), dtype=torch.uint8)
    pixels[0] = 255
    image = ImageData.from_tensor(pixels)

    encoded = encode_image(image, size_wh=(40, 20))

    assert encoded.media_type == "image/jpeg"
    assert encoded.size_wh == (40, 20)
    decoded = _decoded(encoded)
    assert decoded.shape == (20, 40, 3)
    blue, green, red = decoded[10, 20]
    assert red > 240 and green < 15 and blue < 15


def test_encode_image_single_channel() -> None:
    image = ImageData.from_tensor(torch.full((1, 4, 6), 128, dtype=torch.uint8))

    encoded = encode_image(image, image_format="png")

    assert _decoded(encoded).shape == (4, 6)


def test_encode_image_rejects_unknown_format() -> None:
    image = ImageData.from_tensor(torch.zeros((3, 2, 2), dtype=torch.uint8))

    with pytest.raises(ValueError, match="image_format"):
        encode_image(image, image_format="gif")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3), (0, 0), (None, None), (True, None), (2.0, None), ("4", None)],
)
def test_token_count(value, expected) -> None:
    assert token_count(value) == expected


def test_token_outputs_and_third_party_model() -> None:
    outputs = token_outputs(source="images")
    dependency = third_party_model(
        "gpt-5.1", provider="openai", resource="openai_http_client"
    )

    assert set(outputs) == {"input_tokens", "output_tokens"}
    assert outputs["input_tokens"].kinds == (INTEGER_KIND,)
    assert outputs["output_tokens"].source == "images"
    assert dependency.resource_type == "third_party_model"
    assert dependency.identifier == "gpt-5.1"
    assert dependency.details == {
        "provider": "openai",
        "resource": "openai_http_client",
    }
