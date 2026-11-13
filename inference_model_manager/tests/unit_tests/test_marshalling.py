import numpy as np

from inference_model_manager.errors import INPUT_ERROR_PREFIX
from inference_model_manager.marshalling import (
    model_supports_rle,
    split_batched_result,
    tensors_to_numpy,
    to_bytes,
)
from inference_model_manager.model_manager import ModelManager


def test_input_error_prefix_value():
    assert INPUT_ERROR_PREFIX == "INPUT_ERROR: "


def test_to_bytes_roundtrips_ndarray_via_npy():
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = to_bytes(arr)
    assert payload[:6] == b"\x93NUMPY"


def test_to_bytes_passthrough_bytes():
    assert to_bytes(b"abc") == b"abc"


def test_tensors_to_numpy_passthrough_plain_objects():
    obj = {"a": [1, 2], "b": (3,)}
    assert tensors_to_numpy(obj) == obj


class _RleModel:
    supported_mask_formats = {"rle", "dense"}


def test_model_supports_rle():
    assert model_supports_rle(_RleModel()) is True
    assert model_supports_rle(object()) is False


def test_split_batched_result_list_matching_length():
    raw = ["a", "b", "c"]
    assert split_batched_result(raw, 3) is raw


def test_split_batched_result_single_image_returns_whole_result():
    raw = (["text"], ["detections"])
    assert split_batched_result(raw, 1) == [raw]


def test_split_batched_result_array_with_matching_leading_dim():
    raw = np.arange(6).reshape(3, 2)
    results = split_batched_result(raw, 3)
    assert len(results) == 3
    for index, result in enumerate(results):
        assert result.shape == (1, 2)
        assert np.array_equal(result, raw[index : index + 1])


def test_split_batched_result_structured_ocr_tuple_splits_without_retry():
    calls = []

    def retry_single(index):
        calls.append(index)
        return None

    raw = (["first", "second"], ["det-first", "det-second"])
    results = split_batched_result(raw, 2, retry_single=retry_single)

    assert calls == []
    assert results == [(["first"], ["det-first"]), (["second"], ["det-second"])]
    for result in results:
        assert isinstance(result, tuple)
        for element in result:
            assert isinstance(element, list)
            assert len(element) == 1


def test_split_batched_result_tuple_with_non_list_elements_uses_retry():
    calls = []

    def retry_single(index):
        calls.append(index)
        return f"single-{index}"

    raw = (["first", "second"], "not-a-list")
    results = split_batched_result(raw, 2, retry_single=retry_single)

    assert calls == [0, 1]
    assert results == ["single-0", "single-1"]


def test_split_batched_result_tuple_with_wrong_length_elements_uses_retry():
    calls = []

    def retry_single(index):
        calls.append(index)
        return [f"single-{index}"]

    raw = (["first", "second", "third"], ["a", "b", "c"])
    results = split_batched_result(raw, 2, retry_single=retry_single)

    assert calls == [0, 1]
    assert results == ["single-0", "single-1"]


def test_split_batched_result_mismatched_result_uses_retry():
    calls = []

    def retry_single(index):
        calls.append(index)
        return [f"single-{index}"]

    results = split_batched_result("not-batched", 3, retry_single=retry_single)

    assert calls == [0, 1, 2]
    assert results == ["single-0", "single-1", "single-2"]


def test_split_batched_result_mismatched_result_without_retry():
    assert split_batched_result("not-batched", 3) == ["not-batched"]


def test_wire_marshal_result_splits_structured_ocr_tuple_without_retry():
    calls = []

    def retry_single(index):
        calls.append(index)
        return None

    raw = (["first", "second"], ["det-first", "det-second"])
    results = ModelManager._wire_marshal_result(raw, 2, retry_single=retry_single)

    assert calls == []
    assert results == [(["first"], ["det-first"]), (["second"], ["det-second"])]


def test_wire_marshal_result_single_image_returns_unwrapped_result():
    raw = (["text"], ["detections"])
    assert ModelManager._wire_marshal_result(raw, 1) == raw


def test_wire_marshal_result_list_matching_length_unchanged():
    raw = ["a", "b"]
    assert ModelManager._wire_marshal_result(raw, 2) == ["a", "b"]


def test_wire_marshal_result_array_with_matching_leading_dim_unchanged():
    raw = np.arange(6).reshape(3, 2)
    results = ModelManager._wire_marshal_result(raw, 3)
    assert len(results) == 3
    assert np.array_equal(results[1], raw[1:2])


def test_wire_marshal_result_mismatched_result_uses_retry():
    calls = []

    def retry_single(index):
        calls.append(index)
        return [f"single-{index}"]

    results = ModelManager._wire_marshal_result(
        "not-batched", 2, retry_single=retry_single
    )

    assert calls == [0, 1]
    assert results == ["single-0", "single-1"]
