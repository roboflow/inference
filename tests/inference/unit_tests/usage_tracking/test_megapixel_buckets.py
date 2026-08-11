from types import SimpleNamespace

import numpy as np
import pytest

from inference.usage_tracking.megapixel_buckets import (
    billable_hw_from_model,
    build_megapixel_buckets,
    count_inference_images,
    get_fixed_model_input_hw,
    get_tensor_spatial_hw,
    megapixel_bucket_for_hw,
    prepare_sam_usage_billing,
    stamp_billable_model_input,
)
from inference.usage_tracking.payload_helpers import (
    merge_megapixel_buckets,
    merge_usage_dicts,
)


def test_megapixel_bucket_boundaries():
    assert megapixel_bucket_for_hw(420, 420) == "0-0.25"
    assert megapixel_bucket_for_hw(640, 640) == "0.25-0.5"
    assert megapixel_bucket_for_hw(1000, 1000) == "0.5-1"
    assert megapixel_bucket_for_hw(1280, 1280) == "1-2"
    assert megapixel_bucket_for_hw(2000, 2000) == "2-4"
    assert megapixel_bucket_for_hw(3000, 3000) == "4+"


def test_build_and_merge_megapixel_buckets():
    first = build_megapixel_buckets(
        height=640,
        width=640,
        frames=2,
        execution_duration=0.4,
    )
    second = build_megapixel_buckets(
        height=1280,
        width=1280,
        frames=1,
        execution_duration=0.3,
    )
    third = build_megapixel_buckets(
        height=640,
        width=640,
        frames=3,
        execution_duration=0.6,
    )

    merged = merge_megapixel_buckets(first, second)
    merged = merge_megapixel_buckets(merged, third)

    assert merged["0.25-0.5"]["processed_frames"] == 5
    assert merged["0.25-0.5"]["execution_duration"] == 1.0
    assert merged["1-2"]["processed_frames"] == 1
    assert merged["1-2"]["execution_duration"] == 0.3


def test_merge_usage_dicts_sums_megapixel_buckets():
    left = {
        "resource_id": "st-inst-seg/9",
        "processed_frames": 2,
        "execution_duration": 0.4,
        "megapixel_buckets": {
            "0.25-0.5": {"processed_frames": 2, "execution_duration": 0.4},
        },
    }
    right = {
        "resource_id": "st-inst-seg/9",
        "processed_frames": 1,
        "execution_duration": 0.3,
        "megapixel_buckets": {
            "0.25-0.5": {"processed_frames": 1, "execution_duration": 0.2},
            "1-2": {"processed_frames": 1, "execution_duration": 0.1},
        },
    }

    merged = merge_usage_dicts(d1=left, d2=right)

    assert merged["processed_frames"] == 3
    assert merged["execution_duration"] == pytest.approx(0.7)
    assert merged["megapixel_buckets"]["0.25-0.5"]["processed_frames"] == 3
    assert merged["megapixel_buckets"]["0.25-0.5"][
        "execution_duration"
    ] == pytest.approx(0.6)
    assert merged["megapixel_buckets"]["1-2"]["processed_frames"] == 1


def test_get_fixed_model_input_hw_prefers_img_size_attrs():
    model = SimpleNamespace(img_size_h=640, img_size_w=640)

    assert get_fixed_model_input_hw(model) == (640, 640)


def test_get_fixed_model_input_hw_ignores_dynamic_onnx_strings():
    model = SimpleNamespace(img_size_h="height", img_size_w="width", preproc={})

    assert get_fixed_model_input_hw(model) is None


def test_get_tensor_spatial_hw_nchw_and_nhwc():
    assert get_tensor_spatial_hw(np.zeros((2, 3, 640, 480))) == (640, 480)
    assert get_tensor_spatial_hw(np.zeros((2, 640, 480, 3))) == (640, 480)


def test_billable_hw_prefers_fixed_over_stamped():
    model = SimpleNamespace(img_size_h=420, img_size_w=420)
    stamp_billable_model_input(model, np.zeros((1, 3, 800, 800)))

    assert billable_hw_from_model(model) == (420, 420)


def test_billable_hw_uses_stamped_when_dynamic():
    model = SimpleNamespace()
    stamp_billable_model_input(model, np.zeros((4, 3, 512, 768)))

    assert billable_hw_from_model(model) == (512, 768)
    assert model._usage_billable_frames == 4


def test_count_inference_images():
    assert count_inference_images(None) == 0
    assert count_inference_images(np.zeros((10, 10, 3))) == 1
    assert count_inference_images([1, 2, 3]) == 3


def test_get_fixed_model_input_hw_from_image_size_and_nested_backend():
    assert get_fixed_model_input_hw(SimpleNamespace(image_size=1024)) == (1024, 1024)
    assert get_fixed_model_input_hw(
        SimpleNamespace(_model=SimpleNamespace(_image_size=1008))
    ) == (1008, 1008)
    assert get_fixed_model_input_hw(
        SimpleNamespace(_model=SimpleNamespace(_model=SimpleNamespace(image_size=1024)))
    ) == (1024, 1024)


def test_prepare_sam_usage_billing_stamps_encoder_size():
    model = SimpleNamespace(image_size=1024)
    request = SimpleNamespace(image=object())

    prepare_sam_usage_billing(model, request)

    assert model._usage_billable_input_hw == (1024, 1024)
    assert model._usage_billable_frames == 1
    assert billable_hw_from_model(model) == (1024, 1024)


def test_sam_model_decorator_records_megapixel_buckets(
    usage_collector_with_mocked_threads,
):
    from inference.usage_tracking.collector import usage_collector

    collector = usage_collector_with_mocked_threads

    class SamLikeModel:
        api_key = "test_key"
        dataset_id = "sam2"
        version_id = "hiera_tiny"
        task_type = "unsupervised-segmentation"
        model_type = "sam2"
        image_size = 1024

        @usage_collector(category="model")
        def infer_from_request(self, request):
            prepare_sam_usage_billing(self, request)
            return {"ok": True}

    SamLikeModel().infer_from_request(
        SimpleNamespace(image=object(), api_key="test_key")
    )

    key = f"model:sam2/hiera_tiny:billable=true:outcome=success"
    row = collector._usage["test_key"][key]
    # 1024x1024 = 1.048576 MP -> 1-2 bucket
    assert row["megapixel_buckets"]["1-2"]["processed_frames"] == 1
    assert row["processed_frames"] == 1
