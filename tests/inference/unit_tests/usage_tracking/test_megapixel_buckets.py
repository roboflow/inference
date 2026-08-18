from types import SimpleNamespace

import numpy as np
import pytest

from inference.usage_tracking.decorator_helpers import (
    record_fixed_model_input_for_request,
)
from inference.usage_tracking.megapixel_buckets import (
    MEGAPIXEL_BUCKET_UNKNOWN,
    build_megapixel_buckets,
    clear_measured_model_input,
    consume_measured_model_input,
    count_inference_images,
    get_fixed_model_input_hw,
    get_tensor_spatial_hw,
    megapixel_bucket_for_hw,
    parse_image_dims_hw,
    record_measured_model_input,
    resolve_model_input_hw,
)
from inference.usage_tracking.payload_helpers import (
    merge_megapixel_buckets,
    merge_usage_dicts,
)


@pytest.fixture(autouse=True)
def _clear_measured_input():
    clear_measured_model_input()
    yield
    clear_measured_model_input()


def test_megapixel_bucket_boundaries():
    assert megapixel_bucket_for_hw(420, 420) == "0-0.25"
    assert megapixel_bucket_for_hw(640, 640) == "0.25-0.5"
    assert megapixel_bucket_for_hw(1000, 1000) == "0.5-1"
    assert megapixel_bucket_for_hw(1280, 1280) == "1-2"
    assert megapixel_bucket_for_hw(2000, 2000) == "2-4"
    assert megapixel_bucket_for_hw(2500, 2500) == "4-8"
    assert megapixel_bucket_for_hw(3000, 3000) == "8+"


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


def test_build_megapixel_buckets_falls_back_to_unknown_bucket():
    # Frames whose size cannot be resolved still have to be accounted for, so
    # that bucket frames always sum to the row's processed_frames.
    buckets = build_megapixel_buckets(
        height=None,
        width=None,
        frames=4,
        execution_duration=0.8,
    )

    assert buckets == {
        MEGAPIXEL_BUCKET_UNKNOWN: {
            "processed_frames": 4,
            "execution_duration": 0.8,
        }
    }


def test_build_megapixel_buckets_is_empty_without_frames():
    assert (
        build_megapixel_buckets(height=640, width=640, frames=0, execution_duration=0.1)
        == {}
    )


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


def test_get_tensor_spatial_hw_reads_hf_pixel_values():
    pixel_values = np.zeros((1, 3, 224, 224), dtype=np.float32)

    assert get_tensor_spatial_hw({"pixel_values": pixel_values}) == (224, 224)
    assert get_tensor_spatial_hw(SimpleNamespace(pixel_values=pixel_values)) == (
        224,
        224,
    )


def test_get_tensor_spatial_hw_ignores_non_spatial_pixel_values():
    # Qwen-style patch tokens: no channel axis, must not be treated as H×W.
    assert (
        get_tensor_spatial_hw({"pixel_values": np.zeros((256, 1176), dtype=np.float32)})
        is None
    )


def test_parse_image_dims_hw_converts_width_height():
    assert parse_image_dims_hw({"image_dims": (1920, 1080)}) == (1080, 1920)
    assert parse_image_dims_hw(None) is None
    assert parse_image_dims_hw({"image_dims": (0, 1080)}) is None


def test_input_hw_prefers_fixed_size_over_measured():
    model = SimpleNamespace(img_size_h=420, img_size_w=420)
    record_measured_model_input(np.zeros((1, 3, 800, 800)))
    measured_hw, _ = consume_measured_model_input()

    assert resolve_model_input_hw(model, measured_hw=measured_hw) == (420, 420)


def test_input_hw_uses_measured_size_when_dynamic():
    record_measured_model_input(np.zeros((4, 3, 512, 768)))
    measured_hw, measured_frames = consume_measured_model_input()

    assert measured_frames == 4
    assert resolve_model_input_hw(SimpleNamespace(), measured_hw=measured_hw) == (
        512,
        768,
    )


def test_fixed_input_hw_reads_non_square_image_size_pair():
    # OwlV2 exposes image_size as a (height, width) pair rather than an edge length.
    model = SimpleNamespace(image_size=(960, 1024))

    assert get_fixed_model_input_hw(model) == (960, 1024)


def test_fixed_input_hw_reads_square_image_size_scalar():
    assert get_fixed_model_input_hw(SimpleNamespace(image_size=768)) == (768, 768)


def test_fixed_input_hw_ignores_malformed_image_size():
    assert get_fixed_model_input_hw(SimpleNamespace(image_size=(960,))) is None
    assert get_fixed_model_input_hw(SimpleNamespace(image_size=(0, 640))) is None
    assert get_fixed_model_input_hw(SimpleNamespace(image_size=(None, None))) is None


def test_fixed_input_hw_reads_wrapped_owlv2_backend():
    # SerializedOwlV2 delegates inference to a wrapped OwlV2 instance.
    model = SimpleNamespace(owlv2=SimpleNamespace(image_size=(960, 960)))

    assert get_fixed_model_input_hw(model) == (960, 960)


def test_consume_measured_model_input_clears_value():
    record_measured_model_input(np.zeros((1, 3, 512, 512)))

    assert consume_measured_model_input() == ((512, 512), 1)
    # A later call that publishes nothing must not inherit the previous size.
    assert consume_measured_model_input() == (None, None)


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


def test_fixed_input_hw_reads_hf_processor_size():
    model = SimpleNamespace(
        processor=SimpleNamespace(
            image_processor=SimpleNamespace(size={"height": 224, "width": 224})
        )
    )

    assert get_fixed_model_input_hw(model) == (224, 224)
    assert resolve_model_input_hw(model, measured_hw=(1080, 1920)) == (224, 224)


def test_fixed_input_hw_reads_hf_processor_size_object():
    model = SimpleNamespace(
        processor=SimpleNamespace(
            image_processor=SimpleNamespace(size=SimpleNamespace(height=448, width=448))
        )
    )

    assert get_fixed_model_input_hw(model) == (448, 448)


def test_fixed_input_hw_reads_hf_vision_config_image_size():
    model = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(image_size=224))
        )
    )

    assert get_fixed_model_input_hw(model) == (224, 224)


def test_legacy_hf_vlm_uses_processor_size_not_native_image_dims():
    from inference.core.models.base import BaseInference

    class LegacyHFVLM(BaseInference):
        def __init__(self):
            self.processor = SimpleNamespace(
                image_processor=SimpleNamespace(size={"height": 224, "width": 224})
            )

        def preprocess(self, image, **kwargs):
            return object(), {"image_dims": (1920, 1080)}

        def predict(self, img_in, **kwargs):
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    model = LegacyHFVLM()
    BaseInference.infer.__wrapped__(model, object())
    measured_hw, _ = consume_measured_model_input()

    assert measured_hw == (1080, 1920)
    assert resolve_model_input_hw(model, measured_hw=measured_hw) == (224, 224)


def test_record_fixed_model_input_for_request_publishes_encoder_size():
    record_fixed_model_input_for_request(
        SimpleNamespace(image_size=1024),
        SimpleNamespace(image=object()),
    )

    assert consume_measured_model_input() == ((1024, 1024), 1)


def test_record_fixed_model_input_for_request_clears_stale_value_for_unknown_encoder():
    record_measured_model_input(np.zeros((1, 3, 4000, 4000)))

    record_fixed_model_input_for_request(
        SimpleNamespace(), SimpleNamespace(image=object())
    )

    assert consume_measured_model_input() == (None, None)


def test_base_inference_publishes_preprocessed_input_size():
    from inference.core.models.base import BaseInference

    class DynamicInputModel(BaseInference):
        def preprocess(self, image, **kwargs):
            return np.zeros((2, 3, 512, 768), dtype=np.float32), None

        def predict(self, img_in, **kwargs):
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    # Call the undecorated function so the published value survives for assertion;
    # the usage decorator consumes it.
    BaseInference.infer.__wrapped__(DynamicInputModel(), [object(), object()])

    assert consume_measured_model_input() == ((512, 768), 2)


def test_record_measured_model_input_reads_hf_pixel_values():
    record_measured_model_input(
        {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)},
        fallback_hw=(1080, 1920),
    )

    assert consume_measured_model_input() == ((224, 224), 1)


def test_record_measured_model_input_falls_back_to_image_dims_for_unreadable_pixel_values():
    record_measured_model_input(
        {"pixel_values": np.zeros((256, 1176), dtype=np.float32)},
        fallback_hw=(1080, 1920),
    )

    assert consume_measured_model_input() == ((1080, 1920), None)


def test_record_measured_model_input_uses_image_dims_when_no_model_tensor():
    record_measured_model_input(object(), fallback_hw=(1080, 1920))

    assert consume_measured_model_input() == ((1080, 1920), None)


def test_base_inference_publishes_hf_pixel_values_not_native_image_dims():
    from inference.core.models.base import BaseInference

    class PaligemmaLikeModel(BaseInference):
        def preprocess(self, image, **kwargs):
            return (
                {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)},
                {"image_dims": (1920, 1080)},
            )

        def predict(self, img_in, **kwargs):
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    BaseInference.infer.__wrapped__(PaligemmaLikeModel(), object())

    assert consume_measured_model_input() == ((224, 224), 1)


def test_base_inference_publishes_image_dims_when_preprocess_has_no_tensor():
    from inference.core.models.base import BaseInference

    class NativeSizeModel(BaseInference):
        def preprocess(self, image, **kwargs):
            return object(), {"image_dims": (1920, 1080)}

        def predict(self, img_in, **kwargs):
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    BaseInference.infer.__wrapped__(NativeSizeModel(), object())

    assert consume_measured_model_input() == ((1080, 1920), None)


def test_base_inference_falls_back_to_image_dims_when_pixel_values_are_unreadable():
    from inference.core.models.base import BaseInference

    class PatchTokenModel(BaseInference):
        def preprocess(self, image, **kwargs):
            return (
                {"pixel_values": np.zeros((256, 1176), dtype=np.float32)},
                {"image_dims": (1920, 1080)},
            )

        def predict(self, img_in, **kwargs):
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    BaseInference.infer.__wrapped__(PatchTokenModel(), object())

    assert consume_measured_model_input() == ((1080, 1920), None)
