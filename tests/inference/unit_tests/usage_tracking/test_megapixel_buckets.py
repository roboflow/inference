from types import SimpleNamespace

import numpy as np
import pytest

from inference.usage_tracking.decorator_helpers import get_model_megapixel_buckets
from inference.usage_tracking.megapixel_buckets import (
    MEGAPIXEL_BUCKET_UNKNOWN,
    build_megapixel_buckets,
    clear_measured_image_input,
    consume_measured_image_input,
    count_inference_images,
    megapixel_bucket_for_hw,
    parse_image_input_hw,
    record_measured_image_input,
)
from inference.usage_tracking.payload_helpers import (
    merge_megapixel_buckets,
    merge_usage_dicts,
)
from inference.usage_tracking.predict_timing import (
    consume_measured_predict_duration,
    record_measured_predict_duration,
)


@pytest.fixture(autouse=True)
def _clear_measured_input():
    clear_measured_image_input()
    yield
    clear_measured_image_input()


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


def test_parse_image_input_hw_reads_core_img_dims():
    # Core Roboflow preprocess records one (height, width) pair per image.
    assert parse_image_input_hw({"img_dims": [(1080, 1920)]}) == (1080, 1920)
    assert parse_image_input_hw({"img_dims": ((1080, 1920),)}) == (1080, 1920)


def test_parse_image_input_hw_attributes_a_batch_to_its_first_image():
    metadata = {"img_dims": [(1080, 1920), (480, 640)]}

    assert parse_image_input_hw(metadata) == (1080, 1920)


def test_parse_image_input_hw_accepts_an_unwrapped_pair():
    assert parse_image_input_hw({"img_dims": (1080, 1920)}) == (1080, 1920)


def test_parse_image_input_hw_reads_vlm_image_dims_as_width_height():
    # The VLM and depth families record a single (width, height) pair.
    assert parse_image_input_hw({"image_dims": (1920, 1080)}) == (1080, 1920)


def test_parse_image_input_hw_reads_inference_models_original_size():
    # The inference_models detection / segmentation / keypoint adapters return a
    # per-image metadata record rather than a dims key.
    metadata = [
        SimpleNamespace(original_size=SimpleNamespace(height=1080, width=1920)),
        SimpleNamespace(original_size=SimpleNamespace(height=480, width=640)),
    ]

    assert parse_image_input_hw(metadata) == (1080, 1920)


def test_parse_image_input_hw_reads_bare_per_image_shapes():
    # The inference_models classification adapter returns np_image.shape[:2].
    assert parse_image_input_hw([(1080, 1920), (480, 640)]) == (1080, 1920)


def test_parse_image_input_hw_rejects_missing_and_malformed_dims():
    assert parse_image_input_hw(None) is None
    assert parse_image_input_hw({}) is None
    assert parse_image_input_hw({"img_dims": []}) is None
    assert parse_image_input_hw({"img_dims": [(0, 1920)]}) is None
    assert parse_image_input_hw({"image_dims": (0, 1080)}) is None
    assert parse_image_input_hw({"img_dims": [(1080, 1920, 3)]}) is None


def test_parse_image_input_hw_reads_attribute_style_metadata():
    assert parse_image_input_hw(SimpleNamespace(img_dims=[(1080, 1920)])) == (
        1080,
        1920,
    )


def test_consume_measured_image_input_clears_value():
    record_measured_image_input((512, 768), frames=2)

    assert consume_measured_image_input() == ((512, 768), 2)
    # A later call that publishes nothing must not inherit the previous size.
    assert consume_measured_image_input() == (None, None)


def test_record_measured_image_input_rejects_unusable_sizes():
    record_measured_image_input((0, 768))

    assert consume_measured_image_input() == (None, None)


def test_count_inference_images():
    assert count_inference_images(None) == 0
    assert count_inference_images(np.zeros((10, 10, 3))) == 1
    assert count_inference_images([1, 2, 3]) == 3


def test_base_inference_publishes_native_size_not_model_input_size():
    from inference.core.models.base import BaseInference

    class CoreStyleModel(BaseInference):
        # A fixed model input size must no longer influence the bucket.
        img_size_h = 640
        img_size_w = 640

        def preprocess(self, image, **kwargs):
            return (
                np.zeros((1, 3, 640, 640), dtype=np.float32),
                {"img_dims": [(1080, 1920)]},
            )

        def predict(self, img_in, **kwargs):
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    # Call the undecorated function so the published value survives for assertion;
    # the usage decorator consumes it.
    BaseInference.infer.__wrapped__(CoreStyleModel(), object())

    measured_hw, _ = consume_measured_image_input()
    assert measured_hw == (1080, 1920)


def test_base_inference_publishes_native_size_for_vlm_preprocess():
    from inference.core.models.base import BaseInference

    class PaligemmaLikeModel(BaseInference):
        def preprocess(self, image, **kwargs):
            # The processor canvas is a model input size and must be ignored.
            return (
                {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)},
                {"image_dims": (1920, 1080)},
            )

        def predict(self, img_in, **kwargs):
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    BaseInference.infer.__wrapped__(PaligemmaLikeModel(), object())

    measured_hw, _ = consume_measured_image_input()
    assert measured_hw == (1080, 1920)


def test_base_inference_publishes_no_size_when_preprocess_records_no_dims():
    from inference.core.models.base import BaseInference

    class UndimensionedModel(BaseInference):
        def preprocess(self, image, **kwargs):
            return np.zeros((2, 3, 512, 768), dtype=np.float32), None

        def predict(self, img_in, **kwargs):
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    BaseInference.infer.__wrapped__(UndimensionedModel(), [object(), object()])

    measured_hw, measured_frames = consume_measured_image_input()
    # The preprocessed tensor is the model's canvas, so it cannot stand in for
    # the image size - but it still answers how many frames were processed.
    assert measured_hw is None
    assert measured_frames == 2


def test_unsized_call_lands_in_the_unknown_bucket():
    buckets = get_model_megapixel_buckets(
        frames=2,
        input_hw=None,
        execution_duration=0.8,
    )

    assert buckets[MEGAPIXEL_BUCKET_UNKNOWN]["processed_frames"] == 2


def test_bucket_duration_prefers_recorded_predict_duration():
    record_measured_predict_duration(0.2)

    buckets = get_model_megapixel_buckets(
        frames=1,
        input_hw=(640, 640),
        execution_duration=1.5,
    )

    assert buckets["0.25-0.5"]["execution_duration"] == pytest.approx(0.2)


def test_bucket_duration_falls_back_to_call_duration_without_predict_phase():
    buckets = get_model_megapixel_buckets(
        frames=1,
        input_hw=(640, 640),
        execution_duration=1.5,
    )

    assert buckets["0.25-0.5"]["execution_duration"] == pytest.approx(1.5)


def test_bucket_duration_is_consumed_even_for_unbucketed_test_run():
    record_measured_predict_duration(0.2)

    assert (
        get_model_megapixel_buckets(
            frames=1,
            input_hw=(640, 640),
            execution_duration=1.5,
            inference_test_run=True,
        )
        == {}
    )
    # The test run reported nothing, so its measurement must not survive into
    # the next call.
    assert consume_measured_predict_duration() is None
