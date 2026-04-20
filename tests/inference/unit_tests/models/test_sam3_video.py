"""Unit tests for the SegmentAnything3Video model wrapper.

Mocks out the ``transformers`` + ``Sam3VideoModel`` / ``Sam3VideoProcessor``
APIs so per-video session logic can be exercised without real weights
or a GPU.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _install_sam3_package_mocks():
    """Mock the native ``sam3`` package submodules imported by
    ``inference.models.sam3.segment_anything3``.  The image-path SAM3
    module is imported as a side effect of
    ``inference.models.sam3.__init__`` even though we only touch the
    video wrapper here."""
    sam3_mock = MagicMock()
    for submodule in [
        "sam3",
        "sam3.eval",
        "sam3.eval.postprocessors",
        "sam3.model",
        "sam3.model.utils",
        "sam3.model.utils.misc",
        "sam3.model.sam3_image_processor",
        "sam3.train",
        "sam3.train.data",
        "sam3.train.data.collator",
        "sam3.train.data.sam3_image_dataset",
        "sam3.train.transforms",
        "sam3.train.transforms.basic_for_api",
    ]:
        sys.modules[submodule] = MagicMock()


def _install_transformers_sam3_mocks():
    """Install just enough of ``transformers`` to satisfy the imports in
    ``inference.models.sam3.segment_anything3_video``.
    """
    transformers_mock = sys.modules.get("transformers") or MagicMock()

    sam3_model_cls = MagicMock(name="Sam3VideoModel")

    def _from_pretrained(path, *args, **kwargs):
        model = MagicMock(name="Sam3VideoModelInstance")
        model.eval = MagicMock(return_value=model)
        model.to = MagicMock(return_value=model)
        return model

    sam3_model_cls.from_pretrained = MagicMock(side_effect=_from_pretrained)

    sam3_processor_cls = MagicMock(name="Sam3VideoProcessor")

    def _processor_from_pretrained(path, *args, **kwargs):
        processor = MagicMock(name="Sam3VideoProcessorInstance")

        def _call(images, device=None, return_tensors="pt"):
            import torch

            h, w = images.shape[:2]
            result = MagicMock()
            result.pixel_values = torch.zeros(1, 3, h, w)
            result.original_sizes = [(h, w)]
            return result

        processor.side_effect = _call

        def _init_session(**kwargs):
            return MagicMock(name="InferenceSession")

        processor.init_video_session.side_effect = _init_session
        processor.add_text_prompt.side_effect = lambda inference_session, text: (
            inference_session
        )
        processor.add_inputs_to_inference_session = MagicMock()
        processor.postprocess_outputs = None  # force fallback path
        processor.post_process_masks.side_effect = (
            lambda masks_list, original_sizes, binarize: [
                _empty_masks(original_sizes[0])
            ]
        )
        return processor

    sam3_processor_cls.from_pretrained = MagicMock(
        side_effect=_processor_from_pretrained
    )

    transformers_mock.Sam3VideoModel = sam3_model_cls
    transformers_mock.Sam3VideoProcessor = sam3_processor_cls
    sys.modules["transformers"] = transformers_mock

    return sam3_model_cls, sam3_processor_cls


def _empty_masks(hw):
    import torch

    return torch.zeros(0, hw[0], hw[1], dtype=torch.bool)


_install_sam3_package_mocks()
_sam3_model_cls, _sam3_processor_cls = _install_transformers_sam3_mocks()

# Also bypass the TransformerModel weight download machinery.
from inference.core.models.roboflow import RoboflowInferenceModel  # noqa: E402
from inference.models.transformers import transformers as _transformers_mod  # noqa: E402


@pytest.fixture
def fake_model():
    with patch.object(
        _transformers_mod.TransformerModel, "cache_model_artefacts", return_value=None
    ), patch.object(
        RoboflowInferenceModel, "cache_file", return_value="/tmp/fake_sam3_weights"
    ):
        from inference.models.sam3.segment_anything3_video import (
            SegmentAnything3Video,
        )

        yield SegmentAnything3Video(model_id="sam3/sam3_video")


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_has_session_false_for_unknown_video_id(fake_model):
    assert fake_model.has_session("video-A") is False


def test_prompt_and_track_creates_session(fake_model):
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    masks, obj_ids = fake_model.prompt_and_track(
        video_id="video-A",
        frame=frame,
        frame_index=0,
        text="person",
        clear_old_prompts=True,
    )
    assert fake_model.has_session("video-A") is True
    # With our mocks, post_process_masks returns an empty tensor so the
    # fallback extractor produces an empty mask array — that's fine for
    # validating lifecycle; real behaviour is covered by integration tests.
    assert masks.dtype == bool
    assert isinstance(obj_ids, np.ndarray)


def test_track_without_prior_prompt_raises(fake_model):
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="No SAM3 video session"):
        fake_model.track(video_id="unknown", frame=frame)


def test_reset_session_clears_state(fake_model):
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    fake_model.prompt_and_track(
        video_id="video-A",
        frame=frame,
        frame_index=0,
        text="person",
        clear_old_prompts=True,
    )
    assert fake_model.has_session("video-A") is True
    fake_model.reset_session("video-A")
    assert fake_model.has_session("video-A") is False


def test_bucket_file_list_matches_transformers_export(fake_model):
    """Declares the concrete file list that operators need to upload."""
    files = fake_model.get_infer_bucket_file_list()
    plain = [f for f in files if isinstance(f, str)]
    assert "config.json" in plain
    assert "preprocessor_config.json" in plain
    # The model weights entry is a regex so we match on that too.
    import re

    has_safetensors_regex = any(
        isinstance(f, re.Pattern) and f.pattern.startswith("model") for f in files
    )
    assert has_safetensors_regex, (
        "Expected a regex pattern for model*.safetensors in the bucket file list"
    )


# ---------------------------------------------------------------------------
# _unpack_processed_outputs helper
# ---------------------------------------------------------------------------


def test_unpack_processed_outputs_accepts_dict_form():
    from inference.models.sam3.segment_anything3_video import (
        _unpack_processed_outputs,
    )

    masks = np.ones((2, 10, 10), dtype=bool)
    processed = {"masks": masks, "obj_ids": [7, 11]}
    out_masks, out_ids = _unpack_processed_outputs(processed)
    assert out_masks.shape == (2, 10, 10)
    assert out_ids.tolist() == [7, 11]


def test_unpack_processed_outputs_accepts_list_of_dicts():
    from inference.models.sam3.segment_anything3_video import (
        _unpack_processed_outputs,
    )

    processed = [
        {"mask": np.ones((5, 5), dtype=bool), "obj_id": 3},
        {"mask": np.zeros((5, 5), dtype=bool), "obj_id": 4},
    ]
    out_masks, out_ids = _unpack_processed_outputs(processed)
    assert out_masks.shape == (2, 5, 5)
    assert out_ids.tolist() == [3, 4]


def test_unpack_processed_outputs_returns_empty_when_no_masks():
    from inference.models.sam3.segment_anything3_video import (
        _unpack_processed_outputs,
    )

    out_masks, out_ids = _unpack_processed_outputs(None)
    assert out_masks is None
    assert out_ids.shape == (0,)
