"""Unit tests for ``SAM3ForStream`` and its helpers.

Avoids ``from_pretrained`` — builds the class directly with mocked
model and processor so the tests run without real weights / GPU.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.sam3_rt.sam3_pytorch import (
    _SESSION_KEY,
    SAM3ForStream,
    _ensure_numpy_image,
    _extract_object_ids,
    _first_present,
    _normalise_bboxes,
    _to_numpy_binary_masks,
    _unpack_processed_outputs,
)


# ---------------------------------------------------------------------------
# Helpers (pure functions)
# ---------------------------------------------------------------------------


def test_normalise_bboxes_accepts_single_tuple():
    out = _normalise_bboxes((10, 20, 30, 40))
    assert out == [(10.0, 20.0, 30.0, 40.0)]


def test_normalise_bboxes_accepts_list_of_tuples():
    out = _normalise_bboxes([(10, 20, 30, 40), (50, 60, 70, 80)])
    assert out == [(10.0, 20.0, 30.0, 40.0), (50.0, 60.0, 70.0, 80.0)]


def test_normalise_bboxes_reorders_corners():
    # x2 < x1, y2 < y1 — normaliser should swap so we get a proper box.
    out = _normalise_bboxes([(100, 200, 10, 20)])
    assert out == [(10.0, 20.0, 100.0, 200.0)]


def test_normalise_bboxes_drops_malformed_entries():
    out = _normalise_bboxes([(10, 20, 30, 40), None, (1, 2)])
    assert out == [(10.0, 20.0, 30.0, 40.0)]


def test_normalise_bboxes_none_returns_empty():
    assert _normalise_bboxes(None) == []


def test_ensure_numpy_image_passes_ndarray_through():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert _ensure_numpy_image(img) is img


def test_ensure_numpy_image_converts_tensor():
    img = torch.zeros(4, 4, 3, dtype=torch.uint8)
    out = _ensure_numpy_image(img)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 4, 3)


def test_first_present_returns_first_hit():
    d = {"a": None, "b": 7, "c": 9}
    assert _first_present(d, ("a", "b", "c")) == 7


def test_first_present_skips_missing_and_none():
    d = {"a": None, "b": None}
    assert _first_present(d, ("a", "b", "c")) is None


def test_to_numpy_binary_masks_squeezes_channel_dim():
    t = torch.ones(2, 1, 5, 5, dtype=torch.uint8)
    out = _to_numpy_binary_masks(t)
    assert out.shape == (2, 5, 5)
    assert out.dtype == bool
    assert out.all()


def test_to_numpy_binary_masks_lifts_2d_to_3d():
    out = _to_numpy_binary_masks(np.ones((5, 5), dtype=np.uint8))
    assert out.shape == (1, 5, 5)


def test_to_numpy_binary_masks_none_returns_empty():
    out = _to_numpy_binary_masks(None)
    assert out.shape == (0, 0, 0)
    assert out.dtype == bool


def test_extract_object_ids_reads_obj_ids_attr():
    mo = MagicMock(spec=["obj_ids"])
    mo.obj_ids = [3, 4, 5]
    out = _extract_object_ids(mo, n=3)
    assert out.tolist() == [3, 4, 5]


def test_extract_object_ids_falls_back_to_range():
    mo = MagicMock(spec=[])  # no obj_ids / object_ids attrs
    out = _extract_object_ids(mo, n=3)
    assert out.tolist() == [0, 1, 2]


def test_unpack_processed_outputs_handles_dict_form():
    processed = {
        "masks": np.ones((2, 4, 4), dtype=bool),
        "obj_ids": [11, 12],
    }
    masks, ids = _unpack_processed_outputs(processed)
    assert masks.shape == (2, 4, 4)
    assert ids.tolist() == [11, 12]


def test_unpack_processed_outputs_handles_list_of_dicts():
    processed = [
        {"mask": np.ones((3, 3), dtype=bool), "obj_id": 0},
        {"mask": np.zeros((3, 3), dtype=bool), "obj_id": 1},
    ]
    masks, ids = _unpack_processed_outputs(processed)
    assert masks.shape == (2, 3, 3)
    assert ids.tolist() == [0, 1]


def test_unpack_processed_outputs_none_returns_sentinel():
    masks, ids = _unpack_processed_outputs(None)
    assert masks is None
    assert ids.shape == (0,)


def test_unpack_processed_outputs_dict_without_masks_returns_none():
    masks, ids = _unpack_processed_outputs({"obj_ids": [1, 2]})
    assert masks is None


# ---------------------------------------------------------------------------
# Class behaviour (mocked model + processor)
# ---------------------------------------------------------------------------


class _FakeInferenceSession:
    """Stand-in for a HF Sam3Video inference session."""


def _build_model_with_mocks():
    fake_session = _FakeInferenceSession()
    model = MagicMock(name="Sam3VideoModel")
    processor = MagicMock(name="Sam3VideoProcessor")
    processor.init_video_session.return_value = fake_session
    processor.add_text_prompt.side_effect = (
        lambda inference_session, text: inference_session
    )

    def _processor_call(images, device=None, return_tensors="pt"):
        h, w = images.shape[:2]
        result = MagicMock()
        result.pixel_values = torch.zeros(1, 3, h, w)
        result.original_sizes = [(h, w)]
        return result

    processor.side_effect = _processor_call
    # Force the fallback path in _extract_masks_and_ids so we don't have
    # to mock postprocess_outputs's return shape.
    processor.postprocess_outputs = None

    def _post_process_masks(masks_list, original_sizes, binarize):
        h, w = original_sizes[0]
        return [torch.ones(1, h, w, dtype=torch.bool)]

    processor.post_process_masks.side_effect = _post_process_masks

    model_outputs = MagicMock(spec=["pred_masks", "obj_ids"])
    model_outputs.pred_masks = torch.ones(1, 1, 1, 1)
    model_outputs.obj_ids = [0]
    model.return_value = model_outputs

    sam3 = SAM3ForStream(
        model=model,
        processor=processor,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return sam3, model, processor, fake_session


def test_track_without_state_raises():
    sam3, _, _, _ = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    with pytest.raises(ModelRuntimeError):
        sam3.track(image=frame, state_dict=None)


def test_prompt_returns_state_dict_with_session_handle():
    sam3, _, processor, fake_session = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    masks, obj_ids, state = sam3.prompt(
        image=frame, bboxes=[(1, 2, 10, 12)], clear_old_prompts=True
    )

    assert masks.shape == (1, 32, 48)
    assert state[_SESSION_KEY] is fake_session
    processor.add_inputs_to_inference_session.assert_called_once()
    assert processor.init_video_session.call_count == 1


def test_prompt_with_clear_old_prompts_false_reuses_session_from_state_dict():
    sam3, _, processor, fake_session = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    # Pre-built state_dict pointing at the fake session.
    preexisting_state = {_SESSION_KEY: fake_session}
    sam3.prompt(
        image=frame,
        bboxes=[(1, 2, 10, 12)],
        state_dict=preexisting_state,
        clear_old_prompts=False,
    )
    # When reusing state, no new session should be initialised.
    processor.init_video_session.assert_not_called()


def test_track_reuses_session_from_state_dict():
    sam3, _, processor, fake_session = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    # Seed a session via prompt first, grab the state dict.
    _, _, state = sam3.prompt(image=frame, bboxes=[(1, 2, 10, 12)])
    processor.init_video_session.reset_mock()

    # Now a track call with the returned state must NOT start a new session.
    masks, obj_ids, new_state = sam3.track(image=frame, state_dict=state)
    processor.init_video_session.assert_not_called()
    assert new_state[_SESSION_KEY] is fake_session


def test_prompt_accepts_text_only():
    sam3, _, processor, _ = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    sam3.prompt(image=frame, text="person")
    processor.add_text_prompt.assert_called_once()
    processor.add_inputs_to_inference_session.assert_not_called()


def test_prompt_accepts_text_and_boxes_together():
    sam3, _, processor, _ = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    sam3.prompt(
        image=frame,
        text="white square",
        bboxes=[(1, 2, 10, 12)],
    )
    processor.add_text_prompt.assert_called_once()
    processor.add_inputs_to_inference_session.assert_called_once()
