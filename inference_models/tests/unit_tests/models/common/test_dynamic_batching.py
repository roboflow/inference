import gc
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from threading import Event, Lock
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch

from inference_models import configuration
from inference_models.models.common.dynamic_batching import (
    DynamicBatcher,
    DynamicBatchingMixin,
    InputsCollationError,
    collate_left_padded_inputs,
    strip_trailing_padding,
)

PAD_TOKEN_ID = 0


def wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return None
        time.sleep(0.005)
    raise AssertionError("Condition not met before timeout")


def request_inputs(*token_values: int) -> dict:
    input_ids = torch.tensor([list(token_values)], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def expected_result(last_token_value: int, max_new_tokens: int) -> list:
    # mirrors ControllableRunner output for a single-row request
    return [
        [last_token_value * 10 + step for step in range(1, max_new_tokens + 1)]
    ]


class ControllableRunner:
    """
    Fake runner standing in for the wrappers' locked `generate()` body.

    For each input row it emits `max_new_tokens` values derived from the
    row's last (non-padded) token, so per-request results are identical
    whether the request ran alone or inside a batch.
    """

    def __init__(
        self,
        gate_for_first_call: Optional[Event] = None,
        poison_value: Optional[int] = None,
        batched_calls_raise: Optional[Exception] = None,
    ):
        self.calls = []
        self._calls_lock = Lock()
        self._gate = gate_for_first_call
        self._poison_value = poison_value
        self._batched_calls_raise = batched_calls_raise

    def __call__(self, inputs: dict, gen_kwargs: dict) -> torch.Tensor:
        with self._calls_lock:
            self.calls.append((inputs, dict(gen_kwargs)))
            gate, self._gate = self._gate, None
        if gate is not None:
            assert gate.wait(timeout=10.0), "Runner gate never released"
        input_ids = inputs["input_ids"]
        if self._batched_calls_raise is not None and input_ids.shape[0] > 1:
            raise self._batched_calls_raise
        if self._poison_value is not None and bool(
            (input_ids == self._poison_value).any()
        ):
            raise ValueError("poisoned request")
        max_new_tokens = gen_kwargs["max_new_tokens"]
        rows = [
            [int(row[-1]) * 10 + step for step in range(1, max_new_tokens + 1)]
            for row in input_ids
        ]
        return torch.tensor(rows, dtype=torch.long)


def make_batcher(
    runner,
    max_batch_size: int = 8,
    max_wait_ms: float = 30,
    result_timeout_s: float = 10.0,
    pad_token_id: Optional[int] = PAD_TOKEN_ID,
    eos_token_id: Optional[int] = None,
) -> DynamicBatcher:
    return DynamicBatcher(
        runner=runner,
        collate_inputs=lambda inputs_list: collate_left_padded_inputs(
            inputs_list=inputs_list,
            pad_token_id=pad_token_id,
        ),
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        result_timeout_s=result_timeout_s,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        name="test-model",
    )


def test_single_request_passes_through_unbatched() -> None:
    # given
    runner = ControllableRunner()
    batcher = make_batcher(runner=runner)

    # when
    result = batcher.submit(
        inputs=request_inputs(1, 2, 3),
        gen_kwargs={"max_new_tokens": 4, "do_sample": False},
    )

    # then
    assert result.tolist() == expected_result(3, 4)
    assert len(runner.calls) == 1
    assert runner.calls[0][0]["input_ids"].tolist() == [[1, 2, 3]]
    assert runner.calls[0][1] == {"max_new_tokens": 4, "do_sample": False}


def test_concurrent_requests_are_collected_into_one_batched_call() -> None:
    # given
    gate = Event()
    runner = ControllableRunner(gate_for_first_call=gate)
    batcher = make_batcher(runner=runner)

    with ThreadPoolExecutor(max_workers=4) as executor:
        # when - first request occupies the batcher thread inside the runner
        first = executor.submit(
            batcher.submit, request_inputs(1, 2, 3), {"max_new_tokens": 3, "do_sample": False}
        )
        wait_for(lambda: len(runner.calls) == 1)
        # two more requests queue up behind the in-flight one
        second = executor.submit(
            batcher.submit, request_inputs(5), {"max_new_tokens": 2, "do_sample": False}
        )
        third = executor.submit(
            batcher.submit, request_inputs(6, 7), {"max_new_tokens": 4, "do_sample": False}
        )
        wait_for(lambda: batcher._queue.qsize() == 2)
        gate.set()

        # then - each result matches the single-request contract
        assert first.result(timeout=5).tolist() == expected_result(3, 3)
        assert second.result(timeout=5).tolist() == expected_result(5, 2)
        assert third.result(timeout=5).tolist() == expected_result(7, 4)

    # then - queued requests ran as one batched call at max(max_new_tokens),
    # with the shorter request left-padded
    assert len(runner.calls) == 2
    batched_inputs, batched_gen_kwargs = runner.calls[1]
    assert batched_inputs["input_ids"].tolist() == [[PAD_TOKEN_ID, 5], [6, 7]]
    assert batched_inputs["attention_mask"].tolist() == [[0, 1], [1, 1]]
    assert batched_gen_kwargs == {"max_new_tokens": 4, "do_sample": False}
    assert batcher.stats["batched_requests"] == 2


def test_requests_with_different_sampling_params_do_not_share_a_batch() -> None:
    # given
    gate = Event()
    runner = ControllableRunner(gate_for_first_call=gate)
    batcher = make_batcher(runner=runner)

    with ThreadPoolExecutor(max_workers=4) as executor:
        # when
        first = executor.submit(
            batcher.submit, request_inputs(1), {"max_new_tokens": 2, "do_sample": False}
        )
        wait_for(lambda: len(runner.calls) == 1)
        second = executor.submit(
            batcher.submit, request_inputs(5), {"max_new_tokens": 2, "do_sample": False}
        )
        third = executor.submit(
            batcher.submit, request_inputs(6), {"max_new_tokens": 2, "do_sample": True}
        )
        wait_for(lambda: batcher._queue.qsize() == 2)
        gate.set()

        # then
        assert first.result(timeout=5).tolist() == expected_result(1, 2)
        assert second.result(timeout=5).tolist() == expected_result(5, 2)
        assert third.result(timeout=5).tolist() == expected_result(6, 2)

    # then - do_sample=True must not batch with do_sample=False
    assert len(runner.calls) == 3
    assert all(call[0]["input_ids"].shape[0] == 1 for call in runner.calls)


def test_poisoned_request_only_fails_its_own_submitter() -> None:
    # given
    gate = Event()
    runner = ControllableRunner(gate_for_first_call=gate, poison_value=666)
    batcher = make_batcher(runner=runner)

    with ThreadPoolExecutor(max_workers=4) as executor:
        # when
        first = executor.submit(
            batcher.submit, request_inputs(1), {"max_new_tokens": 2, "do_sample": False}
        )
        wait_for(lambda: len(runner.calls) == 1)
        good = executor.submit(
            batcher.submit, request_inputs(5), {"max_new_tokens": 2, "do_sample": False}
        )
        poisoned = executor.submit(
            batcher.submit, request_inputs(666), {"max_new_tokens": 2, "do_sample": False}
        )
        another_good = executor.submit(
            batcher.submit, request_inputs(7), {"max_new_tokens": 2, "do_sample": False}
        )
        wait_for(lambda: batcher._queue.qsize() == 3)
        gate.set()

        # then - the batched call fails, members re-run serially and only the
        # poisoned request observes the exception
        assert first.result(timeout=5).tolist() == expected_result(1, 2)
        assert good.result(timeout=5).tolist() == expected_result(5, 2)
        assert another_good.result(timeout=5).tolist() == expected_result(7, 2)
        with pytest.raises(ValueError):
            poisoned.result(timeout=5)

    # then - 1 single + 1 failed batched + 3 serial re-runs
    assert len(runner.calls) == 5
    assert batcher.stats["serial_fallbacks"] == 1


def test_cuda_oom_on_batched_call_triggers_cache_cleanup_and_serial_retry(
    monkeypatch,
) -> None:
    # given
    empty_cache_calls = []
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: empty_cache_calls.append(1)
    )
    gate = Event()
    runner = ControllableRunner(
        gate_for_first_call=gate,
        batched_calls_raise=torch.cuda.OutOfMemoryError("CUDA out of memory"),
    )
    batcher = make_batcher(runner=runner)

    with ThreadPoolExecutor(max_workers=4) as executor:
        # when
        first = executor.submit(
            batcher.submit, request_inputs(1), {"max_new_tokens": 2, "do_sample": False}
        )
        wait_for(lambda: len(runner.calls) == 1)
        second = executor.submit(
            batcher.submit, request_inputs(5), {"max_new_tokens": 2, "do_sample": False}
        )
        third = executor.submit(
            batcher.submit, request_inputs(6), {"max_new_tokens": 2, "do_sample": False}
        )
        wait_for(lambda: batcher._queue.qsize() == 2)
        gate.set()

        # then - both requests recover through the serial path
        assert first.result(timeout=5).tolist() == expected_result(1, 2)
        assert second.result(timeout=5).tolist() == expected_result(5, 2)
        assert third.result(timeout=5).tolist() == expected_result(6, 2)

    assert len(empty_cache_calls) == 1
    assert batcher.stats["serial_fallbacks"] == 1


def test_submit_times_out_when_runner_never_returns() -> None:
    # given
    gate = Event()
    runner = ControllableRunner(gate_for_first_call=gate)
    batcher = make_batcher(runner=runner, result_timeout_s=0.2)

    # when / then
    try:
        with pytest.raises(FuturesTimeoutError):
            batcher.submit(
                inputs=request_inputs(1),
                gen_kwargs={"max_new_tokens": 2, "do_sample": False},
            )
    finally:
        gate.set()


def test_concurrent_submit_stress() -> None:
    # given
    runner = ControllableRunner()
    batcher = make_batcher(runner=runner, max_batch_size=8, max_wait_ms=5)

    # when
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {
            executor.submit(
                batcher.submit,
                request_inputs(value),
                {"max_new_tokens": 3, "do_sample": False},
            ): value
            for value in range(1, 65)
        }
        # then - every submitter gets its own result back
        for future, value in futures.items():
            assert future.result(timeout=30).tolist() == expected_result(value, 3)

    # then - no request was lost or duplicated
    total_rows = sum(call[0]["input_ids"].shape[0] for call in runner.calls)
    assert total_rows == 64
    assert batcher.stats["requests"] == 64
    assert batcher.stats["max_batch_size_seen"] <= 8


def test_batcher_thread_survives_collate_errors() -> None:
    # given - a collate raising on every batched attempt
    gate = Event()
    runner = ControllableRunner(gate_for_first_call=gate)

    def broken_collate(inputs_list):
        raise RuntimeError("collate exploded")

    batcher = DynamicBatcher(
        runner=runner,
        collate_inputs=broken_collate,
        max_batch_size=8,
        max_wait_ms=30,
        result_timeout_s=10.0,
        pad_token_id=PAD_TOKEN_ID,
        name="test-model",
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        # when
        first = executor.submit(
            batcher.submit, request_inputs(1), {"max_new_tokens": 2, "do_sample": False}
        )
        wait_for(lambda: len(runner.calls) == 1)
        second = executor.submit(
            batcher.submit, request_inputs(5), {"max_new_tokens": 2, "do_sample": False}
        )
        third = executor.submit(
            batcher.submit, request_inputs(6), {"max_new_tokens": 2, "do_sample": False}
        )
        wait_for(lambda: batcher._queue.qsize() == 2)
        gate.set()

        # then - collate failure falls back to serial execution
        assert first.result(timeout=5).tolist() == expected_result(1, 2)
        assert second.result(timeout=5).tolist() == expected_result(5, 2)
        assert third.result(timeout=5).tolist() == expected_result(6, 2)

    # then - the batcher thread is still alive and serving requests
    assert batcher._thread.is_alive()
    result = batcher.submit(
        inputs=request_inputs(9), gen_kwargs={"max_new_tokens": 2, "do_sample": False}
    )
    assert result.tolist() == expected_result(9, 2)


def test_strip_trailing_padding_removes_shared_trailing_padding() -> None:
    # given
    generated_ids = torch.tensor(
        [
            [11, 12, PAD_TOKEN_ID, PAD_TOKEN_ID],
            [21, 22, 23, PAD_TOKEN_ID],
        ]
    )

    # when
    result = strip_trailing_padding(
        generated_ids=generated_ids, pad_token_id=PAD_TOKEN_ID, eos_token_id=2
    )

    # then - padding making rows of the same request even is preserved,
    # only the columns padded by other batch members are dropped
    assert result.tolist() == [[11, 12, PAD_TOKEN_ID], [21, 22, 23]]


def test_strip_trailing_padding_keeps_single_eos_when_pad_equals_eos() -> None:
    # given
    generated_ids = torch.tensor([[11, 12, 2, 2, 2]])

    # when
    result = strip_trailing_padding(
        generated_ids=generated_ids, pad_token_id=2, eos_token_id=2
    )

    # then - a finished sequence keeps exactly one terminating EOS
    assert result.tolist() == [[11, 12, 2]]


def test_strip_trailing_padding_without_pad_token_returns_input() -> None:
    # given
    generated_ids = torch.tensor([[11, 12, 0, 0]])

    # when
    result = strip_trailing_padding(
        generated_ids=generated_ids, pad_token_id=None, eos_token_id=None
    )

    # then
    assert result.tolist() == [[11, 12, 0, 0]]


def test_strip_trailing_padding_keeps_at_least_one_column() -> None:
    # given
    generated_ids = torch.tensor([[PAD_TOKEN_ID, PAD_TOKEN_ID]])

    # when
    result = strip_trailing_padding(
        generated_ids=generated_ids, pad_token_id=PAD_TOKEN_ID, eos_token_id=2
    )

    # then
    assert result.tolist() == [[PAD_TOKEN_ID]]


def test_collate_qwen_family_shapes() -> None:
    # given - qwen-family: `pixel_values` rows are flattened patches,
    # `image_grid_thw` is [num_images, 3]
    first = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "pixel_values": torch.randn(12, 7),
        "image_grid_thw": torch.tensor([[1, 3, 4]]),
    }
    second = {
        "input_ids": torch.tensor([[6, 7, 8]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.randn(8, 7),
        "image_grid_thw": torch.tensor([[1, 2, 4]]),
    }

    # when
    collated = collate_left_padded_inputs(
        inputs_list=[first, second],
        pad_token_id=PAD_TOKEN_ID,
        cat_keys=("pixel_values", "image_grid_thw"),
    )

    # then
    assert collated["input_ids"].tolist() == [
        [1, 2, 3, 4, 5],
        [PAD_TOKEN_ID, PAD_TOKEN_ID, 6, 7, 8],
    ]
    assert collated["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]
    assert collated["pixel_values"].shape == (20, 7)
    assert torch.equal(collated["pixel_values"][:12], first["pixel_values"])
    assert torch.equal(collated["pixel_values"][12:], second["pixel_values"])
    assert collated["image_grid_thw"].tolist() == [[1, 3, 4], [1, 2, 4]]


def test_collate_smolvlm_family_shapes() -> None:
    # given - smolvlm: `pixel_values` is [1, num_images, 3, H, W] with
    # `pixel_attention_mask` [1, num_images, H, W]; num_images varies
    first = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.randn(1, 2, 3, 8, 8),
        "pixel_attention_mask": torch.ones(1, 2, 8, 8, dtype=torch.long),
    }
    second = {
        "input_ids": torch.tensor([[4, 5, 6]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.randn(1, 4, 3, 8, 8),
        "pixel_attention_mask": torch.ones(1, 4, 8, 8, dtype=torch.long),
    }

    # when
    collated = collate_left_padded_inputs(
        inputs_list=[first, second],
        pad_token_id=PAD_TOKEN_ID,
        frame_pad_keys=("pixel_values", "pixel_attention_mask"),
    )

    # then - the shorter request gets all-zero image padding (which the
    # Idefics3-style model drops), masks padded with zeros
    assert collated["pixel_values"].shape == (2, 4, 3, 8, 8)
    assert torch.equal(collated["pixel_values"][0, :2], first["pixel_values"][0])
    assert torch.all(collated["pixel_values"][0, 2:] == 0)
    assert torch.equal(collated["pixel_values"][1], second["pixel_values"][0])
    assert collated["pixel_attention_mask"].shape == (2, 4, 8, 8)
    assert torch.all(collated["pixel_attention_mask"][0, 2:] == 0)
    assert torch.all(collated["pixel_attention_mask"][0, :2] == 1)


def test_collate_paligemma_family_shapes() -> None:
    # given - paligemma: fixed-resolution `pixel_values` [batch, 3, H, W],
    # requests may carry multiple rows (one per image)
    first = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 8, 8),
    }
    second = {
        "input_ids": torch.tensor([[4, 5], [6, 7]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "pixel_values": torch.randn(2, 3, 8, 8),
    }

    # when
    collated = collate_left_padded_inputs(
        inputs_list=[first, second],
        pad_token_id=PAD_TOKEN_ID,
        cat_keys=("pixel_values",),
    )

    # then
    assert collated["input_ids"].tolist() == [
        [1, 2, 3],
        [PAD_TOKEN_ID, 4, 5],
        [PAD_TOKEN_ID, 6, 7],
    ]
    assert collated["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1], [0, 1, 1]]
    assert collated["pixel_values"].shape == (3, 3, 8, 8)


def test_collate_rejects_inconsistent_keys() -> None:
    # given
    first = request_inputs(1, 2)
    second = {"input_ids": torch.tensor([[3, 4]])}

    # when / then
    with pytest.raises(InputsCollationError):
        collate_left_padded_inputs(
            inputs_list=[first, second], pad_token_id=PAD_TOKEN_ID
        )


def test_collate_rejects_unsupported_keys() -> None:
    # given
    first = request_inputs(1, 2)
    first["token_type_ids"] = torch.zeros(1, 2, dtype=torch.long)
    second = request_inputs(3, 4)
    second["token_type_ids"] = torch.zeros(1, 2, dtype=torch.long)

    # when / then
    with pytest.raises(InputsCollationError):
        collate_left_padded_inputs(
            inputs_list=[first, second], pad_token_id=PAD_TOKEN_ID
        )


def test_collate_requires_pad_token_when_padding_is_needed() -> None:
    # when / then
    with pytest.raises(InputsCollationError):
        collate_left_padded_inputs(
            inputs_list=[request_inputs(1, 2, 3), request_inputs(4)],
            pad_token_id=None,
        )


def test_collate_requires_attention_mask_when_padding_is_needed() -> None:
    # given
    first = {"input_ids": torch.tensor([[1, 2, 3]])}
    second = {"input_ids": torch.tensor([[4]])}

    # when / then
    with pytest.raises(InputsCollationError):
        collate_left_padded_inputs(
            inputs_list=[first, second], pad_token_id=PAD_TOKEN_ID
        )


def test_collate_frame_padding_rejects_mismatched_image_shapes() -> None:
    # given
    first = request_inputs(1, 2)
    first["pixel_values"] = torch.randn(1, 2, 3, 8, 8)
    second = request_inputs(3, 4)
    second["pixel_values"] = torch.randn(1, 2, 3, 16, 16)

    # when / then
    with pytest.raises(InputsCollationError):
        collate_left_padded_inputs(
            inputs_list=[first, second],
            pad_token_id=PAD_TOKEN_ID,
            frame_pad_keys=("pixel_values",),
        )


class _FakeTokenizer:
    pad_token_id = PAD_TOKEN_ID
    eos_token_id = 2


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class _FakeVLM(DynamicBatchingMixin):
    def __init__(self):
        self._processor = _FakeProcessor()
        self.runner_calls = []

    def _run_locked_generation(self, inputs: dict, gen_kwargs: dict) -> torch.Tensor:
        self.runner_calls.append((inputs, dict(gen_kwargs)))
        rows = [
            [int(row[-1]) * 10 + step for step in range(1, gen_kwargs["max_new_tokens"] + 1)]
            for row in inputs["input_ids"]
        ]
        return torch.tensor(rows, dtype=torch.long)


def test_mixin_dynamic_batching_disabled_by_default() -> None:
    # given
    model = _FakeVLM()

    # when / then
    assert model._dynamic_batching_enabled() is False
    assert getattr(model, "_dynamic_batcher", None) is None


def test_mixin_reflects_configuration_flag(monkeypatch) -> None:
    # given
    model = _FakeVLM()
    monkeypatch.setattr(
        configuration, "INFERENCE_MODELS_DYNAMIC_BATCHING_ENABLED", True
    )

    # when / then
    assert model._dynamic_batching_enabled() is True


def test_mixin_creates_single_batcher_per_instance(monkeypatch) -> None:
    # given
    monkeypatch.setattr(configuration, "INFERENCE_MODELS_DYNAMIC_BATCH_MAX_SIZE", 4)
    model = _FakeVLM()

    # when - racing threads asking for the batcher
    with ThreadPoolExecutor(max_workers=8) as executor:
        batchers = list(
            executor.map(lambda _: model._get_dynamic_batcher(), range(16))
        )

    # then
    assert all(batcher is batchers[0] for batcher in batchers)
    assert batchers[0]._pad_token_id == PAD_TOKEN_ID
    assert batchers[0]._eos_token_id == 2
    assert batchers[0]._max_batch_size == 4


def test_mixin_submit_round_trip(monkeypatch) -> None:
    # given
    monkeypatch.setattr(
        configuration, "INFERENCE_MODELS_DYNAMIC_BATCHING_ENABLED", True
    )
    model = _FakeVLM()

    # when
    result = model._submit_to_dynamic_batcher(
        inputs=request_inputs(3), gen_kwargs={"max_new_tokens": 2, "do_sample": False}
    )

    # then
    assert result.tolist() == expected_result(3, 2)
    assert len(model.runner_calls) == 1


def test_batcher_thread_exits_when_owning_model_is_garbage_collected(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(
        configuration, "INFERENCE_MODELS_DYNAMIC_BATCHING_ENABLED", True
    )
    monkeypatch.setattr(DynamicBatcher, "_IDLE_LIVENESS_INTERVAL_S", 0.05)
    model = _FakeVLM()
    result = model._submit_to_dynamic_batcher(
        inputs=request_inputs(3), gen_kwargs={"max_new_tokens": 2, "do_sample": False}
    )
    assert result.tolist() == expected_result(3, 2)
    batcher = model._get_dynamic_batcher()
    thread = batcher._thread
    assert thread.is_alive()

    # when - the owning wrapper gets released (e.g. model cache eviction)
    del model
    gc.collect()

    # then - the batcher thread does not keep the model alive and exits
    wait_for(lambda: not thread.is_alive(), timeout=5.0)


def test_glm_ocr_generate_keeps_lock_path_when_batching_disabled() -> None:
    # given
    from inference_models.models.glm_ocr.glm_ocr_hf import GlmOcrHF

    model = MagicMock()
    model.generate.return_value = torch.tensor([[11, 12, 21, 22]])
    glm_ocr = GlmOcrHF(model=model, processor=MagicMock(), device=MagicMock())

    # when
    result = glm_ocr.generate(
        inputs={"input_ids": torch.tensor([[11, 12]])},
        max_new_tokens=4,
    )

    # then - no batcher is created when the feature flag is off
    assert getattr(glm_ocr, "_dynamic_batcher", None) is None
    assert result.tolist() == [[21, 22]]


def test_glm_ocr_generate_routes_through_batcher_when_enabled(monkeypatch) -> None:
    # given
    from inference_models.models.glm_ocr.glm_ocr_hf import GlmOcrHF

    monkeypatch.setattr(
        configuration, "INFERENCE_MODELS_DYNAMIC_BATCHING_ENABLED", True
    )
    model = MagicMock()
    model.generate.return_value = torch.tensor([[11, 12, 21, 22]])
    processor = MagicMock()
    processor.tokenizer.pad_token_id = PAD_TOKEN_ID
    processor.tokenizer.eos_token_id = 2
    glm_ocr = GlmOcrHF(model=model, processor=processor, device=MagicMock())

    # when
    result = glm_ocr.generate(
        inputs={"input_ids": torch.tensor([[11, 12]])},
        max_new_tokens=4,
    )

    # then
    assert getattr(glm_ocr, "_dynamic_batcher", None) is not None
    assert result.tolist() == [[21, 22]]
    assert model.generate.call_args.kwargs["max_new_tokens"] == 4
