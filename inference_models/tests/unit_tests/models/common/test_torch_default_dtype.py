import threading
from contextlib import nullcontext
from typing import Callable, ContextManager, Generator, List
from unittest import mock

import pytest
import torch
from transformers.modeling_utils import local_torch_dtype

from inference_models.models.auto_loaders.core import load_model_with_capabilities
from inference_models.models.common import torch as common_torch_module
from inference_models.models.common.torch import (
    reset_leaked_torch_default_dtype,
    torch_default_dtype_lock,
)

EVENT_TIMEOUT_SECONDS = 0.5


@pytest.fixture(autouse=True)
def restore_default_dtype() -> Generator[None, None, None]:
    original_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def _run_overlapping_loads(
    guard: Callable[[], ContextManager[None]],
) -> List[BaseException]:
    # Reproduces the interleaving behind the 2026-11-02 incident: load B enters
    # transformers' `local_torch_dtype` while load A is still inside it, so B saves
    # A's bfloat16 as the "original" and restores it after A has exited.
    a_entered = threading.Event()
    b_entered = threading.Event()
    a_exited = threading.Event()
    errors: List[BaseException] = []

    def load_a() -> None:
        try:
            with guard(), local_torch_dtype(torch.bfloat16):
                a_entered.set()
                b_entered.wait(timeout=EVENT_TIMEOUT_SECONDS)
            a_exited.set()
        except BaseException as error:
            errors.append(error)

    def load_b() -> None:
        try:
            a_entered.wait(timeout=EVENT_TIMEOUT_SECONDS)
            with guard(), local_torch_dtype(torch.bfloat16):
                b_entered.set()
                a_exited.wait(timeout=EVENT_TIMEOUT_SECONDS)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=load_a), threading.Thread(target=load_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    return errors


def test_overlapping_local_torch_dtype_leaks_default_dtype_without_lock() -> None:
    # given
    torch.set_default_dtype(torch.float32)

    # when
    errors = _run_overlapping_loads(guard=nullcontext)

    # then
    assert errors == []
    assert torch.get_default_dtype() == torch.bfloat16


def test_overlapping_local_torch_dtype_restores_default_dtype_with_lock() -> None:
    # given
    torch.set_default_dtype(torch.float32)

    # when
    errors = _run_overlapping_loads(guard=torch_default_dtype_lock)

    # then
    assert errors == []
    assert torch.get_default_dtype() == torch.float32


def test_many_concurrent_loads_with_lock_leave_default_dtype_at_float32() -> None:
    # given
    torch.set_default_dtype(torch.float32)
    start = threading.Barrier(8)
    dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64] * 2

    def load(dtype: torch.dtype) -> None:
        start.wait()
        for _ in range(50):
            with torch_default_dtype_lock(), local_torch_dtype(dtype):
                assert torch.get_default_dtype() == dtype

    threads = [threading.Thread(target=load, args=(dtype,)) for dtype in dtypes]

    # when
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    # then
    assert torch.get_default_dtype() == torch.float32


def test_torch_default_dtype_lock_restores_dtype_left_changed_inside() -> None:
    # given
    torch.set_default_dtype(torch.float32)

    # when
    with torch_default_dtype_lock():
        torch.set_default_dtype(torch.bfloat16)

    # then
    assert torch.get_default_dtype() == torch.float32


def test_torch_default_dtype_lock_is_reentrant() -> None:
    # when
    with torch_default_dtype_lock(), torch_default_dtype_lock():
        torch.set_default_dtype(torch.bfloat16)

    # then
    assert torch.get_default_dtype() == torch.float32


def test_reset_leaked_torch_default_dtype_resets_and_warns() -> None:
    # given
    torch.set_default_dtype(torch.bfloat16)

    # when
    with mock.patch.object(common_torch_module, "LOGGER") as logger_mock:
        result = reset_leaked_torch_default_dtype(
            model_name_or_path="some-workspace/cosmos3-model"
        )

    # then
    assert result is True
    assert torch.get_default_dtype() == torch.float32
    logger_mock.warning.assert_called_once()
    assert torch.bfloat16 in logger_mock.warning.call_args.args
    assert "some-workspace/cosmos3-model" in logger_mock.warning.call_args.args


def test_reset_leaked_torch_default_dtype_does_nothing_when_default_is_float32() -> (
    None
):
    # given
    torch.set_default_dtype(torch.float32)

    # when
    with mock.patch.object(common_torch_module, "LOGGER") as logger_mock:
        result = reset_leaked_torch_default_dtype(model_name_or_path="some/model")

    # then
    assert result is False
    assert torch.get_default_dtype() == torch.float32
    logger_mock.warning.assert_not_called()


def test_reset_leaked_torch_default_dtype_skips_while_another_thread_loads() -> None:
    # given
    load_started = threading.Event()
    check_done = threading.Event()

    def load() -> None:
        with torch_default_dtype_lock(), local_torch_dtype(torch.bfloat16):
            load_started.set()
            check_done.wait(timeout=10)

    thread = threading.Thread(target=load)
    thread.start()
    load_started.wait(timeout=10)

    # when
    with mock.patch.object(common_torch_module, "LOGGER") as logger_mock:
        result = reset_leaked_torch_default_dtype(model_name_or_path="some/model")
    dtype_during_load = torch.get_default_dtype()
    check_done.set()
    thread.join(timeout=10)

    # then
    assert result is False
    assert dtype_during_load == torch.bfloat16
    logger_mock.warning.assert_not_called()
    assert torch.get_default_dtype() == torch.float32


def test_load_model_with_capabilities_resets_default_dtype_leaked_by_model_load() -> (
    None
):
    # given
    loaded_model = object()

    class LeakingModel:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs) -> object:
            torch.set_default_dtype(torch.bfloat16)
            return loaded_model

    # when
    with mock.patch.object(common_torch_module, "LOGGER") as logger_mock:
        result = load_model_with_capabilities(LeakingModel, "/local/package")

    # then
    assert result is loaded_model
    assert torch.get_default_dtype() == torch.float32
    logger_mock.warning.assert_called_once()
    assert "/local/package" in logger_mock.warning.call_args.args
