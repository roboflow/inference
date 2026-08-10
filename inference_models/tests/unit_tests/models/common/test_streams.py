from contextlib import contextmanager
from typing import Generator

import pytest
import torch

from inference_models.models.common.streams import get_cuda_stream, use_cuda_stream


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("mps")])
def test_get_cuda_stream_returns_none_for_non_cuda_device(
    device: torch.device, monkeypatch
) -> None:
    # given
    def raise_if_called(*args, **kwargs):
        raise AssertionError("torch.cuda.Stream must not be used for non-CUDA devices")

    monkeypatch.setattr(torch.cuda, "Stream", raise_if_called)

    # when
    result = get_cuda_stream(device=device, purpose="pre-processing")

    # then
    assert result is None


def test_use_cuda_stream_is_no_op_when_stream_is_none(monkeypatch) -> None:
    # given
    def raise_if_called(*args, **kwargs):
        raise AssertionError("torch.cuda.stream must not be used without a CUDA stream")

    monkeypatch.setattr(torch.cuda, "stream", raise_if_called)

    # when
    with use_cuda_stream(None):
        result = "processed"

    # then
    assert result == "processed"


def test_use_cuda_stream_activates_provided_stream(monkeypatch) -> None:
    # given
    calls = []
    cuda_stream = object()

    @contextmanager
    def track_cuda_stream(stream) -> Generator[None, None, None]:
        calls.append(("enter", stream))
        yield
        calls.append(("exit", stream))

    monkeypatch.setattr(torch.cuda, "stream", track_cuda_stream)

    # when
    with use_cuda_stream(cuda_stream):
        calls.append(("body", cuda_stream))

    # then
    assert calls == [
        ("enter", cuda_stream),
        ("body", cuda_stream),
        ("exit", cuda_stream),
    ]
