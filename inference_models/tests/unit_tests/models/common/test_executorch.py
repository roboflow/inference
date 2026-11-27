import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from inference_models.errors import MissingDependencyError
from inference_models.models.common import executorch


class RecordingMethod:
    """Stand-in for an ExecuTorch method exported for a static batch size."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.calls = []
        self.devices = []

    def execute(self, inputs, clone_outputs):
        assert clone_outputs is True
        (chunk,) = inputs
        assert chunk.shape[0] == self.batch_size
        assert chunk.is_contiguous()
        self.calls.append(chunk.clone())
        self.devices.append(chunk.device)
        per_image_sum = chunk.flatten(start_dim=1).sum(dim=1)

        return [per_image_sum[:, None], per_image_sum[:, None, None] * 2]


def test_run_executorch_method_pads_last_chunk_and_drops_padding() -> None:
    # given
    method = RecordingMethod(batch_size=2)
    inputs = torch.arange(5, dtype=torch.float32)[:, None, None, None].expand(
        5, 3, 2, 2
    )

    # when
    first, second = executorch.run_executorch_method_with_static_batch_size(
        method,
        inputs=inputs,
        batch_size=2,
        input_device=torch.device("cpu"),
    )

    # then
    expected = torch.arange(5, dtype=torch.float32) * 12
    assert len(method.calls) == 3
    assert torch.equal(method.calls[-1][1], torch.zeros(3, 2, 2))
    assert torch.equal(first, expected[:, None])
    assert torch.equal(second, expected[:, None, None] * 2)


def test_run_executorch_method_passes_contiguous_input() -> None:
    # given
    method = RecordingMethod(batch_size=1)
    channels_last = torch.ones((1, 4, 4, 3)).permute(0, 3, 1, 2)

    # when
    result, _ = executorch.run_executorch_method_with_static_batch_size(
        method,
        inputs=channels_last,
        batch_size=1,
        input_device=torch.device("cpu"),
    )

    # then
    assert not channels_last.is_contiguous()
    assert torch.equal(result, torch.tensor([[48.0]]))


def test_get_executorch_input_spec_reads_program_metadata() -> None:
    # given
    input_meta = SimpleNamespace(
        sizes=lambda: (1, 3, 384, 384),
        device=lambda: "cuda:0",
        dtype_name=lambda: "float32",
    )
    program = SimpleNamespace(
        metadata=lambda name: SimpleNamespace(
            input_tensor_meta=lambda index: input_meta
        )
    )

    # when
    result = executorch.get_executorch_input_spec(program, method_name="forward")

    # then
    assert result == ((1, 3, 384, 384), torch.device("cuda:0"), "float32")


def test_load_executorch_program_raises_missing_dependency_without_executorch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    monkeypatch.setitem(sys.modules, "executorch.runtime", None)

    # when
    with pytest.raises(MissingDependencyError):
        executorch.load_executorch_program("weights.pte")


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch, executorch_version: str = "1.6.0"
) -> Mock:
    runtime = Mock()
    monkeypatch.setattr(
        executorch.importlib.metadata, "version", lambda name: executorch_version
    )
    monkeypatch.setitem(
        sys.modules,
        "executorch.runtime",
        SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime)),
    )

    return runtime


@pytest.mark.parametrize("executorch_version", ["1.4.0", "1.5.1"])
def test_load_executorch_program_rejects_executorch_older_than_1_6(
    monkeypatch: pytest.MonkeyPatch, executorch_version: str
) -> None:
    # given
    runtime = _patch_runtime(monkeypatch, executorch_version=executorch_version)

    # when
    with pytest.raises(MissingDependencyError, match=executorch_version):
        executorch.load_executorch_program("weights.pte")

    # then
    runtime.load_program.assert_not_called()


@pytest.mark.parametrize(
    "executorch_version", ["1.6.0", "1.6.0.dev20260904+cu132", "1.7.0"]
)
def test_load_executorch_program_accepts_executorch_1_6_and_newer(
    monkeypatch: pytest.MonkeyPatch, executorch_version: str
) -> None:
    # given
    monkeypatch.setattr(executorch.importlib.util, "find_spec", lambda name: None)
    runtime = _patch_runtime(monkeypatch, executorch_version=executorch_version)

    # when
    program = executorch.load_executorch_program("weights.pte")

    # then
    assert program is runtime.load_program.return_value


def test_load_executorch_program_registers_tensorrt_delegate_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    imported = []
    monkeypatch.setattr(executorch.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(executorch.importlib, "import_module", imported.append)
    runtime = _patch_runtime(monkeypatch)

    # when
    program = executorch.load_executorch_program("weights.pte", data_path="weights.ptd")

    # then
    assert imported == [executorch.TENSORRT_DELEGATE_MODULE]
    assert str(runtime.load_program.call_args.args[0]) == "weights.pte"
    assert runtime.load_program.call_args.kwargs == {"data_path": "weights.ptd"}
    assert program is runtime.load_program.return_value


def test_load_executorch_program_reports_broken_tensorrt_delegate_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    def import_module(name):
        raise ImportError("libnvinfer.so.11: cannot open shared object file")

    monkeypatch.setattr(executorch.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(executorch.importlib, "import_module", import_module)
    _patch_runtime(monkeypatch)

    # when
    with pytest.raises(ImportError, match="libnvinfer") as error:
        executorch.load_executorch_program("weights.pte")

    # then
    assert not isinstance(error.value, MissingDependencyError)


@pytest.mark.gpu_only
@pytest.mark.trt_extras
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("input_device", ["cpu", "cuda"])
def test_run_executorch_method_moves_gpu_batches_to_program_device(
    input_device: str,
) -> None:
    # given
    method = RecordingMethod(batch_size=2)
    inputs = torch.arange(3, dtype=torch.float32, device="cuda")[
        :, None, None, None
    ].expand(3, 3, 2, 2)
    stream = torch.cuda.Stream()

    # when
    first, second = executorch.run_executorch_method_with_static_batch_size(
        method,
        inputs=inputs,
        batch_size=2,
        input_device=torch.device(input_device),
        stream=stream,
    )

    # then
    expected = torch.arange(3, dtype=torch.float32) * 12
    assert [device.type for device in method.devices] == [input_device] * 2
    assert torch.equal(first.cpu(), expected[:, None])
    assert torch.equal(second.cpu(), expected[:, None, None] * 2)
