from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from filelock import FileLock, Timeout

onnx = pytest.importorskip(
    "onnx",
    reason="onnx is not installed (requires the onnx-* extra)",
)
onnxruntime = pytest.importorskip(
    "onnxruntime",
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)

from inference_models.models.common import onnx_attention_fusion
from inference_models.models.common.onnx_attention_fusion import (
    fuse_attention_for_cuda_execution,
)

CUDA_PROVIDERS = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
EMBEDDING_DIM = 64
NUM_HEADS = 4
SEQUENCE_LENGTH = 32


class SelfAttention(torch.nn.Module):
    def __init__(self, attention_mask: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(EMBEDDING_DIM, 3 * EMBEDDING_DIM)
        self.projection = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.attention_mask = attention_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        query, key, value = (
            part.reshape(batch, sequence, NUM_HEADS, -1).transpose(1, 2)
            for part in self.qkv(x).chunk(3, dim=-1)
        )
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=self.attention_mask
        )
        merged = attended.transpose(1, 2).reshape(batch, sequence, EMBEDDING_DIM)
        return self.projection(merged)


class Projection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=-1)


def export_model(module: torch.nn.Module, directory: Path) -> str:
    torch.manual_seed(0)
    module.eval()
    model_path = directory / "weights.onnx"
    torch.onnx.export(
        module,
        (torch.rand(1, SEQUENCE_LENGTH, EMBEDDING_DIM),),
        str(model_path),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        dynamo=False,
    )
    return str(model_path)


def count_operators(model_path: str) -> Counter:
    graph = onnx.load(model_path).graph
    return Counter(node.op_type for node in graph.node)


def run_on_cpu(model_path: str, model_input: np.ndarray) -> np.ndarray:
    session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    (output,) = session.run(None, {"x": model_input})
    return output


def fuse(model_path: str, *, providers: list) -> str:
    with fuse_attention_for_cuda_execution(model_path, providers=providers) as path:
        return path


def watchdog_can_lock(file_path: str) -> bool:
    path = Path(file_path)
    try:
        with FileLock(str(path.parent / f".{path.name}.lock"), timeout=0):
            return True
    except Timeout:
        return False


@pytest.fixture
def attention_model_path(tmp_path: Path) -> str:
    torch.manual_seed(0)
    return export_model(SelfAttention(), tmp_path)


def test_fuses_sdpa_attention_into_multi_head_attention(
    attention_model_path: str,
) -> None:
    # when
    fused_model_path = fuse(attention_model_path, providers=CUDA_PROVIDERS)

    # then
    operators = count_operators(fused_model_path)
    assert fused_model_path != attention_model_path
    assert operators["MultiHeadAttention"] == 1
    assert operators["Softmax"] == 0


def test_fused_model_keeps_original_outputs(attention_model_path: str) -> None:
    # given
    model_input = np.random.default_rng(0).random(
        (1, SEQUENCE_LENGTH, EMBEDDING_DIM), dtype=np.float32
    )

    # when
    fused_model_path = fuse(attention_model_path, providers=CUDA_PROVIDERS)

    # then
    expected = run_on_cpu(attention_model_path, model_input)
    actual = run_on_cpu(fused_model_path, model_input)
    assert np.allclose(actual, expected, atol=1e-5)


def test_sets_scale_from_the_exported_query_and_key_scaling(
    attention_model_path: str,
) -> None:
    # when
    fused_model_path = fuse(attention_model_path, providers=CUDA_PROVIDERS)

    # then
    (attention_node,) = [
        node
        for node in onnx.load(fused_model_path).graph.node
        if node.op_type == "MultiHeadAttention"
    ]
    attributes = {
        a.name: onnx.helper.get_attribute_value(a) for a in attention_node.attribute
    }
    head_size = EMBEDDING_DIM // NUM_HEADS
    assert attributes["num_heads"] == NUM_HEADS
    assert attributes["scale"] == pytest.approx(head_size**-0.5)


def test_keeps_masked_attention_unchanged(tmp_path: Path) -> None:
    # given
    torch.manual_seed(0)
    attention_mask = torch.zeros(1, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
    attention_mask[..., 0] = float("-inf")
    model_path = export_model(SelfAttention(attention_mask=attention_mask), tmp_path)

    # when
    result = fuse(model_path, providers=CUDA_PROVIDERS)

    # then
    assert result == model_path


def test_keeps_model_without_attention_unchanged(tmp_path: Path) -> None:
    # given
    model_path = export_model(Projection(), tmp_path)

    # when
    result = fuse(model_path, providers=CUDA_PROVIDERS)

    # then
    assert result == model_path


@pytest.mark.parametrize(
    "providers",
    [
        ["CPUExecutionProvider"],
        ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        [("TensorrtExecutionProvider", {}), "CUDAExecutionProvider"],
    ],
)
def test_keeps_model_unchanged_when_cuda_is_not_the_first_provider(
    attention_model_path: str, providers: list
) -> None:
    # when
    result = fuse(attention_model_path, providers=providers)

    # then
    assert result == attention_model_path
    assert not (
        Path(attention_model_path).parent / onnx_attention_fusion.CACHE_DIR_NAME
    ).exists()


def count_fold_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []
    fold_constants = onnx_attention_fusion._fold_constants

    def counting_fold(*args, **kwargs):
        calls.append(args)
        fold_constants(*args, **kwargs)

    monkeypatch.setattr(onnx_attention_fusion, "_fold_constants", counting_fold)
    return calls


def test_reuses_the_cached_fused_model(
    attention_model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    first_result = fuse(attention_model_path, providers=CUDA_PROVIDERS)
    fold_calls = count_fold_calls(monkeypatch)

    # when
    second_result = fuse(attention_model_path, providers=CUDA_PROVIDERS)

    # then
    assert second_result == first_result
    assert fold_calls == []


def test_remembers_that_a_model_has_no_fusable_attention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    model_path = export_model(Projection(), tmp_path)
    fuse(model_path, providers=CUDA_PROVIDERS)
    fold_calls = count_fold_calls(monkeypatch)

    # when
    result = fuse(model_path, providers=CUDA_PROVIDERS)

    # then
    assert result == model_path
    assert fold_calls == []


def test_fuses_again_when_the_cached_fused_model_was_deleted(
    attention_model_path: str,
) -> None:
    # given
    fused_model_path = fuse(attention_model_path, providers=CUDA_PROVIDERS)
    Path(fused_model_path).unlink()

    # when
    result = fuse(attention_model_path, providers=CUDA_PROVIDERS)

    # then
    assert result == fused_model_path
    assert count_operators(result)["MultiHeadAttention"] == 1


def test_keeps_model_unchanged_when_fusion_fails(
    attention_model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    def broken_fold(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(onnx_attention_fusion, "_fold_constants", broken_fold)

    # when
    result = fuse(attention_model_path, providers=CUDA_PROVIDERS)

    # then
    assert result == attention_model_path


@pytest.mark.parametrize("warm_cache", [False, True])
def test_holds_the_watchdog_lock_of_the_fused_model_until_the_session_is_created(
    attention_model_path: str, warm_cache: bool
) -> None:
    # given
    if warm_cache:
        fuse(attention_model_path, providers=CUDA_PROVIDERS)

    # when
    with fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    ) as fused_model_path:
        locked_while_loading = not watchdog_can_lock(fused_model_path)
        session = onnxruntime.InferenceSession(
            fused_model_path, providers=["CPUExecutionProvider"]
        )

    # then
    assert locked_while_loading
    assert watchdog_can_lock(fused_model_path)
    Path(fused_model_path).unlink()
    model_input = np.zeros((1, SEQUENCE_LENGTH, EMBEDDING_DIM), dtype=np.float32)
    assert session.run(None, {"x": model_input})[0].shape == model_input.shape


COREML_PROVIDERS = [
    (
        "CoreMLExecutionProvider",
        {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"},
    ),
    "CPUExecutionProvider",
]


@pytest.fixture
def coreml_rewrite(monkeypatch):
    monkeypatch.setattr(
        onnx_attention_fusion, "INFERENCE_MODELS_COREML_KEY_SCALE_WORKAROUND", True
    )
    return onnx_attention_fusion.reorder_key_scale_for_coreml_execution


@pytest.mark.parametrize("warm_cache", [False, True])
def test_coreml_rewrite_keeps_lock_through_load(
    attention_model_path, coreml_rewrite, monkeypatch, warm_cache
):
    if warm_cache:
        with coreml_rewrite(attention_model_path, providers=COREML_PROVIDERS) as cached:
            assert cached != attention_model_path
    folds = count_fold_calls(monkeypatch)

    def load(*, model_path, providers):
        assert model_path != attention_model_path
        assert not watchdog_can_lock(model_path)
        if warm_cache:
            assert model_path == cached
        return model_path, onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    rewritten, session = onnx_attention_fusion.create_attention_session(
        attention_model_path, providers=COREML_PROVIDERS, session_factory=load
    )
    assert len(folds) == (0 if warm_cache else 1)
    assert watchdog_can_lock(rewritten)
    producers = {o: n for n in onnx.load(rewritten).graph.node for o in n.output}
    softmax = next(n for n in producers.values() if n.op_type == "Softmax")
    key = producers[producers[softmax.input[0]].input[1]]
    assert key.op_type == "Transpose"
    assert producers[key.input[0]].op_type == "Mul"
    Path(rewritten).unlink()
    x = np.random.default_rng(0).random(
        (1, SEQUENCE_LENGTH, EMBEDDING_DIM), dtype=np.float32
    )
    np.testing.assert_allclose(
        session.run(None, {"x": x})[0],
        run_on_cpu(attention_model_path, x),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "enabled,first,units",
    [
        (False, True, "CPUAndGPU"),
        (True, False, "CPUAndGPU"),
        (True, True, "CPUOnly"),
    ],
)
def test_coreml_gates_before_folding(
    attention_model_path, monkeypatch, enabled, first, units
):
    folds = count_fold_calls(monkeypatch)
    monkeypatch.setattr(
        onnx_attention_fusion, "INFERENCE_MODELS_COREML_KEY_SCALE_WORKAROUND", enabled
    )
    coreml = (
        "CoreMLExecutionProvider",
        {"ModelFormat": "MLProgram", "MLComputeUnits": units},
    )
    providers = [coreml, "CPUExecutionProvider"]
    if not first:
        providers.reverse()
    with onnx_attention_fusion.reorder_key_scale_for_coreml_execution(
        attention_model_path, providers=providers
    ) as result:
        assert result == attention_model_path
    assert not folds


@pytest.mark.parametrize("failure", ["rewrite", "load"])
def test_coreml_falls_back_to_source(
    attention_model_path, coreml_rewrite, monkeypatch, failure
):
    rewritten = []

    def fail(*args, **kwargs):
        raise RuntimeError("simulated failure")

    def load(*, model_path, providers):
        if model_path != attention_model_path:
            rewritten.append(model_path)
            assert not watchdog_can_lock(model_path)
            fail()
        assert all(watchdog_can_lock(path) for path in rewritten)
        return model_path

    if failure == "rewrite":
        monkeypatch.setattr(onnx_attention_fusion, "_fold_constants", fail)
    result = onnx_attention_fusion.create_attention_session(
        attention_model_path, providers=COREML_PROVIDERS, session_factory=load
    )
    assert result == attention_model_path
    assert len(rewritten) == int(failure == "load")
