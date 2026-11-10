from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnxruntime
import pytest
import torch

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


@pytest.fixture
def attention_model_path(tmp_path: Path) -> str:
    torch.manual_seed(0)
    return export_model(SelfAttention(), tmp_path)


def test_fuses_sdpa_attention_into_multi_head_attention(
    attention_model_path: str,
) -> None:
    # when
    fused_model_path = fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    )

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
    fused_model_path = fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    )

    # then
    expected = run_on_cpu(attention_model_path, model_input)
    actual = run_on_cpu(fused_model_path, model_input)
    assert np.allclose(actual, expected, atol=1e-5)


def test_sets_scale_from_the_exported_query_and_key_scaling(
    attention_model_path: str,
) -> None:
    # when
    fused_model_path = fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    )

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
    result = fuse_attention_for_cuda_execution(model_path, providers=CUDA_PROVIDERS)

    # then
    assert result == model_path


def test_keeps_model_without_attention_unchanged(tmp_path: Path) -> None:
    # given
    model_path = export_model(Projection(), tmp_path)

    # when
    result = fuse_attention_for_cuda_execution(model_path, providers=CUDA_PROVIDERS)

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
    result = fuse_attention_for_cuda_execution(
        attention_model_path, providers=providers
    )

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
    first_result = fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    )
    fold_calls = count_fold_calls(monkeypatch)

    # when
    second_result = fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    )

    # then
    assert second_result == first_result
    assert fold_calls == []


def test_remembers_that_a_model_has_no_fusable_attention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    model_path = export_model(Projection(), tmp_path)
    fuse_attention_for_cuda_execution(model_path, providers=CUDA_PROVIDERS)
    fold_calls = count_fold_calls(monkeypatch)

    # when
    result = fuse_attention_for_cuda_execution(model_path, providers=CUDA_PROVIDERS)

    # then
    assert result == model_path
    assert fold_calls == []


def test_fuses_again_when_the_cached_fused_model_was_deleted(
    attention_model_path: str,
) -> None:
    # given
    fused_model_path = fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    )
    Path(fused_model_path).unlink()

    # when
    result = fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    )

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
    result = fuse_attention_for_cuda_execution(
        attention_model_path, providers=CUDA_PROVIDERS
    )

    # then
    assert result == attention_model_path
