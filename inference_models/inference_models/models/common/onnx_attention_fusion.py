"""Fuse exported scaled-dot-product attention into ONNX Runtime's ``MultiHeadAttention`` for CUDA.

The TorchScript ONNX exporter writes ``scaled_dot_product_attention`` as ``MatMul -> Softmax -> MatMul``,
which materializes the full attention score tensor. On the CUDA execution provider,
``com.microsoft.MultiHeadAttention`` runs a memory-efficient kernel that does not. ONNX Runtime's own
fusions do not match this pattern, so this module adds the fusion before the session is created.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from filelock import FileLock

from inference_models.logger import LOGGER
from inference_models.models.common.model_packages import get_file_identity

CACHE_DIR_NAME = "attention_fusion"
TRANSFORM_VERSION = 1
FUSED_MODEL_FILE_NAME = "weights.onnx"
RESULT_FILE_NAME = "result.json"
CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"
MICROSOFT_DOMAIN = "com.microsoft"


@dataclass(frozen=True)
class _AttentionBlock:
    query_scale: object
    key_scale: object
    scores: object
    softmax: object
    weighted_values: object
    query: str
    transposed_key: str
    value: str
    scale: float
    num_heads: int


def fuse_attention_for_cuda_execution(
    model_path: str,
    *,
    providers: List[Union[str, tuple]],
) -> str:
    """Return a model path whose attention is fused into ONNX Runtime's ``MultiHeadAttention`` op.

    The fused model is written to an ``attention_fusion`` directory next to the source model and reused
    on later loads. The source model is returned unchanged when CUDA is not the first execution provider,
    when the model has no unmasked scaled-dot-product attention with static shapes, or when the fusion
    fails.

    Args:
        model_path: Path to the ONNX model the session would load.
        providers: Execution providers the session will use, in priority order.

    Returns:
        Path to the fused model, or ``model_path`` when no fusion applies.
    """
    if not _runs_on_cuda_first(providers=providers):
        return model_path

    cache_dir = (
        Path(model_path).parent
        / CACHE_DIR_NAME
        / f"{get_file_identity(model_path)}-v{TRANSFORM_VERSION}"
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cache_dir.with_suffix(".lock"))):
            fused_model_path = _load_or_fuse(model_path, cache_dir=cache_dir)
    except Exception as error:
        LOGGER.warning(
            "Could not fuse attention for %s (%s). Loading the model without attention fusion.",
            model_path,
            error,
        )
        return model_path

    return fused_model_path


def _runs_on_cuda_first(providers: List[Union[str, tuple]]) -> bool:
    if not providers:
        return False

    first_provider = providers[0]
    first_provider_name = (
        first_provider[0] if isinstance(first_provider, tuple) else first_provider
    )
    return first_provider_name == CUDA_EXECUTION_PROVIDER


def _load_or_fuse(model_path: str, *, cache_dir: Path) -> str:
    result_path = cache_dir / RESULT_FILE_NAME
    fused_model_path = cache_dir / FUSED_MODEL_FILE_NAME
    if result_path.exists():
        fused_blocks = json.loads(result_path.read_text())["fused_attention_blocks"]
        if not fused_blocks:
            return model_path

        # The inference cache watchdog purges files one by one, so the fused model can be gone.
        if fused_model_path.exists():
            return str(fused_model_path)

    fused_blocks = _fuse_into(model_path, destination=fused_model_path)
    staged_result_path = result_path.with_suffix(".tmp")
    staged_result_path.write_text(json.dumps({"fused_attention_blocks": fused_blocks}))
    os.replace(staged_result_path, result_path)
    LOGGER.info(
        "Fused %d attention blocks of %s into MultiHeadAttention.",
        fused_blocks,
        model_path,
    )
    return str(fused_model_path) if fused_blocks else model_path


def _fuse_into(model_path: str, *, destination: Path) -> int:
    """Fold constants, rewrite every matching attention block, and save the result atomically."""
    import onnx

    with tempfile.TemporaryDirectory(dir=destination.parent) as work_dir:
        folded_path = Path(work_dir) / "folded.onnx"
        _fold_constants(model_path, destination=folded_path)
        model = onnx.load(str(folded_path))
        blocks = _find_attention_blocks(model)
        if not blocks:
            return 0

        _replace_attention_blocks(model, blocks=blocks)
        onnx.checker.check_model(model)
        staged_path = Path(work_dir) / FUSED_MODEL_FILE_NAME
        onnx.save(model, str(staged_path))
        os.replace(staged_path, destination)

    return len(blocks)


def _fold_constants(model_path: str, *, destination: Path) -> None:
    """Run ONNX Runtime's basic optimizations, which fold the exporter's shape-derived scale."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    options.optimized_model_filepath = str(destination)
    onnxruntime.InferenceSession(
        model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )


def _find_attention_blocks(model) -> List[_AttentionBlock]:
    """Find ``Mul(Q, s) @ Mul(Kᵀ, s) -> Softmax(-1) -> @ V`` blocks that have no mask."""
    graph = model.graph
    constants = _constant_values(model)
    shapes = _static_shapes(model)
    producers = {output: node for node in graph.node for output in node.output}
    consumers: Dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    graph_outputs = {output.name for output in graph.output}

    blocks = []
    for softmax in graph.node:
        block = _match_block(
            softmax,
            producers=producers,
            consumers=consumers,
            constants=constants,
            shapes=shapes,
            graph_outputs=graph_outputs,
        )
        if block is not None:
            blocks.append(block)

    return blocks


def _match_block(
    softmax,
    *,
    producers: dict,
    consumers: dict,
    constants: Dict[str, np.ndarray],
    shapes: Dict[str, List[int]],
    graph_outputs: set,
) -> Optional[_AttentionBlock]:
    if softmax.op_type != "Softmax" or _softmax_axis(softmax) != -1:
        return None

    scores = producers.get(softmax.input[0])
    users = consumers.get(softmax.output[0], [])
    if scores is None or scores.op_type != "MatMul" or len(users) != 1:
        return None

    weighted_values = users[0]
    if (
        weighted_values.op_type != "MatMul"
        or weighted_values.input[0] != softmax.output[0]
        or softmax.output[0] in graph_outputs
        or len(consumers.get(scores.output[0], [])) != 1
    ):
        return None

    query_scale = producers.get(scores.input[0])
    key_scale = producers.get(scores.input[1])
    scaled_query = _scaled_operand(query_scale, constants=constants)
    scaled_key = _scaled_operand(key_scale, constants=constants)
    if scaled_query is None or scaled_key is None:
        return None

    if any(
        len(consumers.get(node.output[0], [])) != 1 for node in (query_scale, key_scale)
    ):
        return None

    query, query_factor = scaled_query
    transposed_key, key_factor = scaled_key
    value = weighted_values.input[1]
    query_shape = shapes.get(query)
    key_shape = shapes.get(transposed_key)
    value_shape = shapes.get(value) or shapes.get(weighted_values.output[0])
    if not _shapes_match(query_shape, key_shape=key_shape, value_shape=value_shape):
        return None

    block = _AttentionBlock(
        query_scale=query_scale,
        key_scale=key_scale,
        scores=scores,
        softmax=softmax,
        weighted_values=weighted_values,
        query=query,
        transposed_key=transposed_key,
        value=value,
        scale=query_factor * key_factor,
        num_heads=query_shape[1],
    )
    return block


def _softmax_axis(softmax) -> int:
    for attribute in softmax.attribute:
        if attribute.name == "axis":
            return attribute.i

    return -1


def _scaled_operand(
    node, *, constants: Dict[str, np.ndarray]
) -> Optional[Tuple[str, float]]:
    if node is None or node.op_type != "Mul":
        return None

    for tensor_index, factor_index in ((0, 1), (1, 0)):
        factor = constants.get(node.input[factor_index])
        tensor = node.input[tensor_index]
        if factor is not None and factor.size == 1 and tensor not in constants:
            return tensor, float(factor.reshape(()))

    return None


def _shapes_match(
    query_shape: Optional[List[int]],
    *,
    key_shape: Optional[List[int]],
    value_shape: Optional[List[int]],
) -> bool:
    if not query_shape or not key_shape or not value_shape:
        return False

    if any(len(shape) != 4 for shape in (query_shape, key_shape, value_shape)):
        return False

    batch, heads, _, head_size = query_shape
    return (
        key_shape[:2] == [batch, heads]
        and key_shape[2] == head_size
        and value_shape[:2] == [batch, heads]
    )


def _constant_values(model) -> Dict[str, np.ndarray]:
    from onnx import numpy_helper

    values = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if (
            node.op_type == "Constant"
            and node.attribute
            and node.attribute[0].name == "value"
        ):
            values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)

    return values


def _static_shapes(model) -> Dict[str, List[int]]:
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    shapes = {}
    for value_info in (
        list(inferred.graph.value_info)
        + list(inferred.graph.input)
        + list(inferred.graph.output)
    ):
        dims = value_info.type.tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") for dim in dims):
            shapes[value_info.name] = [dim.dim_value for dim in dims]

    return shapes


def _replace_attention_blocks(model, *, blocks: List[_AttentionBlock]) -> None:
    """Replace each block with ``MultiHeadAttention`` on (batch, sequence, hidden) inputs."""
    from onnx import helper, numpy_helper

    graph = model.graph
    merge_heads_shape = numpy_helper.from_array(
        np.array([0, 0, -1], dtype=np.int64), "attention_fusion/merge_heads_shape"
    )
    graph.initializer.append(merge_heads_shape)
    replacements = {}
    removed = set()
    for index, block in enumerate(blocks):
        prefix = f"attention_fusion/{index}"
        split_heads_shape = numpy_helper.from_array(
            np.array([0, 0, block.num_heads, -1], dtype=np.int64),
            f"{prefix}/split_heads_shape",
        )
        graph.initializer.append(split_heads_shape)
        replacements[id(block.weighted_values)] = [
            helper.make_node(
                "Transpose", [block.query], [f"{prefix}/query_bsnh"], perm=[0, 2, 1, 3]
            ),
            helper.make_node(
                "Reshape",
                [f"{prefix}/query_bsnh", merge_heads_shape.name],
                [f"{prefix}/query"],
            ),
            helper.make_node(
                "Transpose",
                [block.transposed_key],
                [f"{prefix}/key_bsnh"],
                perm=[0, 3, 1, 2],
            ),
            helper.make_node(
                "Reshape",
                [f"{prefix}/key_bsnh", merge_heads_shape.name],
                [f"{prefix}/key"],
            ),
            helper.make_node(
                "Transpose", [block.value], [f"{prefix}/value_bsnh"], perm=[0, 2, 1, 3]
            ),
            helper.make_node(
                "Reshape",
                [f"{prefix}/value_bsnh", merge_heads_shape.name],
                [f"{prefix}/value"],
            ),
            helper.make_node(
                "MultiHeadAttention",
                [f"{prefix}/query", f"{prefix}/key", f"{prefix}/value"],
                [f"{prefix}/attended"],
                name=f"{prefix}/MultiHeadAttention",
                domain=MICROSOFT_DOMAIN,
                num_heads=block.num_heads,
                scale=block.scale,
            ),
            helper.make_node(
                "Reshape",
                [f"{prefix}/attended", split_heads_shape.name],
                [f"{prefix}/attended_bsnh"],
            ),
            helper.make_node(
                "Transpose",
                [f"{prefix}/attended_bsnh"],
                [block.weighted_values.output[0]],
                perm=[0, 2, 1, 3],
            ),
        ]
        for node in (
            block.query_scale,
            block.key_scale,
            block.scores,
            block.softmax,
            block.weighted_values,
        ):
            removed.add(id(node))

    nodes = []
    for node in graph.node:
        if id(node) in replacements:
            nodes.extend(replacements[id(node)])
        elif id(node) not in removed:
            nodes.append(node)

    del graph.node[:]
    graph.node.extend(nodes)
    _remove_unused_initializers(graph)
    if not any(opset.domain == MICROSOFT_DOMAIN for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid(MICROSOFT_DOMAIN, 1))


def _remove_unused_initializers(graph) -> None:
    """Drop initializers no node reads, such as the replaced scales; ONNX Runtime warns about each one."""
    used = {name for node in graph.node for name in node.input}
    used.update(output.name for output in graph.output)
    initializers = [
        initializer for initializer in graph.initializer if initializer.name in used
    ]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
