"""Recover features or raw logits from a supported classifier's final linear layer.

Keep the complete backbone rather than using ONNX's generic subgraph extractor:
control-flow subgraphs can capture tensors from the enclosing graph.
"""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Tuple

from filelock import FileLock

FEATURE_DEFINITION = "classifier-linear-input@v1"
LOGITS_DEFINITION = "classifier-linear-output@v1"
TRANSFORM_VERSION = 1
_HEAD = re.compile(r"(?:^|[./])(fc|classifier|linear_layer)(?:[./]|$)")


def embedding_definition(output_type="feature_vector"):
    if output_type == "feature_vector":
        return FEATURE_DEFINITION
    if output_type == "logits":
        return LOGITS_DEFINITION
    raise ValueError(f"Unknown embedding output type: {output_type}")


def extract_classifier_embedding(
    model, output_type="feature_vector"
) -> Tuple[object, dict]:
    import onnx

    definition = embedding_definition(output_type)
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError("Embedding extraction requires one classification output.")
    producers = {output: node for node in graph.node for output in node.output}
    initializers = {value.name: value for value in graph.initializer}
    removed = []
    tensor = graph.output[0].name
    while tensor in producers and producers[tensor].op_type in {
        "Identity",
        "Softmax",
        "Sigmoid",
        "Cast",
    }:
        node = producers[tensor]
        removed.append(node)
        tensor = node.input[0]
    logits_tensor = tensor
    node = producers.get(tensor)
    if node is not None and node.op_type == "Add":
        bias_inputs = [name for name in node.input if name in initializers]
        if len(bias_inputs) != 1 or not _HEAD.search(bias_inputs[0]):
            raise ValueError("Cannot identify the final classifier bias.")
        if output_type == "feature_vector":
            removed.append(node)
        tensor = next(name for name in node.input if name != bias_inputs[0])
        node = producers.get(tensor)
    if node is None or node.op_type not in {"Gemm", "MatMul"}:
        raise ValueError("Cannot recover a supported final linear classifier.")
    weight = initializers.get(node.input[1])
    if weight is None or not _HEAD.search(weight.name) or len(weight.dims) != 2:
        raise ValueError("Cannot identify the final classifier weight tensor.")
    attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if node.op_type == "Gemm" and attributes.get("transA", 0):
        raise ValueError("Transposed classifier inputs are unsupported.")
    input_dimension = weight.dims[1 if attributes.get("transB", 0) else 0]
    dimension = (
        input_dimension
        if output_type == "feature_vector"
        else weight.dims[0 if attributes.get("transB", 0) else 1]
    )
    feature_tensor = node.input[0] if output_type == "feature_vector" else logits_tensor
    if output_type == "feature_vector":
        removed.append(node)
    removed_outputs = {output for item in removed for output in item.output}

    # Include nested graph inputs when checking whether a removed tensor is shared.
    def inputs(nodes):
        for item in nodes:
            yield from item.input
            for attribute in item.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from inputs(attribute.g.node)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        yield from inputs(subgraph.node)

    retained = [item for item in graph.node if item not in removed]
    if removed_outputs.intersection(inputs(retained)):
        raise ValueError("The classifier output is shared with the backbone.")
    inferred = onnx.shape_inference.infer_shapes(model)
    value_infos = {
        value.name: value
        for value in list(inferred.graph.value_info)
        + list(inferred.graph.input)
        + list(inferred.graph.output)
    }
    feature = value_infos.get(feature_tensor)
    if feature is None or len(feature.type.tensor_type.shape.dim) != 2:
        raise ValueError("Cannot establish a batch-by-features embedding shape.")
    if feature.type.tensor_type.shape.dim[1].dim_value != dimension:
        raise ValueError("Classifier embedding shape does not match its weights.")
    # Work on a copy; never alter the source classifier or its outputs.
    result = onnx.ModelProto()
    result.CopyFrom(model)
    del result.graph.node[:]
    result.graph.node.extend(retained)
    del result.graph.output[:]
    result.graph.output.append(feature)
    onnx.checker.check_model(result)
    return result, {
        "feature_definition": definition,
        "output_type": output_type,
        "feature_tensor": feature_tensor,
        "dimension": dimension,
        "normalization": "none",
        "transform_version": TRANSFORM_VERSION,
    }


def prepare_classifier_embedding(
    source_path: str, output_type="feature_vector"
) -> Tuple[str, dict]:
    import onnx

    definition = embedding_definition(output_type)
    # Loading external data before hashing includes it in the artifact identity.
    source = onnx.load(source_path)
    source_digest = hashlib.sha256(source.SerializeToString()).hexdigest()
    key = hashlib.sha256(
        f"{source_digest}:{definition}:{TRANSFORM_VERSION}".encode()
    ).hexdigest()
    directory = Path(source_path).parent / "embeddings" / key
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "weights.onnx"
    metadata_path = directory / "embedding.json"
    with FileLock(str(directory / "extract.lock")):
        if output_path.exists() and metadata_path.exists():
            return str(output_path), json.loads(metadata_path.read_text())
        model, metadata = extract_classifier_embedding(source, output_type=output_type)
        metadata["source_artifact_sha256"] = source_digest
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary_path = temporary.name
        try:
            onnx.save(model, temporary_path)
            os.replace(temporary_path, output_path)
            metadata_path.write_text(json.dumps(metadata, sort_keys=True))
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
    return str(output_path), metadata
