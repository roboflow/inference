"""Recording codecs of the native V2 payloads: tensors, images and predictions.

Every tensor, including image pixels and instance masks, is stored as raw
little-endian C-order element bytes in the recording's blob file::

    torch.Tensor   {"dtype", "shape", "data": <blob ref>}
    ImageData      {"pixels": <tensor node>, "image_id", "parent", "root",
                    "video_metadata": <JSON>|null, "composite_sources": [..]|null,
                    optional "parent_row": {"namespace", "id"}}
    carrier        {"fields": {"<dataclass field>": <node>, ...}}
    semantic       {"fields": {...}, "device": "<source device>"}
    PredictionRows {"native": <node>, "columns": {"<name>": <tensor node>}}

Prediction carriers are the ``NATIVE_CARRIERS`` of the native wire format;
their fields are encoded recursively, so their tensors use the tensor codec
and RLE mask bytes the generic bytes codec. ``SemanticSegmentationResult``
also records its source device as provenance and is checked against its kind
when decoded; so is a decoded ``PredictionRows``. Encoding copies device tensors to
the host. Decoding always builds new CPU tensors; moving them to a device is
the reader's explicit choice. Autograd state is not recorded.

The image codec stays ``v2/image_data@1``: ``parent_row`` is written only for
an image that has one, so every other image is encoded exactly as before and
records written before the key existed decode unchanged. A reader older than
the key rejects an image that has it, by its exact key check.
"""

import dataclasses
import math
from typing import Any, Callable, Dict, Tuple

import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import (
    CompositeSource,
    FrameMapping,
    ImageData,
)
from roboflow_workflows.v2.data.payload_codec import (
    NATIVE_CARRIERS,
    SUPPORTED_TENSOR_DTYPES,
    tensor_dtype_problem,
)
from roboflow_workflows.v2.data.predictions import NATIVE_SEMANTIC_SEGMENTATION_KIND
from roboflow_workflows.v2.data.row_identity import RowRef
from roboflow_workflows.v2.data.rows import PredictionRows
from roboflow_workflows.v2.engine.recording.codecs import (
    BlobSink,
    BlobSource,
    JsonValue,
    PayloadCodec,
    PayloadDecoder,
    PayloadEncoder,
    check_array_bytes,
    read_array_header,
    type_name_of,
)
from roboflow_workflows.v2.engine.recording.errors import (
    RecordingCodecError,
    RecordingCorruptError,
)

from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

TENSOR_CODEC_NAME = "v2/torch_tensor@1"
IMAGE_DATA_CODEC_NAME = "v2/image_data@1"

_TENSOR_DTYPE_NAMES = {dtype: name for name, dtype in SUPPORTED_TENSOR_DTYPES.items()}
_IMAGE_KEYS = {
    "pixels",
    "image_id",
    "parent",
    "root",
    "video_metadata",
    "composite_sources",
}
_OPTIONAL_IMAGE_KEYS = {"parent_row"}


def encode_tensor(
    tensor: torch.Tensor, sink: BlobSink, encoder: PayloadEncoder
) -> JsonValue:
    """Store a tensor's elements in the blob file.

    Args:
        tensor: Tensor on any device, of any memory layout.
        sink: Blob destination.
        encoder: Unused; part of the codec signature.

    Returns:
        ``{"dtype", "shape", "data"}``.

    Raises:
        RecordingCodecError: On an unsupported dtype.
    """
    problem = tensor_dtype_problem(tensor.dtype)
    if problem is not None:
        raise RecordingCodecError(problem)

    host = tensor.detach().cpu().contiguous()
    flat = host.reshape(-1)
    # Singleton and empty views can be contiguous with a non-unit stride.
    if flat.stride(0) != 1:
        flat = flat.clone(memory_format=torch.contiguous_format)

    raw = memoryview(flat.view(torch.uint8).numpy())
    node = {
        "dtype": _TENSOR_DTYPE_NAMES[tensor.dtype],
        "shape": [int(size) for size in host.shape],
        "data": sink.put(raw),
    }

    return node


def decode_tensor(
    node: JsonValue, source: BlobSource, decoder: PayloadDecoder
) -> torch.Tensor:
    """Rebuild a new CPU tensor written by ``encode_tensor``.

    Args:
        node: ``{"dtype", "shape", "data"}``.
        source: Blob source.
        decoder: Unused; part of the codec signature.

    Returns:
        A contiguous CPU tensor that owns its memory.

    Raises:
        RecordingCorruptError: When the node or the data are invalid.
    """
    dtype_name, shape = read_array_header(
        node, dtype_names=tuple(SUPPORTED_TENSOR_DTYPES)
    )
    dtype = SUPPORTED_TENSOR_DTYPES[dtype_name]
    raw = source.get(node["data"])
    item_size = torch.empty((), dtype=dtype).element_size()
    check_array_bytes(raw, shape=shape, item_size=item_size, dtype_name=dtype_name)
    if not math.prod(shape):
        return torch.empty(shape, dtype=dtype)

    owned = raw if isinstance(raw, bytearray) else bytearray(raw)
    tensor = torch.frombuffer(owned, dtype=torch.uint8).view(dtype).reshape(shape)

    return tensor


def encode_image_data(
    image: ImageData, sink: BlobSink, encoder: PayloadEncoder
) -> JsonValue:
    """Encode pixels, identity, frame mappings and video metadata of an image.

    Args:
        image: Tensor image on any device.
        sink: Unused; pixels go through ``encoder`` and the tensor codec.
        encoder: Encoder of nested values.

    Returns:
        The image node described in the module docstring.
    """
    composite_sources = image.composite_sources
    node = {
        "pixels": encoder.encode(image.tensor_image, path=".tensor_image"),
        "image_id": image.image_id,
        "parent": image.parent.to_dict(),
        "root": image.root.to_dict(),
        "video_metadata": (
            None
            if image.video_metadata is None
            else image.video_metadata.model_dump(mode="json")
        ),
        "composite_sources": (
            None
            if composite_sources is None
            else [source.to_dict() for source in composite_sources]
        ),
    }
    if image.parent_row is not None:
        node["parent_row"] = image.parent_row.to_dict()

    return node


def decode_image_data(
    node: JsonValue, source: BlobSource, decoder: PayloadDecoder
) -> ImageData:
    """Rebuild an image written by ``encode_image_data``, pixels on the CPU.

    Args:
        node: Image node.
        source: Unused; pixels go through ``decoder``.
        decoder: Decoder of nested values.

    Returns:
        The image.

    Raises:
        RecordingCorruptError: When the node is malformed.
    """
    if type(node) is not dict or set(node) - _OPTIONAL_IMAGE_KEYS != _IMAGE_KEYS:
        raise RecordingCorruptError(
            f"an image needs exactly the keys {sorted(_IMAGE_KEYS)}, plus "
            f"optionally {sorted(_OPTIONAL_IMAGE_KEYS)}"
        )

    pixels = decoder.decode(node["pixels"], path=".tensor_image")
    if type(pixels) is not torch.Tensor:
        raise RecordingCorruptError("image pixels must decode to a tensor")

    try:
        image = ImageData(
            tensor_image=pixels,
            image_id=node["image_id"],
            parent=FrameMapping.from_dict(node["parent"]),
            root=FrameMapping.from_dict(node["root"]),
            video_metadata=_decode_video_metadata(node["video_metadata"]),
            composite_sources=(
                None
                if node["composite_sources"] is None
                else tuple(
                    CompositeSource.from_dict(item)
                    for item in node["composite_sources"]
                )
            ),
            parent_row=(
                RowRef.from_dict(node["parent_row"]) if "parent_row" in node else None
            ),
        )
    except (TypeError, ValueError) as error:
        raise RecordingCorruptError(f"invalid image: {error}") from error

    return image


def carrier_codec(carrier_class: type) -> PayloadCodec:
    """Build the recording codec of one native prediction carrier dataclass.

    Args:
        carrier_class: One of ``NATIVE_CARRIERS``.

    Returns:
        A codec that encodes every dataclass field recursively.
    """
    field_names = tuple(field.name for field in dataclasses.fields(carrier_class))

    def encode(carrier: Any, sink: BlobSink, encoder: PayloadEncoder) -> JsonValue:
        fields = {
            name: encoder.encode(getattr(carrier, name), path=f".{name}")
            for name in field_names
        }
        node = {"fields": fields}

        return node

    def decode(node: JsonValue, source: BlobSource, decoder: PayloadDecoder) -> Any:
        if (
            type(node) is not dict
            or set(node) != {"fields"}
            or type(node["fields"]) is not dict
            or set(node["fields"]) != set(field_names)
        ):
            raise RecordingCorruptError(
                f"a {carrier_class.__name__} needs exactly the fields {list(field_names)}"
            )

        fields = {
            name: decoder.decode(node["fields"][name], path=f".{name}")
            for name in field_names
        }
        carrier = carrier_class(**fields)

        return carrier

    codec = PayloadCodec(
        name=f"v2/{carrier_class.__name__}@1",
        type_name=type_name_of(carrier_class),
        encode=encode,
        decode=decode,
    )

    return codec


def encode_native_semantic(
    result: SemanticSegmentationResult, sink: BlobSink, encoder: PayloadEncoder
) -> JsonValue:
    """Encode labels, confidence and metadata, plus the source device.

    Args:
        result: Native semantic segmentation result on any device.
        sink: Unused; tensors go through ``encoder`` and the tensor codec.
        encoder: Encoder of nested values.

    Returns:
        ``{"fields": {...}, "device": "<device>"}``. The device is recorded
        provenance only: decoding never reads it and places the tensors on
        the CPU, as for every tensor.
    """
    node = {
        "fields": _SEMANTIC_CARRIER_CODEC.encode(result, sink, encoder)["fields"],
        "device": str(result.segmentation_map.device),
    }

    return node


def decode_native_semantic(
    node: JsonValue, source: BlobSource, decoder: PayloadDecoder
) -> SemanticSegmentationResult:
    """Rebuild a result written by ``encode_native_semantic`` on the CPU.

    Args:
        node: Semantic result node.
        source: Blob source of the nested tensors.
        decoder: Decoder of nested values.

    The recorded ``device`` must be present but is not used. The decoded
    result is re-checked against ``NATIVE_SEMANTIC_SEGMENTATION_KIND``, which
    reads tensor shapes and dtypes only.

    Returns:
        The result on the CPU, with both tensors in their recorded dtype.

    Raises:
        RecordingCorruptError: When the node is malformed or the decoded
            labels and confidence break the kind contract.
    """
    if (
        type(node) is not dict
        or set(node) != {"fields", "device"}
        or type(node["device"]) is not str
    ):
        raise RecordingCorruptError(
            "a SemanticSegmentationResult needs exactly the keys 'fields' and "
            "'device' (a string)"
        )

    result = _SEMANTIC_CARRIER_CODEC.decode({"fields": node["fields"]}, source, decoder)
    try:
        NATIVE_SEMANTIC_SEGMENTATION_KIND.check(result)
    except ContractError as error:
        raise RecordingCorruptError(f"invalid semantic result: {error}") from error

    return result


def encode_prediction_rows(
    rows: PredictionRows, sink: BlobSink, encoder: PayloadEncoder
) -> JsonValue:
    """Encode the native prediction and every column of a row carrier.

    Args:
        rows: Row carrier on any device.
        sink: Unused; tensors go through ``encoder`` and the tensor codec.
        encoder: Encoder of nested values.

    Returns:
        ``{"native": <node>, "columns": <node>}``.
    """
    node = {
        "native": encoder.encode(rows.native, path=".native"),
        "columns": encoder.encode(dict(rows.columns), path=".columns"),
    }

    return node


def decode_prediction_rows(
    node: JsonValue, source: BlobSource, decoder: PayloadDecoder
) -> PredictionRows:
    """Rebuild a row carrier written by ``encode_prediction_rows`` on the CPU.

    Args:
        node: Row carrier node.
        source: Unused; tensors go through ``decoder``.
        decoder: Decoder of nested values.

    Returns:
        The carrier, validated again.

    Raises:
        RecordingCorruptError: When the node is malformed or the decoded
            columns do not align with the native rows.
    """
    if type(node) is not dict or set(node) != {"native", "columns"}:
        raise RecordingCorruptError(
            "a PredictionRows needs exactly the keys 'native' and 'columns'"
        )

    native = decoder.decode(node["native"], path=".native")
    columns = decoder.decode(node["columns"], path=".columns")
    try:
        rows = PredictionRows(native=native, columns=columns)
    except ContractError as error:
        raise RecordingCorruptError(f"invalid prediction rows: {error}") from error

    return rows


def _decode_video_metadata(value: JsonValue) -> Any:
    if value is None:
        return None

    from roboflow_workflows.execution_engine.entities.base import VideoMetadata

    metadata = VideoMetadata.model_validate(value)

    return metadata


def _codec(
    name: str, value_type: type, encode: Callable, decode: Callable
) -> PayloadCodec:
    codec = PayloadCodec(
        name=name, type_name=type_name_of(value_type), encode=encode, decode=decode
    )

    return codec


TENSOR_CODEC = _codec(TENSOR_CODEC_NAME, torch.Tensor, encode_tensor, decode_tensor)
IMAGE_DATA_CODEC = _codec(
    IMAGE_DATA_CODEC_NAME, ImageData, encode_image_data, decode_image_data
)
_SEMANTIC_CARRIER_CODEC = carrier_codec(SemanticSegmentationResult)
# The semantic codec is the one asymmetric carrier codec: it was added with the
# V2 native semantic kind, records the source device as provenance and
# re-checks the decoded result against that kind. The older carrier codecs keep
# their existing recording format: no device field and no kind check on
# decode. Every codec decodes tensors onto the CPU.
CARRIER_CODECS: Dict[str, PayloadCodec] = {
    name: carrier_codec(carrier)
    for name, carrier in NATIVE_CARRIERS.items()
    if carrier is not SemanticSegmentationResult
}
CARRIER_CODECS[SemanticSegmentationResult.__name__] = _codec(
    _SEMANTIC_CARRIER_CODEC.name,
    SemanticSegmentationResult,
    encode_native_semantic,
    decode_native_semantic,
)
PREDICTION_ROWS_CODEC = _codec(
    "v2/PredictionRows@1",
    PredictionRows,
    encode_prediction_rows,
    decode_prediction_rows,
)
NATIVE_RECORDING_CODECS: Tuple[PayloadCodec, ...] = (
    TENSOR_CODEC,
    IMAGE_DATA_CODEC,
    *CARRIER_CODECS.values(),
    PREDICTION_ROWS_CODEC,
)
