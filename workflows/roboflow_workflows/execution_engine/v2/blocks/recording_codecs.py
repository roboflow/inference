"""Recording codecs of the native V2 payloads: tensors, images and predictions.

Every tensor, including image pixels and instance masks, is stored as raw
little-endian C-order element bytes in the recording's blob file::

    torch.Tensor   {"dtype", "shape", "data": <blob ref>}
    ImageData      {"pixels": <tensor node>, "image_id", "parent", "root",
                    "video_metadata": <JSON>|null, "composite_sources": [..]|null}
    carrier        {"fields": {"<dataclass field>": <node>, ...}}

Prediction carriers are the ``NATIVE_CARRIERS`` of the native wire format;
their fields are encoded recursively, so their tensors use the tensor codec
and RLE mask bytes the generic bytes codec. Encoding copies device tensors to
the host. Decoding always builds new CPU tensors; moving them to a device is
the reader's explicit choice. Autograd state is not recorded.
"""

import dataclasses
import math
from typing import Any, Callable, Dict, Tuple

import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import (
    CompositeSource,
    FrameMapping,
    ImageData,
)
from roboflow_workflows.execution_engine.v2.blocks.payload_codec import (
    NATIVE_CARRIERS,
    SUPPORTED_TENSOR_DTYPES,
    tensor_dtype_problem,
)
from roboflow_workflows.execution_engine.v2.recording.codecs import (
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
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingCodecError,
    RecordingCorruptError,
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
    if type(node) is not dict or set(node) != _IMAGE_KEYS:
        raise RecordingCorruptError(
            f"an image needs exactly the keys {sorted(_IMAGE_KEYS)}"
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
CARRIER_CODECS: Dict[str, PayloadCodec] = {
    name: carrier_codec(carrier) for name, carrier in NATIVE_CARRIERS.items()
}
NATIVE_RECORDING_CODECS: Tuple[PayloadCodec, ...] = (
    TENSOR_CODEC,
    IMAGE_DATA_CODEC,
    *CARRIER_CODECS.values(),
)
