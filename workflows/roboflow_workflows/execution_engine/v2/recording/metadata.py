"""Exact JSON form of entry metadata in a recording.

::

    EntryMetadata   {"sample": [[index, sample|null], ...],
                     "temporal": [[index, temporal|null], ...]}
    sample          {"source_id", "source_type", "source_metadata": <node>}
    temporal        {"observed": cover, "media": cover|null, "capture": cover|null}
    cover           {"timestamp": stamp} | {"span": [stamp, stamp]}
    stamp           {"ticks": int, "time_base": [numerator, denominator],
                     "clock_id": str}

An index explicitly mapped to ``null`` is kept: it is an inheritance barrier,
which an absent index is not. Map order is preserved. ``source_metadata`` goes
through the payload encoder, so its values keep their exact types; the
read-only views ``SampleContext`` creates are rebuilt by its constructor.
"""

from fractions import Fraction
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    Index,
    SampleContext,
    TemporalContext,
    TimeCoverage,
    TimeSpan,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.recording.codecs import (
    JsonValue,
    PayloadDecoder,
    PayloadEncoder,
)
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingCorruptError,
)


def encode_entry_metadata(
    metadata: EntryMetadata, *, encoder: PayloadEncoder
) -> JsonValue:
    """Encode entry metadata exactly.

    Args:
        metadata: Metadata of one recorded entry.
        encoder: Encoder for ``source_metadata`` values.

    Returns:
        The JSON form described in the module docstring.

    Raises:
        RecordingCodecError: When a ``source_metadata`` value has no codec.
    """
    sample = [
        [
            list(index),
            (
                None
                if context is None
                else _encode_sample(context, encoder=encoder, index=index)
            ),
        ]
        for index, context in metadata.sample.items()
    ]
    temporal = [
        [list(index), None if context is None else _encode_temporal(context)]
        for index, context in metadata.temporal.items()
    ]
    encoded = {"sample": sample, "temporal": temporal}

    return encoded


def decode_entry_metadata(node: JsonValue, *, decoder: PayloadDecoder) -> EntryMetadata:
    """Rebuild entry metadata from ``encode_entry_metadata`` output.

    Args:
        node: JSON form of the metadata.
        decoder: Decoder for ``source_metadata`` values.

    Returns:
        Metadata equal to the encoded one, including explicit ``None``
        barriers and map order.

    Raises:
        RecordingCorruptError: When the node is malformed.
    """
    body = _object(node, keys=("sample", "temporal"), what="entry metadata")
    sample = {
        index: None if context is None else _decode_sample(context, decoder=decoder)
        for index, context in _indexed(body["sample"], what="sample")
    }
    temporal = {
        index: None if context is None else _decode_temporal(context)
        for index, context in _indexed(body["temporal"], what="temporal")
    }

    try:
        metadata = EntryMetadata(sample=sample, temporal=temporal)
    except ContractError as error:
        raise RecordingCorruptError(f"invalid entry metadata: {error}") from error

    return metadata


def encode_timestamp(stamp: Timestamp) -> JsonValue:
    """Encode a timestamp with its exact rational time base.

    Args:
        stamp: Timestamp to encode.

    Returns:
        ``{"ticks", "time_base": [numerator, denominator], "clock_id"}``.
    """
    encoded = {
        "ticks": stamp.ticks,
        "time_base": [stamp.time_base.numerator, stamp.time_base.denominator],
        "clock_id": stamp.clock_id,
    }

    return encoded


def decode_timestamp(node: JsonValue) -> Timestamp:
    """Rebuild a timestamp from ``encode_timestamp`` output.

    Args:
        node: JSON form of the timestamp.

    Returns:
        The timestamp.

    Raises:
        RecordingCorruptError: When the node is malformed.
    """
    body = _object(node, keys=("ticks", "time_base", "clock_id"), what="timestamp")
    time_base = body["time_base"]
    if (
        type(time_base) is not list
        or len(time_base) != 2
        or not all(type(part) is int for part in time_base)
    ):
        raise RecordingCorruptError("a timestamp time_base must be [int, int]")

    try:
        stamp = Timestamp(
            ticks=body["ticks"],
            time_base=Fraction(time_base[0], time_base[1]),
            clock_id=body["clock_id"],
        )
    except (ContractError, ZeroDivisionError) as error:
        raise RecordingCorruptError(f"invalid timestamp: {error}") from error

    return stamp


def _encode_sample(
    context: SampleContext, *, encoder: PayloadEncoder, index: Index
) -> JsonValue:
    source_metadata = encoder.encode(
        _thaw(context.source_metadata),
        path=f".metadata.sample{list(index)}.source_metadata",
    )
    encoded = {
        "source_id": context.source_id,
        "source_type": context.source_type,
        "source_metadata": source_metadata,
    }

    return encoded


def _decode_sample(node: JsonValue, *, decoder: PayloadDecoder) -> SampleContext:
    body = _object(
        node, keys=("source_id", "source_type", "source_metadata"), what="sample"
    )
    source_metadata = decoder.decode(body["source_metadata"], path=".source_metadata")
    if type(source_metadata) is not dict:
        raise RecordingCorruptError("sample source_metadata must decode to a dict")

    try:
        context = SampleContext(
            source_id=body["source_id"],
            source_type=body["source_type"],
            source_metadata=source_metadata,
        )
    except ContractError as error:
        raise RecordingCorruptError(f"invalid sample context: {error}") from error

    return context


def _encode_temporal(context: TemporalContext) -> JsonValue:
    encoded = {
        "observed": _encode_cover(context.observed_coverage),
        "media": _encode_cover(context.media_coverage),
        "capture": _encode_cover(context.capture_coverage),
    }

    return encoded


def _decode_temporal(node: JsonValue) -> TemporalContext:
    body = _object(node, keys=("observed", "media", "capture"), what="temporal")
    observed = _decode_cover(body["observed"])
    if observed is None:
        raise RecordingCorruptError("a temporal context needs an observed coverage")

    try:
        context = TemporalContext(
            observed_coverage=observed,
            media_coverage=_decode_cover(body["media"]),
            capture_coverage=_decode_cover(body["capture"]),
        )
    except ContractError as error:
        raise RecordingCorruptError(f"invalid temporal context: {error}") from error

    return context


def _encode_cover(cover: Optional[TimeCoverage]) -> JsonValue:
    if cover is None:
        return None
    if isinstance(cover, TimeSpan):
        return {"span": [encode_timestamp(cover.start), encode_timestamp(cover.end)]}

    encoded = {"timestamp": encode_timestamp(cover)}

    return encoded


def _decode_cover(node: JsonValue) -> Optional[TimeCoverage]:
    if node is None:
        return None
    if type(node) is not dict or len(node) != 1:
        raise RecordingCorruptError("a time coverage must be a timestamp or a span")

    ((tag, body),) = node.items()
    if tag == "timestamp":
        return decode_timestamp(body)
    if tag != "span" or type(body) is not list or len(body) != 2:
        raise RecordingCorruptError("a time coverage must be a timestamp or a span")

    try:
        span = TimeSpan(start=decode_timestamp(body[0]), end=decode_timestamp(body[1]))
    except ContractError as error:
        raise RecordingCorruptError(f"invalid time span: {error}") from error

    return span


def _indexed(node: JsonValue, *, what: str) -> List[Tuple[Index, Any]]:
    if type(node) is not list:
        raise RecordingCorruptError(f"metadata {what} must be a list of pairs")

    pairs = []
    for pair in node:
        if type(pair) is not list or len(pair) != 2:
            raise RecordingCorruptError(f"metadata {what} entries are [index, value]")
        index, context = pair
        if type(index) is not list or not all(
            type(component) is int and component >= 0 for component in index
        ):
            raise RecordingCorruptError(
                f"metadata {what} index must be non-negative integers"
            )
        pairs.append((tuple(index), context))

    return pairs


def _object(node: JsonValue, *, keys: Tuple[str, ...], what: str) -> Dict[str, Any]:
    if type(node) is not dict or set(node) != set(keys):
        raise RecordingCorruptError(
            f"a recorded {what} needs exactly the keys {list(keys)}"
        )

    return node


def _thaw(value: Any) -> Any:
    """Undo ``SampleContext``'s read-only views; it re-applies them on decode."""
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_thaw(item) for item in value)

    return value
