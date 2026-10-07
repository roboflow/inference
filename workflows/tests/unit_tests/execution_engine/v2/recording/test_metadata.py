"""Exact recording of entry metadata, including inheritance barriers."""

import json
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    SampleContext,
    TemporalContext,
    TimeSpan,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.recording import (
    CodecRegistry,
    RecordingCodecError,
    RecordingCorruptError,
)
from roboflow_workflows.execution_engine.v2.recording.metadata import (
    decode_entry_metadata,
    decode_timestamp,
    encode_entry_metadata,
    encode_timestamp,
)

from tests.unit_tests.execution_engine.v2.recording.test_codecs import MemoryBlobs

NTSC = Fraction(1001, 30000)


def stamp(ticks: int, *, base: Fraction = NTSC, clock: str = "media") -> Timestamp:
    return Timestamp(ticks=ticks, time_base=base, clock_id=clock)


def round_trip(metadata: EntryMetadata) -> EntryMetadata:
    registry = CodecRegistry()
    blobs = MemoryBlobs()
    node = encode_entry_metadata(metadata, encoder=registry.encoder(blobs))
    text = json.dumps(node, allow_nan=False)
    decoded = decode_entry_metadata(json.loads(text), decoder=registry.decoder(blobs))
    return decoded


def test_sample_and_temporal_maps_round_trip_exactly() -> None:
    camera = SampleContext(
        source_id="camera-1",
        source_type="video",
        source_metadata={
            "fps": 29.97,
            "tags": ["a", "b"],
            "nested": {"ids": {1, 2}, 3: (4, 5)},
            "exposure": np.float32(0.25),
        },
    )
    observed = stamp(10, base=Fraction(1, 1_000_000_000), clock="engine")
    temporal = TemporalContext(
        observed_coverage=observed,
        media_coverage=TimeSpan(start=stamp(3003), end=stamp(4004)),
        capture_coverage=None,
    )
    metadata = EntryMetadata(
        sample={(): camera, (1,): None, (1, 2): camera},
        temporal={(0,): temporal, (0, 1): None},
    )

    decoded = round_trip(metadata)

    assert decoded == metadata
    assert list(decoded.sample) == [(), (1,), (1, 2)]
    assert decoded.sample[()].source_metadata == camera.source_metadata
    assert type(decoded.sample[()].source_metadata["exposure"]) is np.float32
    assert decoded.temporal[(0,)].media_coverage.start.time_base == NTSC
    assert decoded.temporal[(0,)].capture_coverage is None


def test_explicit_none_barriers_keep_their_lookup_meaning() -> None:
    camera = SampleContext(source_id="camera-1")
    metadata = EntryMetadata(sample={(): camera, (1,): None})

    decoded = round_trip(metadata)

    assert decoded.sample_at((0, 4)) == camera
    assert decoded.sample_at((1, 4)) is None
    assert (1,) in decoded.sample


def test_empty_metadata_round_trips() -> None:
    assert round_trip(EntryMetadata()) == EntryMetadata()


def test_timestamp_keeps_an_exact_rational_time_base() -> None:
    value = stamp(-7, base=Fraction(1, 3))

    node = encode_timestamp(value)

    assert node == {"ticks": -7, "time_base": [1, 3], "clock_id": "media"}
    assert decode_timestamp(node) == value


def test_unencodable_source_metadata_is_located() -> None:
    metadata = EntryMetadata(
        sample={
            (2,): SampleContext(source_id="s", source_metadata={"handle": object()})
        }
    )
    encoder = CodecRegistry().encoder(MemoryBlobs(), location="analysis.frame")

    with pytest.raises(RecordingCodecError) as caught:
        encode_entry_metadata(metadata, encoder=encoder)

    assert "analysis.frame.metadata.sample[2].source_metadata['handle']" in str(
        caught.value
    )


@pytest.mark.parametrize(
    "node",
    [
        {"sample": []},
        {"sample": [[[0], None]], "temporal": [], "extra": 1},
        {"sample": [[[-1], None]], "temporal": []},
        {"sample": [["0", None]], "temporal": []},
        {
            "sample": [],
            "temporal": [[[0], {"observed": None, "media": None, "capture": None}]],
        },
        {
            "sample": [],
            "temporal": [
                [
                    [0],
                    {
                        "observed": {
                            "timestamp": {
                                "ticks": 1,
                                "time_base": [1, 0],
                                "clock_id": "c",
                            }
                        },
                        "media": None,
                        "capture": None,
                    },
                ]
            ],
        },
        {
            "sample": [
                [
                    [0],
                    {
                        "source_id": "",
                        "source_type": "x",
                        "source_metadata": {"dict": {}},
                    },
                ]
            ],
            "temporal": [],
        },
    ],
)
def test_malformed_metadata_is_corruption(node: Any) -> None:
    registry = CodecRegistry()

    with pytest.raises(RecordingCorruptError):
        decode_entry_metadata(node, decoder=registry.decoder(MemoryBlobs()))
