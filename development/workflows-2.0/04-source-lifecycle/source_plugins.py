"""Plain finite sources: declarations, acquisition, one emission and cleanup."""

import csv
from pathlib import Path

from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.data import EntryMetadata, InputValue
from roboflow_workflows.execution_engine.v2.declaration import Ref
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)
from source_fixtures import FRAME_PTS_MS, make_frame, make_timing


class CsvTemperature(Source):
    """Read one local CSV independently of any other source or output handler."""

    type = "source_demo/csv_temperature"
    outputs = {"celsius": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        path: str | Ref(STRING_KIND) = Field(
            description="CSV file owned by the source."
        )

    def open(self, *, path: str) -> None:
        """Acquire the file on the engine's reader thread.

        Args:
            path: CSV file with pts_ms, observed_ms and celsius columns.
        """
        self.file = None
        self.file = Path(path).open(newline="")
        self.rows = csv.DictReader(self.file)

    def read(self) -> Emission | None:
        """Return one timestamped reading, or None on EOF/cooperative stop.

        Returns:
            Scalar temperature with indexed timestamp metadata.
        """
        if self.stop_event.is_set():
            return None

        row = next(self.rows, None)
        if row is None:
            return None

        timing = make_timing(
            pts_ms=int(row["pts_ms"]),
            observed_ms=int(row["observed_ms"]),
            media_clock="thermometer-media",
        )
        emission = Emission(
            data={
                "celsius": InputValue(
                    data=float(row["celsius"]),
                    metadata=EntryMetadata(temporal={(): timing}),
                )
            },
            source_metadata={"fixture": "temperatures.csv"},
        )

        return emission

    def close(self) -> None:
        """Release the file, including after a partially failed open."""
        if self.file is not None:
            self.file.close()


class TensorFrames(Source):
    """Generate five native frames without cameras, models or host conversion."""

    type = "source_demo/tensor_frames"
    outputs = {"image": SourceOutput(IMAGE_KIND)}

    def open(self) -> None:
        """Start this fresh source instance at the first frame."""
        self.index = 0

    def read(self) -> Emission | None:
        """Emit one newly allocated image and transfer its ownership to the engine.

        Returns:
            Image plus metadata, or None after the finite stream/stop.
        """
        if self.stop_event.is_set() or self.index == len(FRAME_PTS_MS):
            return None

        index = self.index
        self.index += 1
        timing = make_timing(
            pts_ms=FRAME_PTS_MS[index],
            observed_ms=20000 + FRAME_PTS_MS[index],
            media_clock="camera-media",
        )
        emission = Emission(
            data={
                "image": InputValue(
                    data=make_frame(index),
                    metadata=EntryMetadata(temporal={(): timing}),
                )
            },
            source_metadata={"fixture_frame": index},
        )

        return emission
