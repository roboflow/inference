"""Plain finite sources: each emits its own samples and knows nothing of others.

Neither source waits for, pairs with or buffers another source. Correspondence
between them is declared in the workflow with ``v2/align`` operators.
"""

import csv
from pathlib import Path

from fixtures import RIG_CLOCK, SCHEDULES, make_image, media_timestamp
from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.declaration import Ref
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)


class TensorCamera(Source):
    """Emit one fixture schedule of native tensor frames, then end."""

    type = "temporal_demo/tensor_camera"
    outputs = {"image": SourceOutput(IMAGE_KIND)}

    class Params(SourceParams):
        schedule: str = Field(
            description="Fixture schedule name from fixtures.SCHEDULES.",
            examples=["left", "right"],
        )

    def open(self, *, schedule: str) -> None:
        """Select the schedule and start at its first frame.

        Args:
            schedule: Fixture schedule name.
        """
        self.name = schedule
        self.schedule = SCHEDULES[schedule]
        self.position = 0

    def read(self) -> Emission | None:
        """Return the next frame with its media timestamp.

        Returns:
            One image emission, or None at the end of the schedule or on stop.
        """
        if self.stop_event.is_set() or self.position == len(self.schedule.frames):
            return None

        frame = self.schedule.frames[self.position]
        self.position += 1
        image = make_image(
            frame,
            schedule=self.schedule,
            image_id=f"{self.name}@{frame.pts_ms}ms",
        )
        emission = Emission(
            {"image": image},
            media=media_timestamp(frame.pts_ms, clock=self.schedule.clock),
            source_metadata={"schedule": self.name},
        )

        return emission


class CsvSensor(Source):
    """Read timestamped scalar readings from a local CSV file."""

    type = "temporal_demo/csv_sensor"
    outputs = {"celsius": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        path: str | Ref(STRING_KIND) = Field(
            description="CSV file with pts_ms and celsius columns."
        )
        clock: str = Field(
            default=RIG_CLOCK, description="Media clock of the pts_ms column."
        )

    def open(self, *, path: str, clock: str) -> None:
        """Open the file on the source's reader thread.

        Args:
            path: CSV file owned by this source.
            clock: Media clock identity of its timestamps.
        """
        self.file = None
        self.file = Path(path).open(newline="")
        self.rows = csv.DictReader(self.file)
        self.clock = clock

    def read(self) -> Emission | None:
        """Return one reading with its media timestamp.

        Returns:
            One scalar emission, or None at EOF or on stop.
        """
        if self.stop_event.is_set():
            return None

        row = next(self.rows, None)
        if row is None:
            return None

        emission = Emission(
            {"celsius": float(row["celsius"])},
            media=media_timestamp(int(row["pts_ms"]), clock=self.clock),
        )

        return emission

    def close(self) -> None:
        """Release the file, including after a partially failed open."""
        if self.file is not None:
            self.file.close()
