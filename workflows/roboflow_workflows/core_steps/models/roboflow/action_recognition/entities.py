"""Action-recognition payload type, owned by Workflows.

Moved here from `inference.core.entities.responses.action_recognition`, which
now re-exports it: the workflow kind, the serializers' `isinstance` dispatch and
the HTTP `ActionRecognitionInferenceResponse` must all see ONE class object.
"""

from threading import Lock
from typing import List, Optional, Sequence, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)


class ActionRecognitionPrediction(BaseModel):
    """One classified frame range of a video.

    The HTTP response and the workflow kind carry the same shape, so it is
    declared once here and imported by both. A field added on one transport
    would otherwise be silently missing from the other.
    """

    model_config = ConfigDict(populate_by_name=True)

    start_frame_idx: int = Field(description="First frame of the range")
    end_frame_idx: int = Field(description="Last frame of the range")
    class_name: str = Field(alias="class", description="The class name")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    class_id: int = Field(
        description=(
            "The class position in the model's own class list. A model without "
            "a class list reports -1."
        )
    )

    @model_serializer(mode="wrap")
    def serialize_prediction(self, handler: SerializerFunctionWrapHandler):
        """Serialize optional scores without replacing the prediction schema.

        Args:
            handler (SerializerFunctionWrapHandler): Pydantic's field serializer.

        Returns:
            The prediction mapping, with confidence only when a score exists.
        """
        result = handler(self)
        # Unscored models retain their existing response shape.
        if self.confidence is None:
            result.pop("confidence", None)
        return result


class _SharedTimeline(list):
    """Per-frame list backed by validated snapshot entries until its first public read."""

    _copy_lock = Lock()

    def detach(self):
        # Keep the list identity for shallow copies, serialization, and dict(result).
        with self._copy_lock:
            if not self.detached:
                self[:] = [entry.model_copy() for entry in self]
                self.detached = True

    detached = False


class ActionRecognitionResult(BaseModel):
    """Cumulative action recognition output for one video stream frame.

    Bundles the classified timeline together with the inclusive source-frame
    bounds of the latest successful window and any per-frame error status, so
    downstream blocks receive a consistent snapshot through one connection.
    """

    timeline: List[ActionRecognitionPrediction] = Field(
        description=(
            "Cumulative classified frame segments for the stream, sorted by "
            "start frame."
        ),
    )
    analyzed_from_frame: Optional[int] = Field(
        default=None,
        description=(
            "Inclusive first source frame covered by the latest successful "
            "classification window, or null before the first success."
        ),
    )
    analyzed_until_frame: Optional[int] = Field(
        default=None,
        description=(
            "Inclusive last source frame covered by the latest successful "
            "classification window, or null before the first success."
        ),
    )
    covered_frame_ranges: Optional[Tuple[Tuple[int, int], ...]] = Field(
        default=None,
        description=(
            "Ordered, coalesced inclusive source-frame ranges covered by successful "
            "windows, including windows without actions. Tracks the last 900 seconds "
            "of source video through analyzed_until_frame. Older producers omit this field."
        ),
        examples=[[[2, 15], [18, 22]]],
    )
    stream_generation: Optional[str] = Field(
        default=None,
        description="Opaque identifier for the current lifetime of the model stream. Changes on reset.",
        examples=["f6339ed433184796841edd443898caf5"],
    )
    source_fps: Optional[float] = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Pinned source frames per second used to record window coverage.",
        examples=[30.0],
    )
    error_status: str = Field(
        default="",
        description=(
            "Error or gap message for the current frame, or an empty string "
            "when no error occurred."
        ),
    )

    def _timeline_for_reading(self) -> Sequence[ActionRecognitionPrediction]:
        """Read entries without copying. Internal consumers must neither edit nor expose them."""
        return super().__getattribute__("timeline")

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if name == "timeline" and isinstance(value, _SharedTimeline):
            if not value.detached:
                value.detach()

        return value

    def __iter__(self):
        # Pydantic's dict(result) otherwise exposes shared entries through __dict__.
        for key, value in super().__iter__():
            yield key, self.timeline if key == "timeline" else value

    @field_validator("timeline", mode="wrap")
    @classmethod
    def _validate_entries(cls, value, handler):
        if isinstance(value, _SharedTimeline) and not value.detached:
            # Each entry was validated when the model built its private snapshot.
            return value

        if isinstance(value, (list, tuple)):
            # Existing Python entries may have been edited without validation.
            value = [
                vars(entry) if isinstance(entry, ActionRecognitionPrediction) else entry
                for entry in value
            ]
        return handler(value)

    @model_validator(mode="after")
    def _validate_coverage(self) -> "ActionRecognitionResult":
        start, end = self.analyzed_from_frame, self.analyzed_until_frame
        if (start is None) != (end is None):
            raise ValueError("Set both analyzed frame bounds or leave both null.")
        if start is not None and start > end:
            raise ValueError(
                "analyzed_from_frame must not exceed analyzed_until_frame."
            )

        ranges = self.covered_frame_ranges
        if ranges is None:
            return self

        if not ranges:
            if end is not None:
                raise ValueError("Coverage must include the latest successful window.")
            return self

        previous_end = None
        for first, last in ranges:
            if first > last or (previous_end is not None and first <= previous_end + 1):
                raise ValueError("Coverage ranges must be ordered and coalesced.")
            previous_end = last

        if end != ranges[-1][1] or start is None or start < ranges[-1][0]:
            raise ValueError("Coverage must end with the latest successful window.")

        return self
