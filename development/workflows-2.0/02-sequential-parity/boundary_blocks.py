"""V2 blocks for the boundary examples: validation, nested inputs and context.

* ``capability/threshold@v1``: literal and selected values meet the same
  ``Params`` rules (outer ``Field`` constraints, a field validator and a
  model validator); a selected mapping keeps its identity.
* ``settings`` kind: decodes ``"gain=2"`` into ``{"gain": 2}`` and counts its
  decoder calls, so the nested example can show when decoding happens.
* ``capability/context_probe@v1``: reports the execution context seen in its
  constructor and in ``run``; ``fail_on`` makes a chosen value raise.
"""

from typing import Annotated, Any, Dict

from pydantic import Field, field_validator, model_validator
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Kind,
)

BOUNDARY_NAMESPACE = "boundary"
SETTINGS_DECODER_CALLS = []


def _decode_settings(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    SETTINGS_DECODER_CALLS.append(value)
    key, _, number = value.partition("=")

    return {key: int(number)}


SETTINGS_KIND = Kind(
    name="settings",
    description="Mapping of one setting; accepts 'gain=2' at the boundary.",
    validate=lambda payload: isinstance(payload, dict),
    deserialize=_decode_settings,
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Threshold(Block):
    """Compare a score with a threshold; count calls in the selected options."""

    type = "capability/threshold@v1"
    outputs = {"passed": Output(BOOLEAN_KIND)}
    mutates = ("options",)

    class Params(BlockParams):
        score: Ref(FLOAT_KIND) = Field(description="Score to compare.")
        threshold: float | Ref(FLOAT_KIND) = Field(
            default=0.5,
            ge=0,
            le=1,
            description="Shared constraint: literal or selected.",
        )
        margin: Annotated[float, Field(ge=0)] | Ref(FLOAT_KIND) = Field(
            default=0.0, description="The ge=0 constraint applies to literals only."
        )
        low: float | Ref(FLOAT_KIND) = Field(default=0.0, description="Lower bound.")
        high: float | Ref(FLOAT_KIND) = Field(default=1.0, description="Upper bound.")
        label: str | Ref(STRING_KIND) = Field(default="score", description="Label.")
        options: Dict[str, Any] | Ref(DICTIONARY_KIND) = Field(
            default_factory=dict, description="Mapping counting calls in place."
        )

        @field_validator("label")
        @classmethod
        def _label_not_blank(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("label must not be blank")

            return value

        @model_validator(mode="after")
        def _bounds_ordered(self) -> "Threshold.Params":
            # At definition time bounds may still be selector strings.
            if _is_number(self.low) and _is_number(self.high) and self.low > self.high:
                raise ValueError(f"low {self.low} must not exceed high {self.high}")

            return self

    def run(
        self,
        *,
        score: float,
        threshold: float,
        margin: float,
        low: float,
        high: float,
        label: str,
        options: Dict[str, Any],
    ) -> dict:
        options["calls"] = options.get("calls", 0) + 1

        return {"passed": score >= threshold + margin}


class ReadGain(Block):
    """Read ``gain`` from a settings mapping."""

    type = "capability/read_gain@v1"
    outputs = {"gain": Output()}

    class Params(BlockParams):
        settings: Ref(SETTINGS_KIND) = Field(description="Settings mapping.")

    def run(self, *, settings: Dict[str, Any]) -> dict:
        return {"gain": settings.get("gain")}


class ContextProbe(Block):
    """Report the execution context of the constructor and of each call."""

    type = "capability/context_probe@v1"
    outputs = {"seen": Output()}

    class Params(BlockParams):
        value: Ref() = Field(description="Value of this call.")
        fail_on: int | Ref(INTEGER_KIND) = Field(
            default=-1, description="Raise when value equals this."
        )

    def __init__(self):
        context = self.execution_context
        self.constructed_in = {
            "step_selector": context.step_selector,
            "session_id": context.session_id,
            "run_id": context.run_id,
            "indices": [list(index) for index in context.indices],
        }

    def run(self, *, value: Any, fail_on: int) -> dict:
        context = self.execution_context
        if value == fail_on:
            raise RuntimeError(f"probe asked to fail on {value!r}")

        seen = {
            "step_selector": context.step_selector,
            "session_id": context.session_id,
            "run_id": context.run_id,
            "indices": [list(index) for index in context.indices],
        }

        return {"seen": seen}


BOUNDARY_BLOCKS = (Threshold, ReadGain, ContextProbe)


def create_boundary_catalogue() -> Catalogue:
    """Collect the boundary blocks and the ``settings`` kind.

    Returns:
        Catalogue in namespace ``boundary``.
    """
    catalogue = Catalogue(
        BOUNDARY_BLOCKS, kinds=[SETTINGS_KIND], namespace=BOUNDARY_NAMESPACE
    )

    return catalogue
