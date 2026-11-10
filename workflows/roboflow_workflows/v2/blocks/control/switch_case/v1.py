"""Route to the step of the case whose key equals ``str(value)``.

Canonical type: ``roboflow_core/switch_case@v1``.
"""

from typing import Any, Dict, List
import torch
from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Ref,
    Select,
    StepRef,
    Stop,
)


def _scalar(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value

    if value.dim() != 0:
        raise TypeError(
            "roboflow_core/switch_case@v1 matches one value; got a tensor of shape "
            f"{tuple(value.shape)}. Extract one element first."
        )

    scalar = value.item()

    return scalar


class SwitchCaseBlock(Block):
    """Route to the step of the case whose key equals ``str(value)``.

    Example: ``cases={"dog": "$steps.dogs", "car": "$steps.cars"}`` with
    ``default_next_steps=["$steps.review"]`` sends a ``"dog"`` top class to
    ``dogs``, ``"car"`` to ``cars`` and anything else to ``review``. Bound to
    per-crop values, each crop is routed alone.
    """

    type = "roboflow_core/switch_case@v1"
    metadata = {
        "name": "Switch Case",
        "port_of": "roboflow_core/switch_case@v1",
        "section": "flow_control",
    }

    class Params(BlockParams):
        value: (
            StrictStr | StrictBool | StrictInt | StrictFloat | Ref(allow_none=True)
        ) = Field(
            description=(
                "Value matched against the case keys after str(): True matches "
                "'True', 1.0 matches '1.0' (not '1'), None matches 'None'. A 0-d "
                "tensor is read as its Python scalar first. Bind one value such "
                "as top_class, not a predictions object: its str() matches no "
                "key, so only default_next_steps run."
            ),
            examples=["$steps.top_class.output", "$inputs.mode", "red"],
        )
        cases: Dict[str, StepRef] = Field(
            default_factory=dict,
            description=(
                "Case key to the step that runs when value matches it. A step "
                "may be targeted once across cases and default_next_steps."
            ),
            examples=[{"dog": "$steps.dogs", "car": "$steps.cars"}],
        )
        case_insensitive: StrictBool = Field(
            default=False,
            description="Match keys ignoring letter case ('RED' matches 'red').",
            examples=[False],
        )
        default_next_steps: List[StepRef] = Field(
            default_factory=list,
            description="Steps that run when no case matches; empty stops.",
            examples=[["$steps.review"]],
        )

        @model_validator(mode="after")
        def _targets_and_keys_are_unique(self) -> "SwitchCaseBlock.Params":
            targets = [*self.cases.values(), *self.default_next_steps]
            duplicated = sorted(
                {target for target in targets if targets.count(target) > 1}
            )
            if duplicated:
                raise ValueError(
                    "Each step may be targeted at most once across `cases` and "
                    f"`default_next_steps`. Duplicated targets: {duplicated}"
                )

            lowered = [key.lower() for key in self.cases]
            if self.case_insensitive and len(set(lowered)) != len(lowered):
                raise ValueError(
                    "`cases` contains keys that collide when case_insensitive "
                    "is enabled"
                )

            return self

    def run(
        self,
        *,
        value: Any,
        cases: Dict[str, str],
        case_insensitive: bool,
        default_next_steps: List[str],
    ) -> Select:
        """Select the matching case's step, else the default steps.

        Args:
            value: Value to match; ``None`` matches the key ``"None"``.
            cases: Case key to step selector.
            case_insensitive: Whether keys match ignoring letter case.
            default_next_steps: Steps selected when no case matches.

        Returns:
            ``Select`` of the matched step, of ``default_next_steps`` when
            nothing matches, or ``Stop()`` when both are empty.

        Raises:
            TypeError: When ``value`` is a tensor with one or more dimensions.
        """
        key = str(_scalar(value))
        if case_insensitive:
            key = key.lower()
            matched = next(
                (target for case, target in cases.items() if case.lower() == key),
                None,
            )
        else:
            matched = cases.get(key)

        if matched is not None:
            return Select(matched)

        decision = Select(default_next_steps) if default_next_steps else Stop()

        return decision
