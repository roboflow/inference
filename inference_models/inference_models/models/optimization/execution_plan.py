"""Reusable composed inference execution-plan representation."""

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from inference_models.models.optimization.ids import (
    AUTO_IMPLEMENTATION_ID,
    BASE_IMPLEMENTATION_ID,
)

_IMPLEMENTATION_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_SERIALIZED_FIELDS = {
    "preprocessor",
    "buffer_strategy",
    "scheduler",
    "postprocessor",
    "engine_plugin",
    "allow_compatibility_fallback",
    "allow_runtime_failure_fallback",
}


@dataclass(frozen=True)
class InferenceExecutionPlan:
    """Independent implementation selections and fallback policy.

    Compatibility fallback is the global strictness gate. Runtime-failure fallback
    applies only when both fallback fields are enabled.
    """

    preprocessor_id: str = BASE_IMPLEMENTATION_ID
    buffer_strategy_id: str = BASE_IMPLEMENTATION_ID
    scheduler_id: str = BASE_IMPLEMENTATION_ID
    postprocessor_id: str = BASE_IMPLEMENTATION_ID
    engine_plugin_id: str = BASE_IMPLEMENTATION_ID
    allow_compatibility_fallback: bool = True
    allow_runtime_failure_fallback: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the composed execution plan.

        Returns:
            Stage names mapped to selected implementation IDs.
        """
        serialized = {
            "preprocessor": self.preprocessor_id,
            "buffer_strategy": self.buffer_strategy_id,
            "scheduler": self.scheduler_id,
            "postprocessor": self.postprocessor_id,
            "engine_plugin": self.engine_plugin_id,
            "allow_compatibility_fallback": self.allow_compatibility_fallback,
            "allow_runtime_failure_fallback": self.allow_runtime_failure_fallback,
        }

        return serialized

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceExecutionPlan":
        """Parse the canonical representation emitted by :meth:`to_dict`.

        Args:
            value: Serialized stage selections and fallback policy.

        Returns:
            Parsed immutable execution plan.

        Raises:
            ValueError: If fields, implementation IDs, or fallback values are invalid.
        """
        if not isinstance(value, Mapping):
            raise ValueError("Execution plan must be an object.")

        if set(value) != _SERIALIZED_FIELDS:
            raise ValueError(
                "Execution plan must contain exactly: "
                f"{sorted(_SERIALIZED_FIELDS)!r}."
            )
        implementation_fields = {
            "preprocessor": "preprocessor_id",
            "buffer_strategy": "buffer_strategy_id",
            "scheduler": "scheduler_id",
            "postprocessor": "postprocessor_id",
            "engine_plugin": "engine_plugin_id",
        }
        parsed: Dict[str, Any] = {}
        for serialized_name, attribute_name in implementation_fields.items():
            implementation_id = value[serialized_name]
            if not isinstance(implementation_id, str):
                raise ValueError("Execution plan implementation IDs must be strings.")

            parsed[attribute_name] = implementation_id
        for fallback_name in (
            "allow_compatibility_fallback",
            "allow_runtime_failure_fallback",
        ):
            fallback_value = value[fallback_name]
            if not isinstance(fallback_value, bool):
                raise ValueError("Execution plan fallback values must be booleans.")

            parsed[fallback_name] = fallback_value

        execution_plan = cls(**parsed)

        return execution_plan

    def validate_for_profiling(self) -> "InferenceExecutionPlan":
        """Require an attributable, fully explicit plan with fallback disabled.

        Returns:
            This execution plan after successful validation.

        Raises:
            ValueError: If an ID is invalid or automatic/fallback selection is enabled.
        """
        for stage, implementation_id in self.to_dict().items():
            if stage.startswith("allow_"):
                continue

            if not _IMPLEMENTATION_ID.fullmatch(implementation_id):
                raise ValueError(f"Invalid {stage} implementation ID.")

            if implementation_id == AUTO_IMPLEMENTATION_ID:
                raise ValueError(
                    f"Profiling execution plan {stage} must not use 'auto'."
                )

        if self.allow_compatibility_fallback or self.allow_runtime_failure_fallback:
            raise ValueError(
                "Profiling execution plans must disable both fallback modes."
            )

        return self
