"""Reusable composed inference execution-plan representation."""

from dataclasses import dataclass
from typing import Dict

from inference_models.models.optimization.ids import BASE_IMPLEMENTATION_ID


@dataclass(frozen=True)
class InferenceExecutionPlan:
    """Independent implementation selections for an inference path."""

    preprocessor_id: str = BASE_IMPLEMENTATION_ID
    buffer_strategy_id: str = BASE_IMPLEMENTATION_ID
    scheduler_id: str = BASE_IMPLEMENTATION_ID
    postprocessor_id: str = BASE_IMPLEMENTATION_ID
    engine_plugin_id: str = BASE_IMPLEMENTATION_ID

    def to_dict(self) -> Dict[str, str]:
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
        }

        return serialized
