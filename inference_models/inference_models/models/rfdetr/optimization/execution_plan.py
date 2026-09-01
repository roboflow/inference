"""Composed RF-DETR execution-plan selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.optimization.ids import AUTO_IMPLEMENTATION_ID
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_POSTPROCESSOR_ENV_NAME,
    RFDETR_PREPROCESSOR_ENV_NAME,
)


@dataclass(frozen=True)
class RFDetrExecutionPlan(InferenceExecutionPlan):
    """Independent implementation selections for the RF-DETR inference path."""

    preprocessor_id: str = AUTO_IMPLEMENTATION_ID
    postprocessor_id: str = AUTO_IMPLEMENTATION_ID

    @classmethod
    def resolve(
        cls,
        *,
        execution_plan: Optional[
            Union[InferenceExecutionPlan, Mapping[str, Any]]
        ] = None,
    ) -> "RFDetrExecutionPlan":
        """Resolve a plan from an explicit plan or RF-DETR environment values.

        Args:
            execution_plan: Existing typed or canonical serialized composed plan.

        Returns:
            Immutable requested execution plan.

        Raises:
            ValueError: If a serialized execution plan is invalid.
        """
        if execution_plan is not None:
            if isinstance(execution_plan, cls):
                return execution_plan

            generic_plan = (
                execution_plan
                if isinstance(execution_plan, InferenceExecutionPlan)
                else InferenceExecutionPlan.from_dict(execution_plan)
            )
            plan = cls(
                preprocessor_id=generic_plan.preprocessor_id,
                buffer_strategy_id=generic_plan.buffer_strategy_id,
                scheduler_id=generic_plan.scheduler_id,
                postprocessor_id=generic_plan.postprocessor_id,
                engine_plugin_id=generic_plan.engine_plugin_id,
                allow_compatibility_fallback=(
                    generic_plan.allow_compatibility_fallback
                ),
                allow_runtime_failure_fallback=(
                    generic_plan.allow_runtime_failure_fallback
                ),
            )
        else:
            plan = cls(
                preprocessor_id=os.getenv(
                    RFDETR_PREPROCESSOR_ENV_NAME,
                    AUTO_IMPLEMENTATION_ID,
                ),
                postprocessor_id=os.getenv(
                    RFDETR_POSTPROCESSOR_ENV_NAME,
                    AUTO_IMPLEMENTATION_ID,
                ),
            )

        return plan
