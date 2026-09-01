"""Composed RF-DETR execution-plan selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

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
        execution_plan: Optional["RFDetrExecutionPlan"] = None,
    ) -> "RFDetrExecutionPlan":
        """Resolve a plan from an explicit plan or RF-DETR environment values.

        Args:
            execution_plan: Explicit composed plan. When omitted, stage IDs are read
                from the RF-DETR environment variables.

        Returns:
            Immutable requested execution plan.
        """
        if execution_plan is not None:
            plan = execution_plan
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
