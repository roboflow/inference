"""Composed RF-DETR execution-plan selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.optimization.ids import BASE_IMPLEMENTATION_ID
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_POSTPROCESSOR_ENV_NAME,
    RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
    RFDETR_PREPROCESSOR_ENV_NAME,
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
)


@dataclass(frozen=True)
class RFDetrExecutionPlan(InferenceExecutionPlan):
    """Independent implementation selections for the RF-DETR inference path."""

    preprocessor_id: str = RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    postprocessor_id: str = RFDETR_POSTPROCESSOR_TRITON_FUSED_V1

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

        Raises:
            ModelRuntimeError: If the plan requests an unsupported stage category.
        """
        if execution_plan is not None:
            plan = execution_plan
        else:
            plan = cls(
                preprocessor_id=os.getenv(
                    RFDETR_PREPROCESSOR_ENV_NAME,
                    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
                ),
                postprocessor_id=os.getenv(
                    RFDETR_POSTPROCESSOR_ENV_NAME,
                    RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
                ),
            )

        plan._validate_supported_stage_categories()

        return plan

    def _validate_supported_stage_categories(self) -> None:
        unsupported = {
            "buffer_strategy_id": self.buffer_strategy_id,
            "scheduler_id": self.scheduler_id,
            "engine_plugin_id": self.engine_plugin_id,
        }
        selected = [
            f"{name}={value!r}"
            for name, value in unsupported.items()
            if value != BASE_IMPLEMENTATION_ID
        ]
        if selected:
            raise ModelRuntimeError(
                message=(
                    "RF-DETR does not yet provide these execution-plan stages: "
                    + ", ".join(selected)
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )
