"""Composed RF-DETR execution-plan selection."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.optimization.ids import AUTO_IMPLEMENTATION_ID
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_ALLOW_COMPATIBILITY_FALLBACK_ENV_NAME,
    RFDETR_ALLOW_RUNTIME_FAILURE_FALLBACK_ENV_NAME,
    RFDETR_POSTPROCESSOR_ENV_NAME,
    RFDETR_PREPROCESSOR_ENV_NAME,
)
from inference_models.utils.environment import get_boolean_from_env


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
            Union["RFDetrExecutionPlan", Mapping[str, Any]]
        ] = None,
    ) -> "RFDetrExecutionPlan":
        """Resolve a plan from an explicit plan or RF-DETR environment values.

        Args:
            execution_plan: Explicit composed plan or canonical serialized mapping.
                When omitted, stage IDs are read from the RF-DETR environment
                variables, including fallback policy.

        Returns:
            Immutable requested execution plan.

        Raises:
            ValueError: If the serialized execution plan is invalid.
        """
        if execution_plan is not None:
            plan = (
                execution_plan
                if isinstance(execution_plan, cls)
                else cls.from_dict(execution_plan)
            )
        else:
            plan = cls(
                preprocessor_id=os.getenv(
                    RFDETR_PREPROCESSOR_ENV_NAME,
                    AUTO_IMPLEMENTATION_ID,
                ),
                allow_compatibility_fallback=get_boolean_from_env(
                    RFDETR_ALLOW_COMPATIBILITY_FALLBACK_ENV_NAME, default=True
                ),
                allow_runtime_failure_fallback=get_boolean_from_env(
                    RFDETR_ALLOW_RUNTIME_FAILURE_FALLBACK_ENV_NAME,
                    default=True,
                ),
                postprocessor_id=os.getenv(
                    RFDETR_POSTPROCESSOR_ENV_NAME,
                    AUTO_IMPLEMENTATION_ID,
                ),
            )

        return plan


def _normalize_execution_plan_argument(*, execution_plan, kwargs):
    """Consume the deprecated loader alias without changing default resolution."""
    if "rfdetr_execution_plan" in kwargs:
        if execution_plan is not None:
            raise TypeError(
                "Cannot pass both 'rfdetr_execution_plan' and 'execution_plan'; "
                "use 'execution_plan' only."
            )

        warnings.warn(
            "'rfdetr_execution_plan' is deprecated and will be removed on "
            "October 24, 2026. Use 'execution_plan' instead.",
            FutureWarning,
            stacklevel=3,
        )
        execution_plan = kwargs.pop("rfdetr_execution_plan")

    return execution_plan
