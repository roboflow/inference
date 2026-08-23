"""YOLO26 depth-estimation execution-plan selection."""

import os
from dataclasses import dataclass
from typing import Optional

from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_POSTPROCESSOR_BASE,
    YOLO26_DEPTH_POSTPROCESSOR_ENV_NAME,
)


@dataclass(frozen=True)
class YOLO26DepthExecutionPlan(InferenceExecutionPlan):
    """Select the YOLO26 depth-estimation postprocessing implementation."""

    postprocessor_id: str = YOLO26_DEPTH_POSTPROCESSOR_BASE

    @classmethod
    def resolve(
        cls,
        *,
        execution_plan: Optional["YOLO26DepthExecutionPlan"] = None,
        postprocessor_id: Optional[str] = None,
        allow_compatibility_fallback: bool = True,
    ) -> "YOLO26DepthExecutionPlan":
        """Resolve an explicit plan, implementation ID, or environment selection.

        Args:
            execution_plan: Complete explicit plan. Mutually exclusive with
                ``postprocessor_id``.
            postprocessor_id: Explicit postprocessor ID supplied by a model loader.
            allow_compatibility_fallback: Whether static incompatibility may follow
                the candidate's declared fallback to ``base``.

        Returns:
            Immutable requested execution plan.

        Raises:
            ValueError: If both explicit selection forms are supplied.
        """
        if execution_plan is not None and postprocessor_id is not None:
            raise ValueError(
                "Specify either execution_plan or postprocessor_id, not both."
            )

        if execution_plan is not None:
            plan = execution_plan
        else:
            resolved_postprocessor_id = postprocessor_id or os.getenv(
                YOLO26_DEPTH_POSTPROCESSOR_ENV_NAME,
                YOLO26_DEPTH_POSTPROCESSOR_BASE,
            )
            plan = cls(
                postprocessor_id=resolved_postprocessor_id,
                allow_compatibility_fallback=allow_compatibility_fallback,
            )

        return plan
