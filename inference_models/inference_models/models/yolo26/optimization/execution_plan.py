"""YOLO26 depth-estimation execution-plan selection."""

import os
from dataclasses import dataclass
from typing import Optional

from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.optimization.ids import (
    AUTO_IMPLEMENTATION_ID,
    BASE_IMPLEMENTATION_ID,
)
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_POSTPROCESSOR_ENV_NAME,
    YOLO26_DEPTH_PREPROCESSOR_ENV_NAME,
    YOLO26_DEPTH_SCHEDULER_ENV_NAME,
)


@dataclass(frozen=True)
class YOLO26DepthExecutionPlan(InferenceExecutionPlan):
    """Select YOLO26 depth-estimation preprocessing and postprocessing."""

    postprocessor_id: str = AUTO_IMPLEMENTATION_ID

    @classmethod
    def resolve(
        cls,
        *,
        execution_plan: Optional["YOLO26DepthExecutionPlan"] = None,
        preprocessor_id: Optional[str] = None,
        scheduler_id: Optional[str] = None,
        postprocessor_id: Optional[str] = None,
        allow_compatibility_fallback: bool = True,
    ) -> "YOLO26DepthExecutionPlan":
        """Resolve an explicit plan, implementation ID, or environment selection.

        Args:
            execution_plan: Complete explicit plan. Mutually exclusive with
                stage-specific implementation IDs.
            preprocessor_id: Explicit preprocessor ID supplied by a model loader.
            scheduler_id: Explicit scheduler ID supplied by a model loader.
            postprocessor_id: Explicit postprocessor ID supplied by a model loader.
            allow_compatibility_fallback: Whether static incompatibility may follow
                the candidate's declared fallback to ``base``.

        Returns:
            Immutable requested execution plan.

        Raises:
            ValueError: If a complete plan and stage-specific IDs are supplied.
        """
        if execution_plan is not None and any(
            implementation_id is not None
            for implementation_id in (
                preprocessor_id,
                scheduler_id,
                postprocessor_id,
            )
        ):
            raise ValueError(
                "Specify either execution_plan or stage-specific implementation "
                "IDs, not both."
            )

        if execution_plan is not None:
            plan = execution_plan
        else:
            resolved_preprocessor_id = preprocessor_id or os.getenv(
                YOLO26_DEPTH_PREPROCESSOR_ENV_NAME,
                BASE_IMPLEMENTATION_ID,
            )
            resolved_postprocessor_id = postprocessor_id or os.getenv(
                YOLO26_DEPTH_POSTPROCESSOR_ENV_NAME,
                AUTO_IMPLEMENTATION_ID,
            )
            resolved_scheduler_id = scheduler_id or os.getenv(
                YOLO26_DEPTH_SCHEDULER_ENV_NAME,
                BASE_IMPLEMENTATION_ID,
            )
            plan = cls(
                preprocessor_id=resolved_preprocessor_id,
                scheduler_id=resolved_scheduler_id,
                postprocessor_id=resolved_postprocessor_id,
                allow_compatibility_fallback=allow_compatibility_fallback,
            )

        return plan
