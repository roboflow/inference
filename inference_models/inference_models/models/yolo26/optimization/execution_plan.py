"""Composed execution-plan selection for YOLO26 depth ONNX."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_ONNX_SCHEDULER_ENV_NAME,
)


@dataclass(frozen=True)
class YOLO26DepthOnnxExecutionPlan(InferenceExecutionPlan):
    """Independent implementation selections for YOLO26 depth ONNX."""

    @classmethod
    def resolve(
        cls,
        *,
        execution_plan: Optional[
            Union["YOLO26DepthOnnxExecutionPlan", Dict[str, Any]]
        ] = None,
    ) -> "YOLO26DepthOnnxExecutionPlan":
        """Resolve an object/JSON plan or read the scheduler environment override."""
        if execution_plan is not None:
            if isinstance(execution_plan, dict):
                return cls(**execution_plan)
            return execution_plan
        scheduler_id = os.getenv(YOLO26_DEPTH_ONNX_SCHEDULER_ENV_NAME)
        if scheduler_id is None:
            return cls()
        return cls(scheduler_id=scheduler_id)
