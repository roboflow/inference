"""Composed RF-DETR execution-plan selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

from inference_models.errors import ModelRuntimeError
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_BUFFER_STRATEGY_BASE,
    RFDETR_ENGINE_PLUGIN_BASE,
    RFDETR_POSTPROCESSOR_BASE,
    RFDETR_POSTPROCESSOR_ENV_NAME,
    RFDETR_PREPROCESSOR_BASE,
    RFDETR_PREPROCESSOR_ENV_NAME,
    RFDETR_SCHEDULER_BASE,
)


@dataclass(frozen=True)
class RFDetrExecutionPlan:
    """Independent implementation selections for the RF-DETR inference path."""

    preprocessor_id: str = RFDETR_PREPROCESSOR_BASE
    buffer_strategy_id: str = RFDETR_BUFFER_STRATEGY_BASE
    scheduler_id: str = RFDETR_SCHEDULER_BASE
    postprocessor_id: str = RFDETR_POSTPROCESSOR_BASE
    engine_plugin_id: str = RFDETR_ENGINE_PLUGIN_BASE

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

    @classmethod
    def resolve(
        cls,
        *,
        execution_plan: Optional["RFDetrExecutionPlan"] = None,
        preprocessor_id: Optional[str] = None,
        postprocessor_id: Optional[str] = None,
    ) -> "RFDetrExecutionPlan":
        """Resolve a plan from an explicit plan, legacy arguments, and environment.

        Args:
            execution_plan: Explicit composed plan. It cannot be mixed with legacy
                per-stage arguments.
            preprocessor_id: Explicit legacy preprocessing implementation ID.
            postprocessor_id: Explicit legacy postprocessing implementation ID.

        Returns:
            Immutable requested execution plan.

        Raises:
            ModelRuntimeError: If an explicit plan is mixed with legacy selections or
                requests an unsupported stage category.
        """
        if execution_plan is not None:
            if preprocessor_id is not None or postprocessor_id is not None:
                raise ModelRuntimeError(
                    message=(
                        "rfdetr_execution_plan cannot be combined with "
                        "rfdetr_preprocessor or rfdetr_postprocessor."
                    ),
                    help_url=(
                        "https://inference-models.roboflow.com/errors/models-runtime/"
                        "#modelruntimeerror"
                    ),
                )
            plan = execution_plan
        else:
            plan = cls(
                preprocessor_id=(
                    preprocessor_id
                    if preprocessor_id is not None
                    else os.getenv(
                        RFDETR_PREPROCESSOR_ENV_NAME,
                        RFDETR_PREPROCESSOR_BASE,
                    )
                ),
                postprocessor_id=(
                    postprocessor_id
                    if postprocessor_id is not None
                    else os.getenv(
                        RFDETR_POSTPROCESSOR_ENV_NAME,
                        RFDETR_POSTPROCESSOR_BASE,
                    )
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
            if value != "base"
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
