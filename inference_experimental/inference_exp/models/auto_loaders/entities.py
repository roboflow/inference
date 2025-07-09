from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from inference_exp.weights_providers.entities import BackendType
from pydantic import BaseModel

ModelArchitecture = str
TaskType = Optional[str]
MODEL_CONFIG_FILE_NAME = "model_config.json"


@dataclass(frozen=True)
class InferenceModelConfig:
    model_architecture: Optional[ModelArchitecture]
    task_type: TaskType
    backend_type: Optional[BackendType]
    model_module: Optional[str]
    model_class: Optional[str]

    def is_library_model(self) -> bool:
        return self.model_architecture is not None and self.backend_type is not None


class AutoResolutionCacheEntry(BaseModel):
    model_id: str
    model_package_id: str
    resolved_files: List[str]
    model_architecture: Optional[ModelArchitecture]
    task_type: TaskType
    backend_type: Optional[BackendType]
    created_at: datetime
