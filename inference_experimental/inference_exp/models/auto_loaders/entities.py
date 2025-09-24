from dataclasses import dataclass
from typing import Optional

from inference_exp.weights_providers.entities import BackendType

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
