from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import DependencyModelParametersValidationError
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import Quantization


class DependencyModelParameters(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    model_id_or_path: str
    model_package_id: Optional[str] = Field(default=None)
    backend: Optional[Union[str, BackendType, List[Union[str, BackendType]]]] = Field(
        default=None
    )
    batch_size: Optional[Union[int, Tuple[int, int]]] = Field(default=None)
    quantization: Optional[Union[str, Quantization, List[Union[str, Quantization]]]] = (
        Field(default=None)
    )
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = Field(default=None)
    device: Union[torch.device, str] = Field(default=DEFAULT_DEVICE)
    default_onnx_trt_options: bool = Field(default=True)
    nms_fusion_preferences: Optional[Union[bool, dict]] = Field(default=None)
    model_type: Optional[str] = Field(default=None)
    task_type: Optional[str] = Field(default=None)

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self.model_extra or {}


def prepare_dependency_model_parameters(
    model_parameters: Union[str, dict, DependencyModelParameters],
) -> DependencyModelParameters:
    if isinstance(model_parameters, dict):
        try:
            return DependencyModelParameters.model_validate(model_parameters)
        except ValidationError as error:
            raise DependencyModelParametersValidationError(
                message="Could not validate parameters to initialise dependent model - if you run locally, make sure "
                f"that you initialise model properly, as at least one parameter parameter specified in "
                f"dictionary with model options is invalid. If you use Roboflow hosted offering, contact us to "
                f"get help.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#dependencymodelparametersvalidationerror",
            ) from error
    if isinstance(model_parameters, str):
        model_parameters = DependencyModelParameters(model_id_or_path=model_parameters)
    return model_parameters
