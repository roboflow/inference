from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.errors import ModelPipelineInitializationError
from inference_exp.logger import verbose_info
from inference_exp.model_pipelines.auto_loaders.pipelines_registry import (
    REGISTERED_PIPELINES,
    get_default_pipeline_parameters,
    resolve_pipeline_class,
)
from inference_exp.models.auto_loaders.auto_resolution_cache import AutoResolutionCache
from inference_exp.models.auto_loaders.core import AnyModel, AutoModel
from inference_exp.models.auto_loaders.storage_manager import ModelStorageManager
from inference_exp.weights_providers.entities import BackendType, Quantization
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console
from rich.tree import Tree


class PipelineModelParameters(BaseModel):
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
    device: torch.device = Field(default=DEFAULT_DEVICE)
    default_onnx_trt_options: bool = Field(default=True)
    nms_fusion_preferences: Optional[Union[bool, dict]] = Field(default=None)
    model_type: Optional[str] = Field(default=None)
    task_type: Optional[str] = Field(default=None)

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self.model_extra or {}


class AutoModelPipeline:

    @classmethod
    def list_available_pipelines(cls) -> None:
        console = Console()
        tree = Tree("Available Model Pipelines:")
        for pipeline_id in sorted(REGISTERED_PIPELINES):
            tree.add(pipeline_id)
        console.print(tree)

    @classmethod
    def from_pretrained(
        cls,
        pipline_id: str,
        models_parameters: Optional[
            List[Optional[Union[str, dict, PipelineModelParameters]]]
        ] = None,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        max_package_loading_attempts: Optional[int] = None,
        verbose: bool = False,
        model_download_file_lock_acquire_timeout: int = 10,
        allow_untrusted_packages: bool = False,
        trt_engine_host_code_allowed: bool = True,
        allow_local_code_packages: bool = True,
        verify_hash_while_download: bool = True,
        download_files_without_hash: bool = False,
        use_auto_resolution_cache: bool = True,
        auto_resolution_cache: Optional[AutoResolutionCache] = None,
        allow_direct_local_storage_loading: bool = True,
        model_storage_manager: Optional[ModelStorageManager] = None,
        **kwargs,
    ) -> AnyModel:
        pipeline_class = resolve_pipeline_class(pipline_id=pipline_id)
        models = []
        verbose_info(
            message=f"Initializing models for pipeline `{pipline_id}`",
            verbose_requested=verbose,
        )
        default_parameters = get_default_pipeline_parameters(pipline_id=pipline_id)
        if models_parameters is None and default_parameters is None:
            raise ModelPipelineInitializationError(
                message=f"Could not initialize model pipeline `{pipline_id}` - models parameters not provided and "
                f"default values not registered in the library. If you run locally, please verify your "
                f"integration - it must specify the models to be used by the pipeline. If you use Roboflow "
                f"hosted solution, contact us to get help.",
                help_url="https://todo",
            )
        if models_parameters is None:
            models_parameters = default_parameters
        if default_parameters is None:
            default_parameters = [None] * len(models_parameters)
        for idx, model_parameters in enumerate(models_parameters):
            if model_parameters is None:
                parameters_to_be_used = (
                    default_parameters[idx] if idx < len(default_parameters) else None
                )
            else:
                parameters_to_be_used = model_parameters
            resolved_model_parameters = prepare_model_parameters(
                model_parameters=parameters_to_be_used
            )
            verbose_info(
                message=f"Initializing model: `{resolved_model_parameters.model_id_or_path}`",
                verbose_requested=verbose,
            )
            model = AutoModel.from_pretrained(
                model_id_or_path=resolved_model_parameters.model_id_or_path,
                weights_provider=weights_provider,
                api_key=api_key,
                model_package_id=resolved_model_parameters.model_package_id,
                backend=resolved_model_parameters.backend,
                batch_size=resolved_model_parameters.batch_size,
                quantization=resolved_model_parameters.quantization,
                onnx_execution_providers=resolved_model_parameters.onnx_execution_providers,
                device=resolved_model_parameters.device,
                default_onnx_trt_options=resolved_model_parameters.default_onnx_trt_options,
                max_package_loading_attempts=max_package_loading_attempts,
                verbose=verbose,
                model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                allow_untrusted_packages=allow_untrusted_packages,
                trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                allow_local_code_packages=allow_local_code_packages,
                verify_hash_while_download=verify_hash_while_download,
                download_files_without_hash=download_files_without_hash,
                use_auto_resolution_cache=use_auto_resolution_cache,
                auto_resolution_cache=auto_resolution_cache,
                allow_direct_local_storage_loading=allow_direct_local_storage_loading,
                model_storage_manager=model_storage_manager,
                nms_fusion_preferences=resolved_model_parameters.nms_fusion_preferences,
                model_type=resolved_model_parameters.model_type,
                task_type=resolved_model_parameters.task_type,
                **resolved_model_parameters.kwargs,
            )
            models.append(model)
        return pipeline_class.with_models(models, **kwargs)


def prepare_model_parameters(
    model_parameters: Union[str, dict, PipelineModelParameters],
) -> PipelineModelParameters:
    if isinstance(model_parameters, dict):
        try:
            return PipelineModelParameters.model_validate(model_parameters)
        except ValidationError as error:
            raise ModelPipelineInitializationError(
                message="Could not validate parameters to initialise model pipeline - if you run locally, make sure "
                f"that you initialise model properly, as at least one parameter parameter specified in "
                f"dictionary with model options is invalid. If you use Roboflow hosted offering, contact us to "
                f"get help.",
                help_url="https://todo",
            ) from error
    if isinstance(model_parameters, str):
        model_parameters = PipelineModelParameters(model_id_or_path=model_parameters)
    return model_parameters
