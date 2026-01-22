"""
This module contains set of symbols that should be used by developers creating custom models which are intended to be
used with `inference_models` package without contributing to the package itself. Module expose pretty much all
functions that are used by various models already contributed to the core of library.

This interface is meant to provide stable reference point for anyone who wants to create custom model implementation
that will be compatible with `inference_models` package, without streamlining model to main repository.

We will keep track of changes in the interface in the changelog, minimizing breaking changes and keep clients
informed about changes.

The modul contains standard imports for symbols which are safe to use within basic set of dependencies installed
along with library. Utilities depending on optional dependencies are exposed as lazy imports.
"""

from typing import Any, Dict

from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.runtime_introspection.core import (
    RuntimeXRayResult,
    x_ray_runtime_environment,
)
from inference_models.utils.download import download_files_to_directory
from inference_models.utils.imports import LazyFunction
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)
from inference_models.weights_providers.core import (
    get_model_from_provider,
    register_model_provider,
)
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    ModelDependency,
    ModelMetadata,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
    ServerEnvironmentRequirements,
    TorchScriptPackageDetails,
    TRTPackageDetails,
)

OPTIONAL_IMPORTS: Dict[str, LazyFunction] = {
    "use_primary_cuda_context": LazyFunction(
        module_name="inference_models.models.common.cuda",
        function_name="use_primary_cuda_context",
    ),
    "use_cuda_context": LazyFunction(
        module_name="inference_models.models.common.cuda",
        function_name="use_cuda_context",
    ),
    "set_onnx_execution_provider_defaults": LazyFunction(
        module_name="inference_models.models.common.onnx",
        function_name="set_onnx_execution_provider_defaults",
    ),
    "run_onnx_session_with_batch_size_limit": LazyFunction(
        module_name="inference_models.models.common.onnx",
        function_name="run_onnx_session_with_batch_size_limit",
    ),
    "run_onnx_session_via_iobinding": LazyFunction(
        module_name="inference_models.models.common.onnx",
        function_name="run_onnx_session_via_iobinding",
    ),
    "generate_batch_chunks": LazyFunction(
        module_name="inference_models.models.common.torch",
        function_name="generate_batch_chunks",
    ),
    "get_trt_engine_inputs_and_outputs": LazyFunction(
        module_name="inference_models.models.common.trt",
        function_name="get_trt_engine_inputs_and_outputs",
    ),
    "infer_from_trt_engine": LazyFunction(
        module_name="inference_models.models.common.trt",
        function_name="infer_from_trt_engine",
    ),
    "load_trt_model": LazyFunction(
        module_name="inference_models.models.common.trt",
        function_name="load_trt_model",
    ),
}


__all__ = [
    "get_model_package_contents",
    "x_ray_runtime_environment",
    "RuntimeXRayResult",
    "download_files_to_directory",
    "get_selected_onnx_execution_providers",
    "get_model_from_provider",
    "register_model_provider",
    "ModelMetadata",
    "ModelDependency",
    "ModelPackageMetadata",
    "TorchScriptPackageDetails",
    "ONNXPackageDetails",
    "TRTPackageDetails",
    "JetsonEnvironmentRequirements",
    "ServerEnvironmentRequirements",
    "FileDownloadSpecs",
    "Quantization",
]


def __getattr__(name: str) -> Any:
    if name in OPTIONAL_IMPORTS:
        return OPTIONAL_IMPORTS[name].resolve()
    raise AttributeError(f"module {__name__} has no attribute {name}")
