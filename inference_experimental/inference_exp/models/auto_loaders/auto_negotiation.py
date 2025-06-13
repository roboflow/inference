from functools import cache
from typing import List, Optional, Set, Tuple, Union

import torch

from inference_exp.configuration import ONNXRUNTIME_EXECUTION_PROVIDERS
from inference_exp.errors import (
    AmbiguousModelPackageResolutionError,
    InvalidRequestedBatchSizeError,
    ModelPackageNegotiationError,
    NoModelPackagesAvailableError,
    UnknownBackendTypeError,
    UnknownQuantizationError,
)
from inference_exp.models.auto_loaders.ranking import rank_model_packages
from inference_exp.models.auto_loaders.utils import (
    filter_available_devices_with_selected_device,
)
from inference_exp.runtime_introspection.core import (
    RuntimeXRayResult,
    x_ray_runtime_environment,
)
from inference_exp.weights_providers.entities import (
    BackendType,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    Quantization,
    ServerEnvironmentRequirements,
)
from packaging.version import Version


def negotiate_model_packages(
    model_packages: List[ModelPackageMetadata],
    requested_model_package_id: Optional[str] = None,
    requested_backends: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
    requested_batch_size: Optional[Union[int, Tuple[int, int]]] = None,
    requested_quantization: Optional[
        Union[str, Quantization, List[Union[str, Quantization]]]
    ] = None,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    verbose: bool = False,
) -> List[ModelPackageMetadata]:
    if verbose:
        print_model_packages(model_packages=model_packages)
    if not model_packages:
        raise NoModelPackagesAvailableError(
            f"Could not select model package to load among ones announced by weights provider. "
            f"That may indicate that the 'inference' installation lacks additional dependencies or "
            f"the backend does not provide expected model packages."
        )
    if requested_model_package_id is not None:
        return [
            select_model_package_by_id(
                model_packages=model_packages,
                requested_model_package_id=requested_model_package_id,
                verbose=verbose,
            )
        ]
    if requested_backends is not None:
        model_packages = filter_model_packages_by_requested_backend(
            model_packages=model_packages,
            requested_backends=requested_backends,
            verbose=verbose,
        )
    if requested_batch_size is not None:
        model_packages = filter_model_packages_by_requested_batch_size(
            model_packages=model_packages,
            requested_batch_size=requested_batch_size,
            verbose=verbose,
        )
    if requested_quantization is None:
        requested_quantization = determine_default_allowed_quantization(device=device)
    if requested_quantization:
        model_packages = filter_model_packages_by_requested_quantization(
            model_packages=model_packages,
            requested_quantization=requested_quantization,
            verbose=verbose,
        )
    runtime_x_ray = x_ray_runtime_environment()
    if verbose:
        print("Selecting model packages matching to runtime")
        print(runtime_x_ray)
    results = [
        model_package
        for model_package in model_packages
        if model_package_matches_runtime_environment(
            model_package=model_package,
            runtime_x_ray=runtime_x_ray,
            device=device,
            onnx_execution_providers=onnx_execution_providers,
            verbose=verbose,
        )
    ]
    if not results:
        raise NoModelPackagesAvailableError(
            f"Could not select model package to load among ones announced by weights provider comparing with "
            f"detected runtime environment. That may indicate that the 'inference' installation lacks additional "
            f"dependencies or the model is not registered with packages that would allow `inference` to run."
        )
    results = rank_model_packages(model_packages=results)
    if verbose:
        print("Eligible packages ranked:")
        print_model_packages(model_packages=results)
    return results


@cache
def determine_default_allowed_quantization(
    device: Optional[torch.device] = None,
) -> List[Quantization]:
    if device is not None:
        if device.type == "cpu":
            return [
                Quantization.UNKNOWN,
                Quantization.FP32,
                Quantization.BF16,
            ]
        return [
            Quantization.UNKNOWN,
            Quantization.FP32,
            Quantization.FP16,
        ]
    runtime_x_ray = x_ray_runtime_environment()
    if runtime_x_ray.gpu_devices:
        return [
            Quantization.UNKNOWN,
            Quantization.FP32,
            Quantization.FP16,
        ]
    return [
        Quantization.UNKNOWN,
        Quantization.FP32,
        Quantization.BF16,
    ]


def print_model_packages(model_packages: List[ModelPackageMetadata]) -> None:
    print("The following model packages were exposed by weights provider:")
    for i, model_package in enumerate(model_packages):
        print(f"{i+1}. {model_package.get_summary()}")
    if not model_packages:
        print("No model packages!")


def select_model_package_by_id(
    model_packages: List[ModelPackageMetadata],
    requested_model_package_id: str,
    verbose: bool = False,
) -> ModelPackageMetadata:
    matching_packages = [
        p for p in model_packages if p.package_id == requested_model_package_id
    ]
    if not matching_packages:
        raise NoModelPackagesAvailableError(
            f"Requested model package ID: {requested_model_package_id} cannot be resolved among "
            f"the model packages announced by weights provider. This may indicate either "
            f"typo on the identifier or a change in set of models packages being announced by provider."
        )
    if len(matching_packages) > 1:
        raise AmbiguousModelPackageResolutionError(
            f"Requested model package ID: {requested_model_package_id} resolved to {len(matching_packages)} "
            f"different packages announced by weights provider. That is most likely weights provider "
            f"error, as it is supposed to provide unique identifiers for each model package."
        )
    if verbose:
        print(f"Model package matching requested package id: {matching_packages[0]}")
    return matching_packages[0]


def filter_model_packages_by_requested_backend(
    model_packages: List[ModelPackageMetadata],
    requested_backends: Union[str, BackendType, List[Union[str, BackendType]]],
    verbose: bool = False,
) -> List[ModelPackageMetadata]:
    if not isinstance(requested_backends, list):
        requested_backends = [requested_backends]
    requested_backends_set = set()
    for requested_backend in requested_backends:
        if isinstance(requested_backend, str):
            requested_backend = _parse_backend_type(value=requested_backend)
        requested_backends_set.add(requested_backend)
    if verbose:
        print(f"Filtering model packages by requested backends: {requested_backends}")
    filtered_packages = []
    for model_package in model_packages:
        if model_package.backend not in requested_backends_set:
            continue
        if verbose:
            print(
                f"Model package with id `{model_package.package_id}` matches requested backends."
            )
        filtered_packages.append(model_package)
    if not filtered_packages:
        raise NoModelPackagesAvailableError(
            f"Could not find model packages that match the criteria of requested backends: {requested_backends_set}. "
            f"This error is caused by to strict requirements of model packages backends compared to what weights "
            f"provider announce for selected model."
        )
    return filtered_packages


def filter_model_packages_by_requested_batch_size(
    model_packages: List[ModelPackageMetadata],
    requested_batch_size: Union[int, Tuple[int, int]],
    verbose: bool = False,
) -> List[ModelPackageMetadata]:
    min_batch_size, max_batch_size = _parse_batch_size(
        requested_batch_size=requested_batch_size
    )
    if verbose:
        print(
            f"Filtering model packages by supported batch sizes min={min_batch_size} max={max_batch_size}"
        )
    filtered_packages = []
    for model_package in model_packages:
        if model_package_matches_batch_size_request(
            model_package=model_package,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            verbose=verbose,
        ):
            filtered_packages.append(model_package)
    if not filtered_packages:
        raise NoModelPackagesAvailableError(
            f"Could not find model packages that match the criteria of requested batch size: "
            f"[{min_batch_size}, {max_batch_size}]. This error is caused by to strict requirements of "
            f"supported batch size compared to what weights provider announce for selected model."
        )
    return filtered_packages


def filter_model_packages_by_requested_quantization(
    model_packages: List[ModelPackageMetadata],
    requested_quantization: Union[str, Quantization, List[Union[str, Quantization]]],
    verbose: bool = False,
) -> List[ModelPackageMetadata]:
    requested_quantization = _parse_requested_quantization(value=requested_quantization)
    if verbose:
        print(
            f"Filtering model packages by quantization - allowed values: {requested_quantization}"
        )
    filtered_packages = []
    for model_package in model_packages:
        if model_package.quantization in requested_quantization:
            if verbose:
                print(
                    f"Model package with id `{model_package.package_id}` matches requested quantization."
                )
            filtered_packages.append(model_package)
    if not filtered_packages:
        raise NoModelPackagesAvailableError(
            f"Could not find model packages that match the criteria of quantization: "
            f"{requested_quantization}. This error is caused by to strict requirements of "
            f"supported quantization compared to what weights provider announce for selected model."
        )
    return filtered_packages


def model_package_matches_batch_size_request(
    model_package: ModelPackageMetadata,
    min_batch_size: int,
    max_batch_size: int,
    verbose: bool = False,
) -> bool:
    if model_package.dynamic_batch_size_supported:
        if model_package.specifies_dynamic_batch_boundaries():
            declared_min_batch_size, declared_max_batch_size = (
                model_package.get_dynamic_batch_boundaries()
            )
            ranges_match = _range_within_other(
                external_range=(declared_min_batch_size, declared_max_batch_size),
                internal_range=(min_batch_size, max_batch_size),
            )
            if verbose:
                match_str = (
                    "matches criteria" if ranges_match else "does not match criteria"
                )
                print(
                    f"Model package with id `{model_package.package_id}` declared to support dynamic batch sizes: "
                    f"[{declared_min_batch_size}, {declared_max_batch_size}] and requested batch size was: "
                    f"[{min_batch_size}, {max_batch_size}] - package {match_str}."
                )
            return ranges_match
        if verbose:
            print(
                f"Model package with id `{model_package.package_id}` supports dynamic batches without "
                f"specifying bounds - including into results."
            )
        return True
    else:
        return min_batch_size <= model_package.static_batch_size <= max_batch_size


def model_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    verbose: bool = False,
) -> bool:
    if model_package.backend not in MODEL_TO_RUNTIME_COMPATIBILITY_MATCHERS:
        raise ModelPackageNegotiationError(
            f"Model package negotiation protocol not implemented for model backend {model_package.backend}. "
            f"This is `inference` bug - raise issue: https://github.com/roboflow/inference/issues"
        )
    return MODEL_TO_RUNTIME_COMPATIBILITY_MATCHERS[model_package.backend](
        model_package, runtime_x_ray, device, onnx_execution_providers, verbose
    )


ONNX_RUNTIME_OPSET_COMPATIBILITY = {
    Version("1.15"): 19,
    Version("1.16"): 19,
    Version("1.17"): 20,
    Version("1.18"): 21,
    Version("1.19"): 21,
    Version("1.20"): 21,
    Version("1.21"): 22,
    Version("1.22"): 23,
}


def onnx_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    verbose: bool = False,
) -> bool:
    if verbose and not runtime_x_ray.onnxruntime_version or not runtime_x_ray.available_onnx_execution_providers:
        print(
            f"Mode package with id '{model_package.package_id}' filtered out as onnxruntime not detected"
        )
    if not runtime_x_ray.onnxruntime_version or not runtime_x_ray.available_onnx_execution_providers:
        return False
    if model_package.onnx_package_details is None:
        # no restrictions raised by the backend
        return True
    if not onnx_execution_providers:
        onnx_execution_providers = ONNXRUNTIME_EXECUTION_PROVIDERS
    onnx_execution_providers = [
        provider for provider in onnx_execution_providers
        if provider in runtime_x_ray.available_onnx_execution_providers
    ]
    if not onnx_execution_providers:
        # no actual providers capable of running the model
        return False
    incompatible_providers = model_package.onnx_package_details.incompatible_providers
    if incompatible_providers is None:
        incompatible_providers = []
    incompatible_providers = set(incompatible_providers)
    if onnx_execution_providers[0] in incompatible_providers:
        # checking the first one only - this is kind of heuristic, as
        # probably there may be a fallback - so theoretically it is possible
        # for this function to claim that package is compatible, but specific
        # operation in the graph may fall back to another EP - but that's
        # rather a situation deeply specific for a model and if we see this be
        # problematic, we will implement solution - so far - to counter-act errors
        # which can only be determined in runtime we may either expect model implementation
        # would run test inference in init or user to define specific model package ID
        # to run. Not great, not terrible, yet I can expect this to be a basis of heated
        # debate some time in the future :)
        if verbose:
            print(
                f"Mode package with id '{model_package.package_id}' filtered out as execution provider "
                f"which is selected as primary one ('{onnx_execution_providers[0]}') is enlisted as incompatible "
                f"for model package."
            )
        return False
    package_opset = model_package.onnx_package_details.opset
    onnx_runtime_simple_version = Version(
        f"{runtime_x_ray.onnxruntime_version.major}.{runtime_x_ray.onnxruntime_version.minor}"
    )
    if onnx_runtime_simple_version not in ONNX_RUNTIME_OPSET_COMPATIBILITY:
        return package_opset <= ONNX_RUNTIME_OPSET_COMPATIBILITY[Version("1.15")]
    max_supported_opset = ONNX_RUNTIME_OPSET_COMPATIBILITY[onnx_runtime_simple_version]
    return package_opset <= max_supported_opset


def torch_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    verbose: bool = False,
) -> bool:
    if verbose and not runtime_x_ray.torch_available:
        print(
            f"Mode package with id '{model_package.package_id}' filtered out as torch not detected"
        )
    return runtime_x_ray.torch_available


def hf_transformers_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    verbose: bool = False,
) -> bool:
    if verbose and not runtime_x_ray.hf_transformers_available:
        print(
            f"Mode package with id '{model_package.package_id}' filtered out as transformers not detected"
        )
    return runtime_x_ray.hf_transformers_available


def ultralytics_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    verbose: bool = False,
) -> bool:
    if verbose and not runtime_x_ray.ultralytics_available:
        print(
            f"Mode package with id '{model_package.package_id}' filtered out as ultralytics not detected"
        )
    return runtime_x_ray.ultralytics_available


def trt_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    verbose: bool = False,
) -> bool:
    if not runtime_x_ray.trt_version:
        if verbose:
            print(
                f"Mode package with id '{model_package.package_id}' filtered out as TRT libraries not detected"
            )
        return False
    if not runtime_x_ray.trt_python_package_available:
        if verbose:
            print(
                f"Mode package with id '{model_package.package_id}' filtered out as TRT python package not available"
            )
        return False
    if not runtime_x_ray.cuda_version:
        if verbose:
            print(
                f"Mode package with id '{model_package.package_id}' filtered out as cuda libraries not detected"
            )
        return False
    if model_package.environment_requirements is None:
        if verbose:
            print(
                f"Mode package with id '{model_package.package_id}' filtered out as environment requirements "
                f"not provided by backend."
            )
        return False
    trt_compiled_with_cc_compatibility = False
    if model_package.trt_package_details is not None:
        trt_compiled_with_cc_compatibility = (
            model_package.trt_package_details.same_cc_compatible
        )
    trt_forward_compatible = False
    if model_package.trt_package_details is not None:
        trt_forward_compatible = model_package.trt_package_details.trt_forward_compatible
    trt_lean_runtime_excluded = False
    if model_package.trt_package_details is not None:
        trt_lean_runtime_excluded = model_package.trt_package_details.trt_lean_runtime_excluded
    model_environment = model_package.environment_requirements
    if isinstance(model_environment, JetsonEnvironmentRequirements):
        if model_environment.trt_version is None:
            if verbose:
                print(
                    f"Mode package with id '{model_package.package_id}' filtered out as model TRT version not provided by backend"
                )
            return False
        device_compatibility = verify_trt_package_compatibility_with_cuda_device(
            all_available_cuda_devices=runtime_x_ray.gpu_devices,
            all_available_devices_cc=runtime_x_ray.gpu_devices_cc,
            compilation_device=model_environment.cuda_device_name,
            compilation_device_cc=model_environment.cuda_device_cc,
            selected_device=device,
            trt_compiled_with_cc_compatibility=trt_compiled_with_cc_compatibility,
        )
        if not device_compatibility:
            if verbose:
                print(
                    f"Model package with id '{model_package.package_id}' filtered out due to device incompatibility."
                )
            return False
        if verify_versions_up_to_major_and_minor(
            runtime_x_ray.l4t_version, model_environment.l4t_version
        ):
            if verbose:
                print(
                    f"Mode package with id '{model_package.package_id}' filtered out as package L4T {model_environment.l4t_version} does not match runtime L4T: {runtime_x_ray.l4t_version}"
                )
            return False
        if trt_forward_compatible:
            if not verify_version_larger_equal_up_to_major_and_minor(runtime_x_ray.trt_version, model_environment.trt_version):
                return False
            if trt_lean_runtime_excluded:
                # not supported for now
                return False
        elif not verify_versions_up_to_major_minor_and_micro(
            runtime_x_ray.trt_version, model_environment.trt_version
        ):
            if verbose:
                print(
                    f"Mode package with id '{model_package.package_id}' filtered out as package trt version {model_environment.trt_version} does not match runtime trt version: {runtime_x_ray.trt_version}"
                )
            return False
        if not verify_version_larger_equal_up_to_major_and_minor(
            runtime_x_ray.cuda_version, model_environment.cuda_version
        ):
            if verbose:
                print(
                    f"Mode package with id '{model_package.package_id}' filtered out as package cuda version {model_environment.cuda_version} does not match runtime cuda version: {runtime_x_ray.cuda_version}"
                )
            return False
        return True
    if not isinstance(model_environment, ServerEnvironmentRequirements):
        raise ModelPackageNegotiationError(
            f"Model package negotiation protocol not implemented for environment specification detected "
            f"in runtime. This is `inference` bug - raise issue: https://github.com/roboflow/inference/issues"
        )
    if model_environment.trt_version is None:
        if verbose:
            print(
                f"Mode package with id '{model_package.package_id}' filtered out as model TRT version not provided by backend"
            )
        return False
    device_compatibility = verify_trt_package_compatibility_with_cuda_device(
        all_available_cuda_devices=runtime_x_ray.gpu_devices,
        all_available_devices_cc=runtime_x_ray.gpu_devices_cc,
        compilation_device=model_environment.cuda_device_name,
        compilation_device_cc=model_environment.cuda_device_cc,
        selected_device=device,
        trt_compiled_with_cc_compatibility=trt_compiled_with_cc_compatibility,
    )
    if not device_compatibility:
        if verbose:
            print(
                f"Model package with id '{model_package.package_id}' filtered out due to device incompatibility."
            )
        return False
    if trt_forward_compatible:
        if not verify_version_larger_equal_up_to_major_and_minor(runtime_x_ray.trt_version, model_environment.trt_version):
            return False
        if trt_lean_runtime_excluded:
            # not supported for now
            return False
    elif not verify_versions_up_to_major_minor_and_micro(
        runtime_x_ray.trt_version, model_environment.trt_version
    ):
        if verbose:
            print(
                f"Mode package with id '{model_package.package_id}' filtered out as package trt version {model_environment.trt_version} does not match runtime trt version: {runtime_x_ray.trt_version}"
            )
        return False
    if not verify_version_larger_equal_up_to_major_and_minor(
        runtime_x_ray.cuda_version, model_environment.cuda_version
    ):
        if verbose:
            print(
                f"Mode package with id '{model_package.package_id}' filtered out as package cuda version {model_environment.cuda_version} does not match runtime cuda version: {runtime_x_ray.cuda_version}"
            )
        return False
    return True


def verify_trt_package_compatibility_with_cuda_device(
    selected_device: Optional[torch.device],
    all_available_cuda_devices: List[str],
    all_available_devices_cc: List[Version],
    compilation_device: str,
    compilation_device_cc: Version,
    trt_compiled_with_cc_compatibility: bool,
) -> bool:
    all_available_cuda_devices, all_available_devices_cc = (
        filter_available_devices_with_selected_device(
            selected_device=selected_device,
            all_available_cuda_devices=all_available_cuda_devices,
            all_available_devices_cc=all_available_devices_cc,
        )
    )
    if trt_compiled_with_cc_compatibility:
        return any(cc == compilation_device_cc for cc in all_available_devices_cc)
    return any(dev == compilation_device for dev in all_available_cuda_devices)


def verify_versions_up_to_major_and_minor(x: Version, y: Version) -> bool:
    x_simplified = Version(f"{x.major}.{x.minor}")
    y_simplified = Version(f"{y.major}.{y.minor}")
    return x_simplified == y_simplified


def verify_versions_up_to_major_minor_and_micro(x: Version, y: Version) -> bool:
    x_simplified = Version(f"{x.major}.{x.minor}.{x.micro}")
    y_simplified = Version(f"{y.major}.{y.minor}.{y.micro}")
    return x_simplified == y_simplified


def verify_version_larger_equal_up_to_major_and_minor(x: Version, y: Version) -> bool:
    x_simplified = Version(f"{x.major}.{x.minor}")
    y_simplified = Version(f"{y.major}.{y.minor}")
    return x_simplified >= y_simplified


MODEL_TO_RUNTIME_COMPATIBILITY_MATCHERS = {
    BackendType.HF: hf_transformers_package_matches_runtime_environment,
    BackendType.TRT: trt_package_matches_runtime_environment,
    BackendType.ONNX: onnx_package_matches_runtime_environment,
    BackendType.TORCH: torch_package_matches_runtime_environment,
    BackendType.ULTRALYTICS: ultralytics_package_matches_runtime_environment,
}


def _range_within_other(
    external_range: Tuple[int, int],
    internal_range: Tuple[int, int],
) -> bool:
    external_min, external_max = external_range
    internal_min, internal_max = internal_range
    return external_min <= internal_min <= internal_max <= external_max


def _parse_batch_size(
    requested_batch_size: Union[int, Tuple[int, int]]
) -> Tuple[int, int]:
    if isinstance(requested_batch_size, tuple):
        if len(requested_batch_size) != 2:
            raise InvalidRequestedBatchSizeError(
                "Could not parse batch size requested from model package negotiation procedure. "
                "Batch size request is supposed to be either integer value or tuple specifying (min, max) "
                f"batch size - but detected tuple of invalid size ({len(requested_batch_size)}) - this is "
                f"probably typo while specifying requested batch size."
            )
        min_batch_size, max_batch_size = requested_batch_size
        if not isinstance(min_batch_size, int) or not isinstance(max_batch_size, int):
            raise InvalidRequestedBatchSizeError(
                "Could not parse batch size requested from model package negotiation procedure. "
                "Batch size request is supposed to be either integer value or tuple specifying (min, max) "
                f"batch size - but detected tuple elements which are not integer values - this is "
                f"probably typo while specifying requested batch size."
            )
        return min_batch_size, max_batch_size
    if not isinstance(requested_batch_size, int):
        raise InvalidRequestedBatchSizeError(
            "Could not parse batch size requested from model package negotiation procedure. "
            "Batch size request is supposed to be either integer value or tuple specifying (min, max) "
            f"batch size - but detected single value which is not integer but has type "
            f"{requested_batch_size.__class__.__name__} - this is "
            f"probably typo while specifying requested batch size."
        )
    return requested_batch_size, requested_batch_size


def _parse_backend_type(value: str) -> BackendType:
    try:
        return BackendType(value)
    except ValueError as error:
        raise UnknownBackendTypeError(
            f"Requested backend of type '{value}' which is not recognized by `inference`. Most likely this "
            f"error is a result of typo while specifying requested backend. Supported backends: "
            f"{list(BackendType.__members__)}."
        ) from error


def _parse_requested_quantization(
    value: Union[str, Quantization, List[Union[str, Quantization]]]
) -> Set[Quantization]:
    if not isinstance(value, list):
        value = [value]
    result = set()
    for element in value:
        if isinstance(element, str):
            element = _parse_quantization(value=element)
        result.add(element)
    return result


def _parse_quantization(value: str) -> Quantization:
    try:
        return Quantization(value)
    except ValueError as error:
        raise UnknownQuantizationError(
            f"Requested quantization of type '{value}' which is not recognized by `inference`. Most likely this "
            f"error is a result of typo while specifying requested quantization. Supported values: "
            f"{list(Quantization.__members__)}."
        ) from error
