from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import List, Optional, Set, Tuple, Union

import torch
from inference_exp.errors import (
    AmbiguousModelPackageResolutionError,
    InvalidRequestedBatchSizeError,
    ModelPackageNegotiationError,
    NoModelPackagesAvailableError,
    UnknownBackendTypeError,
    UnknownQuantizationError,
)
from inference_exp.logger import verbose_info
from inference_exp.models.auto_loaders.entities import ModelArchitecture, TaskType
from inference_exp.models.auto_loaders.models_registry import (
    model_implementation_exists,
)
from inference_exp.models.auto_loaders.ranking import rank_model_packages
from inference_exp.models.auto_loaders.utils import (
    filter_available_devices_with_selected_device,
)
from inference_exp.runtime_introspection.core import (
    RuntimeXRayResult,
    x_ray_runtime_environment,
)
from inference_exp.utils.onnx_introspection import get_selected_onnx_execution_providers
from inference_exp.weights_providers.entities import (
    BackendType,
    JetsonEnvironmentRequirements,
    ModelPackageMetadata,
    Quantization,
    ServerEnvironmentRequirements,
)
from packaging.version import Version


@dataclass(frozen=True)
class DiscardedPackage:
    package_id: str
    reason: str


def negotiate_model_packages(
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
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
    allow_untrusted_packages: bool = False,
    trt_engine_host_code_allowed: bool = True,
    verbose: bool = False,
) -> List[ModelPackageMetadata]:
    verbose_info(
        "The following model packages were exposed by weights provider:",
        verbose_requested=verbose,
    )
    print_model_packages(model_packages=model_packages, verbose=verbose)
    if not model_packages:
        raise NoModelPackagesAvailableError(
            message=f"Could not find any model package announced by weights provider. If you see this error "
            f"using Roboflow platform your model may not be ready - if the problem is persistent, "
            f"contact us to get help. If you use weights provider other than Roboflow - this is most likely "
            f"the root cause of the error.",
            help_url="https://todo",
        )
    if requested_model_package_id is not None:
        return [
            select_model_package_by_id(
                model_packages=model_packages,
                requested_model_package_id=requested_model_package_id,
                verbose=verbose,
            )
        ]
    model_packages, discarded_packages = remove_packages_not_matching_implementation(
        model_architecture=model_architecture,
        task_type=task_type,
        model_packages=model_packages,
    )
    if not allow_untrusted_packages:
        model_packages, discarded_untrusted_packages = remove_untrusted_packages(
            model_packages=model_packages,
            verbose=verbose,
        )
        discarded_packages.extend(discarded_untrusted_packages)
    if requested_backends is not None:
        model_packages, discarded_by_not_matching_backend = (
            filter_model_packages_by_requested_backend(
                model_packages=model_packages,
                requested_backends=requested_backends,
                verbose=verbose,
            )
        )
        discarded_packages.extend(discarded_by_not_matching_backend)
    if requested_batch_size is not None:
        model_packages, discarded_by_batch_size = (
            filter_model_packages_by_requested_batch_size(
                model_packages=model_packages,
                requested_batch_size=requested_batch_size,
                verbose=verbose,
            )
        )
        discarded_packages.extend(discarded_by_batch_size)
    default_quantization = False
    if requested_quantization is None:
        default_quantization = True
        requested_quantization = determine_default_allowed_quantization(device=device)
    if requested_quantization:
        model_packages, discarded_by_quantization = (
            filter_model_packages_by_requested_quantization(
                model_packages=model_packages,
                requested_quantization=requested_quantization,
                default_quantization_used=default_quantization,
                verbose=verbose,
            )
        )
        discarded_packages.extend(discarded_by_quantization)
    model_packages, discarded_by_env_matching = (
        filter_model_packages_matching_runtime_environment(
            model_packages=model_packages,
            device=device,
            onnx_execution_providers=onnx_execution_providers,
            trt_engine_host_code_allowed=trt_engine_host_code_allowed,
            verbose=verbose,
        )
    )
    discarded_packages.extend(discarded_by_env_matching)
    if not model_packages:
        rejections_summary = summarise_discarded_packages(
            discarded_packages=discarded_packages
        )
        raise NoModelPackagesAvailableError(
            message=f"Auto-negotiation protocol could not select model packages. This situation may be caused by "
            f"several reasons, with the most common being missing dependencies or too strict requirements "
            f"stated as parameters of loading function. Below you can find reasons why specific model "
            f"packages were rejected:\n{rejections_summary}\n",
            help_url="https://todo",
        )
    model_packages = rank_model_packages(
        model_packages=model_packages, selected_device=device
    )
    verbose_info("Eligible packages ranked:", verbose_requested=verbose)
    print_model_packages(model_packages=model_packages, verbose=verbose)
    return model_packages


def summarise_discarded_packages(discarded_packages: List[DiscardedPackage]) -> str:
    reasons_and_counts = Counter()
    for package in discarded_packages:
        reasons_and_counts[package.reason] += 1
    reasons_stats = reasons_and_counts.most_common()
    result = []
    for reason, count in reasons_stats:
        package_str = "package" if count < 2 else "packages"
        result.append(f"\t* {count} {package_str} with the following note: {reason}")
    return "\n".join(result)


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


def print_model_packages(
    model_packages: List[ModelPackageMetadata], verbose: bool
) -> None:
    if not model_packages:
        verbose_info(message="No model packages.", verbose_requested=verbose)
        return None
    contents = []
    for i, model_package in enumerate(model_packages):
        contents.append(f"{i+1}. {model_package.get_summary()}")
    verbose_info(message="\n".join(contents), verbose_requested=verbose)


def remove_packages_not_matching_implementation(
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    model_packages: List[ModelPackageMetadata],
    verbose: bool = False,
) -> Tuple[List[ModelPackageMetadata], List[DiscardedPackage]]:
    result, discarded = [], []
    for model_package in model_packages:
        if not model_implementation_exists(
            model_architecture=model_architecture,
            task_type=task_type,
            backend=model_package.backend,
        ):
            verbose_info(
                message=f"Model package with id `{model_package.package_id}` is filtered out as `inference-exp` "
                f"does not provide implementation for the model architecture {model_architecture} with "
                f"task type: {task_type} and backend {model_package.backend}.",
                verbose_requested=verbose,
            )
            discarded.append(
                DiscardedPackage(
                    package_id=model_package.package_id,
                    reason=f"`inference-exp` does not provide implementation for the model {model_architecture} "
                    f"({task_type}) with backend {model_package.backend.value}",
                )
            )
            continue
        result.append(model_package)
    return result, discarded


def remove_untrusted_packages(
    model_packages: List[ModelPackageMetadata],
    verbose: bool = False,
) -> Tuple[List[ModelPackageMetadata], List[DiscardedPackage]]:
    result, discarded_packages = [], []
    for model_package in model_packages:
        if not model_package.trusted_source:
            verbose_info(
                message=f"Model package with id `{model_package.package_id}` is filtered out as come from "
                f"untrusted source.",
                verbose_requested=verbose,
            )
            discarded_packages.append(
                DiscardedPackage(
                    package_id=model_package.package_id,
                    reason="Package is marked as `untrusted` and auto-loader was used with "
                    "`allow_untrusted_packages=False`",
                )
            )
            continue
        result.append(model_package)
    return result, discarded_packages


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
            message=f"Requested model package ID: {requested_model_package_id} cannot be resolved among "
            f"the model packages announced by weights provider. This may indicate either "
            f"typo on the identifier or a change in set of models packages being announced by provider.",
            help_url="https://todo",
        )
    if len(matching_packages) > 1:
        raise AmbiguousModelPackageResolutionError(
            message=f"Requested model package ID: {requested_model_package_id} resolved to {len(matching_packages)} "
            f"different packages announced by weights provider. That is most likely weights provider "
            f"error, as it is supposed to provide unique identifiers for each model package.",
            help_url="https://todo",
        )
    verbose_info(
        message=f"Model package matching requested package id: {matching_packages[0].get_summary()}",
        verbose_requested=verbose,
    )
    return matching_packages[0]


def filter_model_packages_by_requested_backend(
    model_packages: List[ModelPackageMetadata],
    requested_backends: Union[str, BackendType, List[Union[str, BackendType]]],
    verbose: bool = False,
) -> Tuple[List[ModelPackageMetadata], List[DiscardedPackage]]:
    if not isinstance(requested_backends, list):
        requested_backends = [requested_backends]
    requested_backends_set = set()
    for requested_backend in requested_backends:
        if isinstance(requested_backend, str):
            requested_backend = parse_backend_type(value=requested_backend)
        requested_backends_set.add(requested_backend)
    verbose_info(
        message=f"Filtering model packages by requested backends: {requested_backends_set}",
        verbose_requested=verbose,
    )
    requested_backends_serialised = [b.value for b in requested_backends_set]
    filtered_packages, discarded_packages = [], []
    for model_package in model_packages:
        if model_package.backend not in requested_backends_set:
            verbose_info(
                message=f"Model package with id `{model_package.package_id}` does not match requested backends.",
                verbose_requested=verbose,
            )
            discarded_packages.append(
                DiscardedPackage(
                    package_id=model_package.package_id,
                    reason=f"Package backend {model_package.backend.value} does not match requested backends: "
                    f"{requested_backends_serialised}",
                )
            )
            continue
        filtered_packages.append(model_package)
    return filtered_packages, discarded_packages


def filter_model_packages_by_requested_batch_size(
    model_packages: List[ModelPackageMetadata],
    requested_batch_size: Union[int, Tuple[int, int]],
    verbose: bool = False,
) -> Tuple[List[ModelPackageMetadata], List[DiscardedPackage]]:
    min_batch_size, max_batch_size = parse_batch_size(
        requested_batch_size=requested_batch_size
    )
    verbose_info(
        message=f"Filtering model packages by supported batch sizes min={min_batch_size} max={max_batch_size}",
        verbose_requested=verbose,
    )
    filtered_packages, discarded_packages = [], []
    for model_package in model_packages:
        if not model_package_matches_batch_size_request(
            model_package=model_package,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            verbose=verbose,
        ):
            verbose_info(
                message=f"Model package with id `{model_package.package_id}` does not match requested batch "
                f"size <{min_batch_size}, {max_batch_size}>.",
                verbose_requested=verbose,
            )
            discarded_packages.append(
                DiscardedPackage(
                    package_id=model_package.package_id,
                    reason=f"Package batch size does not match requested batch size <{min_batch_size}, {max_batch_size}>",
                )
            )
            continue
        filtered_packages.append(model_package)
    return filtered_packages, discarded_packages


def filter_model_packages_by_requested_quantization(
    model_packages: List[ModelPackageMetadata],
    requested_quantization: Union[str, Quantization, List[Union[str, Quantization]]],
    default_quantization_used: bool,
    verbose: bool = False,
) -> Tuple[List[ModelPackageMetadata], List[DiscardedPackage]]:
    requested_quantization = parse_requested_quantization(value=requested_quantization)
    requested_quantization_str = [e.value for e in requested_quantization]
    verbose_info(
        message=f"Filtering model packages by quantization - allowed values: {requested_quantization_str}",
        verbose_requested=verbose,
    )
    default_quantization_used_str = (
        " (which was selected by default)."
        if default_quantization_used
        else " (which was selected by caller)."
    )
    filtered_packages, discarded_packages = [], []
    for model_package in model_packages:
        if model_package.quantization not in requested_quantization:
            verbose_info(
                message=f"Model package with id `{model_package.package_id}` does not match requested quantization "
                f"{requested_quantization_str}{default_quantization_used_str}",
                verbose_requested=verbose,
            )
            discarded_packages.append(
                DiscardedPackage(
                    package_id=model_package.package_id,
                    reason=f"Package does not match requested quantization {requested_quantization_str}"
                    f"{default_quantization_used_str}",
                )
            )
            continue
        filtered_packages.append(model_package)
    return filtered_packages, discarded_packages


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
            ranges_match = range_within_other(
                external_range=(declared_min_batch_size, declared_max_batch_size),
                internal_range=(min_batch_size, max_batch_size),
            )
            if not ranges_match:
                verbose_info(
                    message=f"Model package with id `{model_package.package_id}` declared to support dynamic batch sizes: "
                    f"[{declared_min_batch_size}, {declared_max_batch_size}] and requested batch size was: "
                    f"[{min_batch_size}, {max_batch_size}] - package does not match criteria.",
                    verbose_requested=verbose,
                )
            return ranges_match
        return True
    if min_batch_size <= model_package.static_batch_size <= max_batch_size:
        return True
    verbose_info(
        message=f"Model package with id `{model_package.package_id}` filtered out, as static batch size does not "
        f"match requested values: ({min_batch_size}, {max_batch_size})."
        f"If you see this error on Roboflow platform - contact us to get help. "
        f"Otherwise, consider adjusting requested batch size.",
        verbose_requested=verbose,
    )
    return False


def filter_model_packages_matching_runtime_environment(
    model_packages: List[ModelPackageMetadata],
    device: torch.device,
    onnx_execution_providers: Optional[List[Union[str, tuple]]],
    trt_engine_host_code_allowed: bool,
    verbose: bool = False,
) -> Tuple[List[ModelPackageMetadata], List[DiscardedPackage]]:
    runtime_x_ray = x_ray_runtime_environment()
    verbose_info(
        message=f"Selecting model packages matching to runtime: {runtime_x_ray}",
        verbose_requested=verbose,
    )
    results, discarded_packages = [], []
    for model_package in model_packages:
        matches, reason = model_package_matches_runtime_environment(
            model_package=model_package,
            runtime_x_ray=runtime_x_ray,
            device=device,
            onnx_execution_providers=onnx_execution_providers,
            trt_engine_host_code_allowed=trt_engine_host_code_allowed,
            verbose=verbose,
        )
        if not matches:
            discarded_packages.append(
                DiscardedPackage(
                    package_id=model_package.package_id,
                    reason=reason,
                )
            )
            continue
        results.append(model_package)
    return results, discarded_packages


def model_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    trt_engine_host_code_allowed: bool = True,
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    if model_package.backend not in MODEL_TO_RUNTIME_COMPATIBILITY_MATCHERS:
        raise ModelPackageNegotiationError(
            message=f"Model package negotiation protocol not implemented for model backend {model_package.backend}. "
            f"This is `inference-exp` bug - raise issue: https://github.com/roboflow/inference/issues",
            help_url="https://todo",
        )
    return MODEL_TO_RUNTIME_COMPATIBILITY_MATCHERS[model_package.backend](
        model_package,
        runtime_x_ray,
        device,
        onnx_execution_providers,
        trt_engine_host_code_allowed,
        verbose,
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
    trt_engine_host_code_allowed: bool = True,
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    if (
        not runtime_x_ray.onnxruntime_version
        or not runtime_x_ray.available_onnx_execution_providers
    ):
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as onnxruntime not detected",
            verbose_requested=verbose,
        )
        return False, (
            "ONNX backend not installed - consider installing relevant ONNX extras: "
            "`onnx-cpu`, `onnx-cu118`, `onnx-cu12`, `onnx-jp6-cu126` depending on hardware you run `inference-exp`"
        )
    if model_package.onnx_package_details is None:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as onnxruntime specification "
            f"not provided by weights provider.",
            verbose_requested=verbose,
        )
        return (
            False,
            "Model package metadata delivered by weights provider lack required ONNX package details",
        )
    providers_auto_selected = False
    if not onnx_execution_providers:
        providers_auto_selected = True
        onnx_execution_providers = get_selected_onnx_execution_providers()
    onnx_execution_providers = [
        provider
        for provider in onnx_execution_providers
        if provider in runtime_x_ray.available_onnx_execution_providers
    ]
    if not onnx_execution_providers:
        if providers_auto_selected:
            reason = (
                "Incorrect ONNX backend installation none of the default ONNX Execution Providers "
                "available in environment"
            )
        else:
            reason = (
                "None of the selected ONNX Execution Providers detected in runtime environment - consider "
                "adjusting the settings"
            )
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as `inference-exp` could not find "
            f"matching execution providers that are available in runtime to run a model.",
            verbose_requested=verbose,
        )
        return False, reason
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
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as execution provider "
            f"which is selected as primary one ('{onnx_execution_providers[0]}') is enlisted as incompatible "
            f"for model package.",
            verbose_requested=verbose,
        )
        return (
            False,
            f"Model package cannot be run with default ONNX Execution Provider: {onnx_execution_providers[0]}",
        )
    package_opset = model_package.onnx_package_details.opset
    onnx_runtime_simple_version = Version(
        f"{runtime_x_ray.onnxruntime_version.major}.{runtime_x_ray.onnxruntime_version.minor}"
    )
    if onnx_runtime_simple_version not in ONNX_RUNTIME_OPSET_COMPATIBILITY:
        if package_opset <= ONNX_RUNTIME_OPSET_COMPATIBILITY[Version("1.15")]:
            return True, None
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as onnxruntime version "
            f"detected ({runtime_x_ray.onnxruntime_version}) could not be resolved with the matching "
            f"onnx opset. The auto-negotiation assumes that in such case, maximum supported opset is 19.",
            verbose_requested=verbose,
        )
        return (
            False,
            "ONNX model package was compiled with opset higher than supported for installed ONNX backend",
        )
    max_supported_opset = ONNX_RUNTIME_OPSET_COMPATIBILITY[onnx_runtime_simple_version]
    if package_opset > max_supported_opset:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as onnxruntime version "
            f"detected ({runtime_x_ray.onnxruntime_version}) can only run onnx packages with opset "
            f"up to {max_supported_opset}, but the package opset is {package_opset}.",
            verbose_requested=verbose,
        )
        return (
            False,
            "ONNX model package was compiled with opset higher than supported for installed ONNX backend",
        )
    return True, None


def torch_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    trt_engine_host_code_allowed: bool = True,
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    if not runtime_x_ray.torch_available:
        verbose_info(
            message="Mode package with id '{model_package.package_id}' filtered out as torch not detected",
            verbose_requested=verbose,
        )
        return (
            False,
            "Torch backend not installed - consider installing relevant torch extras: "
            "`torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128` or `torch-jp6-cu126` \
            depending on hardware you run `inference-exp`",
        )
    return True, None


def hf_transformers_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    trt_engine_host_code_allowed: bool = True,
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    if not runtime_x_ray.hf_transformers_available:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as transformers not detected",
            verbose_requested=verbose,
        )
        return False, (
            "Transformers backend not installed - this package should be installed by default and probably "
            "was accidentally deleted - install `inference-exp` package again."
        )
    return True, None


def ultralytics_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    trt_engine_host_code_allowed: bool = True,
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    if not runtime_x_ray.ultralytics_available:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as ultralytics not detected",
            verbose_requested=verbose,
        )
        return False, "Ultralytics backend not installed"
    return True, None


def trt_package_matches_runtime_environment(
    model_package: ModelPackageMetadata,
    runtime_x_ray: RuntimeXRayResult,
    device: Optional[torch.device] = None,
    onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
    trt_engine_host_code_allowed: bool = True,
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    if not runtime_x_ray.trt_version:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as TRT libraries not detected",
            verbose_requested=verbose,
        )
        return False, "TRT backend not installed. Consider installing `trt10` extras."
    if not runtime_x_ray.trt_python_package_available:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as TRT python package not available",
            verbose_requested=verbose,
        )
        return (
            False,
            "Model package metadata delivered by weights provider lack required TRT package details",
        )
    if model_package.environment_requirements is None:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as environment requirements "
            f"not provided by backend.",
            verbose_requested=verbose,
        )
        return (
            False,
            "Model package metadata delivered by weights provider lack required TRT package details",
        )
    trt_compiled_with_cc_compatibility = False
    if model_package.trt_package_details is not None:
        trt_compiled_with_cc_compatibility = (
            model_package.trt_package_details.same_cc_compatible
        )
    trt_forward_compatible = False
    if model_package.trt_package_details is not None:
        trt_forward_compatible = (
            model_package.trt_package_details.trt_forward_compatible
        )
    trt_lean_runtime_excluded = False
    if model_package.trt_package_details is not None:
        trt_lean_runtime_excluded = (
            model_package.trt_package_details.trt_lean_runtime_excluded
        )
    model_environment = model_package.environment_requirements
    if isinstance(model_environment, JetsonEnvironmentRequirements):
        if model_environment.trt_version is None:
            verbose_info(
                message=f"Mode package with id '{model_package.package_id}' filtered out as model TRT version not provided by backend",
                verbose_requested=verbose,
            )
            return (
                False,
                "Model package metadata delivered by weights provider lack required TRT package details",
            )
        if runtime_x_ray.l4t_version is None:
            verbose_info(
                message=f"Mode package with id '{model_package.package_id}' filtered out as runtime environment does not declare L4T version",
                verbose_requested=verbose,
            )
            return (
                False,
                "Model package metadata delivered by weights provider lack required TRT package details",
            )
        device_compatibility = verify_trt_package_compatibility_with_cuda_device(
            all_available_cuda_devices=runtime_x_ray.gpu_devices,
            all_available_devices_cc=runtime_x_ray.gpu_devices_cc,
            compilation_device=model_environment.cuda_device_name,
            compilation_device_cc=model_environment.cuda_device_cc,
            selected_device=device,
            trt_compiled_with_cc_compatibility=trt_compiled_with_cc_compatibility,
        )
        if not device_compatibility:
            verbose_info(
                message=f"Model package with id '{model_package.package_id}' filtered out due to device incompatibility.",
                verbose_requested=verbose,
            )
            return False, "TRT model package is incompatible with your hardware"
        if not verify_versions_up_to_major_and_minor(
            runtime_x_ray.l4t_version, model_environment.l4t_version
        ):
            verbose_info(
                message=f"Mode package with id '{model_package.package_id}' filtered out as package L4T {model_environment.l4t_version} does not match runtime L4T: {runtime_x_ray.l4t_version}",
                verbose_requested=verbose,
            )
            return False, "TRT model package is incompatible with installed L4T version"
        if trt_forward_compatible:
            if runtime_x_ray.trt_version < model_environment.trt_version:
                verbose_info(
                    message=f"Mode package with id '{model_package.package_id}' filtered out as TRT version in "
                    f"environment ({runtime_x_ray.trt_version}) is older than engine TRT version "
                    f"({model_environment.trt_version}) - despite engine being forward compatible, "
                    f"TRT requires that TRT available in runtime is in version higher or equal compared "
                    f"to the one used for compilation.",
                    verbose_requested=verbose,
                )
                return (
                    False,
                    "TRT model package is incompatible with installed TRT version",
                )
            if trt_lean_runtime_excluded:
                # not supported for now
                verbose_info(
                    message=f"Mode package with id '{model_package.package_id}' filtered out as it was compiled to "
                    f"be forward compatible, but with lean runtime excluded from the engine - this mode is "
                    f"currently not supported in `inference-exp`.",
                    verbose_requested=verbose,
                )
                return False, "TRT model package is currently not supported"
            elif not trt_engine_host_code_allowed:
                verbose_info(
                    message=f"Mode package with id '{model_package.package_id}' filtered out as it contains TRT "
                    f"Lean Runtime that requires potentially unsafe deserialization which is forbidden "
                    f"in this configuration of `inference-exp`. Set `trt_engine_host_code_allowed=True` if "
                    f"you want this package to be supported.",
                    verbose_requested=verbose,
                )
                return False, (
                    "TRT model package cannot run with `trt_engine_host_code_allowed=False` - "
                    "consider settings adjustment."
                )
        elif runtime_x_ray.trt_version != model_environment.trt_version:
            verbose_info(
                message=f"Mode package with id '{model_package.package_id}' filtered out as package trt version {model_environment.trt_version} does not match runtime trt version: {runtime_x_ray.trt_version}",
                verbose_requested=verbose,
            )
            return False, "TRT model package is incompatible with installed TRT version"
        return True, None
    if not isinstance(model_environment, ServerEnvironmentRequirements):
        raise ModelPackageNegotiationError(
            message=f"Model package negotiation protocol not implemented for environment specification detected "
            f"in runtime. This is `inference-exp` bug - raise issue: https://github.com/roboflow/inference/issues",
            help_url="https://todo",
        )
    if model_environment.trt_version is None:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as model TRT version not provided by backend",
            verbose_requested=verbose,
        )
        return (
            False,
            "Model package metadata delivered by weights provider lack required TRT package details",
        )
    device_compatibility = verify_trt_package_compatibility_with_cuda_device(
        all_available_cuda_devices=runtime_x_ray.gpu_devices,
        all_available_devices_cc=runtime_x_ray.gpu_devices_cc,
        compilation_device=model_environment.cuda_device_name,
        compilation_device_cc=model_environment.cuda_device_cc,
        selected_device=device,
        trt_compiled_with_cc_compatibility=trt_compiled_with_cc_compatibility,
    )
    if not device_compatibility:
        verbose_info(
            message=f"Model package with id '{model_package.package_id}' filtered out due to device incompatibility.",
            verbose_requested=verbose,
        )
        return False, "TRT model package is incompatible with your hardware"
    if trt_forward_compatible:
        if runtime_x_ray.trt_version < model_environment.trt_version:
            verbose_info(
                message=f"Mode package with id '{model_package.package_id}' filtered out as TRT version in "
                f"environment ({runtime_x_ray.trt_version}) is older than engine TRT version "
                f"({model_environment.trt_version}) - despite engine being forward compatible, "
                f"TRT requires that TRT available in runtime is in version higher or equal compared "
                f"to the one used for compilation.",
                verbose_requested=verbose,
            )
            return False, "TRT model package is incompatible with installed TRT version"
        if trt_lean_runtime_excluded:
            # not supported for now
            verbose_info(
                message=f"Mode package with id '{model_package.package_id}' filtered out as it was compiled to "
                f"be forward compatible, but with lean runtime excluded from the engine - this mode is "
                f"currently not supported in `inference-exp`.",
                verbose_requested=verbose,
            )
            return False, "TRT model package is currently not supported"
        elif not trt_engine_host_code_allowed:
            verbose_info(
                message=f"Mode package with id '{model_package.package_id}' filtered out as it contains TRT "
                f"Lean Runtime that requires potentially unsafe deserialization which is forbidden "
                f"in this configuration of `inference-exp`. Set `trt_engine_host_code_allowed=True` if "
                f"you want this package to be supported.",
                verbose_requested=verbose,
            )
            return False, (
                "TRT model package cannot run with `trt_engine_host_code_allowed=False` - "
                "consider settings adjustment."
            )
    elif runtime_x_ray.trt_version != model_environment.trt_version:
        verbose_info(
            message=f"Mode package with id '{model_package.package_id}' filtered out as package trt version {model_environment.trt_version} does not match runtime trt version: {runtime_x_ray.trt_version}",
            verbose_requested=verbose,
        )
        return False, "TRT model package is incompatible with installed TRT version"
    return True, None


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


MODEL_TO_RUNTIME_COMPATIBILITY_MATCHERS = {
    BackendType.HF: hf_transformers_package_matches_runtime_environment,
    BackendType.TRT: trt_package_matches_runtime_environment,
    BackendType.ONNX: onnx_package_matches_runtime_environment,
    BackendType.TORCH: torch_package_matches_runtime_environment,
    BackendType.ULTRALYTICS: ultralytics_package_matches_runtime_environment,
}


def range_within_other(
    external_range: Tuple[int, int],
    internal_range: Tuple[int, int],
) -> bool:
    external_min, external_max = external_range
    internal_min, internal_max = internal_range
    return external_min <= internal_min <= internal_max <= external_max


def parse_batch_size(
    requested_batch_size: Union[int, Tuple[int, int]]
) -> Tuple[int, int]:
    if isinstance(requested_batch_size, tuple):
        if len(requested_batch_size) != 2:
            raise InvalidRequestedBatchSizeError(
                message="Could not parse batch size requested from model package negotiation procedure. "
                "Batch size request is supposed to be either integer value or tuple specifying (min, max) "
                f"batch size - but detected tuple of invalid size ({len(requested_batch_size)}) - this is "
                f"probably typo while specifying requested batch size.",
                help_url="https://todo",
            )
        min_batch_size, max_batch_size = requested_batch_size
        if not isinstance(min_batch_size, int) or not isinstance(max_batch_size, int):
            raise InvalidRequestedBatchSizeError(
                message="Could not parse batch size requested from model package negotiation procedure. "
                "Batch size request is supposed to be either integer value or tuple specifying (min, max) "
                f"batch size - but detected tuple elements which are not integer values - this is "
                f"probably typo while specifying requested batch size.",
                help_url="https://todo",
            )
        if max_batch_size < min_batch_size:
            raise InvalidRequestedBatchSizeError(
                message="Could not parse batch size requested from model package negotiation procedure. "
                "`max_batch_size` is lower than `min_batch_size` - which is invalid value - this is "
                "probably typo while specifying requested batch size.",
                help_url="https://todo",
            )
        if max_batch_size <= 0 or min_batch_size <= 0:
            raise InvalidRequestedBatchSizeError(
                message="Could not parse batch size requested from model package negotiation procedure. "
                "`min_batch_size` is <= 0 or `max_batch_size` <= - which is invalid value - this is "
                "probably typo while specifying requested batch size.",
                help_url="https://todo",
            )
        return min_batch_size, max_batch_size
    if not isinstance(requested_batch_size, int):
        raise InvalidRequestedBatchSizeError(
            message="Could not parse batch size requested from model package negotiation procedure. "
            "Batch size request is supposed to be either integer value or tuple specifying (min, max) "
            f"batch size - but detected single value which is not integer but has type "
            f"{requested_batch_size.__class__.__name__} - this is "
            f"probably typo while specifying requested batch size.",
            help_url="https://todo",
        )
    if requested_batch_size <= 0:
        raise InvalidRequestedBatchSizeError(
            message="Could not parse batch size requested from model package negotiation procedure. "
            "`requested_batch_size` is <= 0 which is invalid value this is "
            f"probably typo while specifying requested batch size.",
            help_url="https://todo",
        )
    return requested_batch_size, requested_batch_size


def parse_backend_type(value: str) -> BackendType:
    try:
        return BackendType(value)
    except ValueError as error:
        supported_backends = [e.value for e in BackendType]
        raise UnknownBackendTypeError(
            message=f"Requested backend of type '{value}' which is not recognized by `inference-exp`. Most likely this "
            f"error is a result of typo while specifying requested backend. Supported backends: "
            f"{supported_backends}.",
            help_url="https://todo",
        ) from error


def parse_requested_quantization(
    value: Union[str, Quantization, List[Union[str, Quantization]]]
) -> Set[Quantization]:
    if not isinstance(value, list):
        value = [value]
    result = set()
    for element in value:
        if isinstance(element, str):
            element = parse_quantization(value=element)
        result.add(element)
    return result


def parse_quantization(value: str) -> Quantization:
    try:
        return Quantization(value)
    except ValueError as error:
        raise UnknownQuantizationError(
            message=f"Requested quantization of type '{value}' which is not recognized by `inference-exp`. Most likely this "
            f"error is a result of typo while specifying requested quantization. Supported values: "
            f"{list(Quantization.__members__)}.",
            help_url="https://todo",
        ) from error
