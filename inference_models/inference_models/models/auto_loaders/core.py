import importlib
import importlib.util
import inspect
import json
import os.path
import re
import tempfile
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import torch
from filelock import FileLock
from rich.console import Console
from rich.text import Text

from inference_models.configuration import (
    DEFAULT_DEVICE,
    FILE_LOCK_ACQUIRE_TIMEOUT,
    INFERENCE_HOME,
    OFFLINE_MODE,
)
from inference_models.errors import (
    CorruptedModelPackageError,
    DirectLocalStorageAccessError,
    ForbiddenLocalCodePackageAccessError,
    InvalidModelInitParameterError,
    InvalidParameterError,
    MissingModelInitParameterError,
    ModelPackageAlternativesExhaustedError,
    ModelRetrievalError,
    NoModelPackagesAvailableError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_models.logger import LOGGER, verbose_info
from inference_models.models.auto_loaders.access_manager import (
    AccessIdentifiers,
    LiberalModelAccessManager,
    ModelAccessManager,
)
from inference_models.models.auto_loaders.auto_negotiation import (
    determine_default_allowed_quantization,
    filter_model_packages_based_on_model_features,
    filter_model_packages_by_requested_backend,
    filter_model_packages_by_requested_batch_size,
    filter_model_packages_by_requested_quantization,
    negotiate_model_packages,
    parse_backend_type,
)
from inference_models.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCache,
    AutoResolutionCacheEntry,
    BaseAutoLoadMetadataCache,
)
from inference_models.models.auto_loaders.constants import (
    MODEL_DEPENDENCIES_KEY,
    MODEL_DEPENDENCIES_SUB_DIR,
)
from inference_models.models.auto_loaders.dependency_models import (
    prepare_dependency_model_parameters,
)
from inference_models.models.auto_loaders.entities import (
    MODEL_CONFIG_FILE_NAME,
    AnyModel,
    BackendType,
    InferenceModelConfig,
    ModelArchitecture,
    TaskType,
)
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_model_cache_root_for_model_id,
    generate_model_package_cache_path,
    generate_models_cache_dir,
    generate_shared_blobs_path,
)
from inference_models.models.auto_loaders.models_registry import (
    INSTANCE_SEGMENTATION_TASK,
    OBJECT_DETECTION_TASK,
    model_implementation_exists,
    resolve_model_class,
)
from inference_models.models.auto_loaders.ranking import rank_model_packages
from inference_models.models.auto_loaders.presentation_utils import (
    calculate_artefacts_size,
    calculate_size_of_all_model_packages_artefacts,
    render_model_package_details_table,
    render_runtime_x_ray,
    render_table_with_model_overview,
    render_table_with_model_packages,
)
from inference_models.runtime_introspection.core import x_ray_runtime_environment
from inference_models.utils.download import (
    FileHandle,
    download_files_to_directory,
    is_valid_md5_hash,
)
from inference_models.utils.file_system import dump_json, read_json
from inference_models.utils.hashing import hash_dict_content
from inference_models.weights_providers.core import (
    get_model_from_provider,
    model_provider_requires_network,
)
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    LocalFileArtefactSpecs,
    ModelDependency,
    ModelPackageMetadata,
    PackageSourceType,
    Quantization,
    RecommendedParameters,
)

MODEL_TYPES_TO_LOAD_FROM_CHECKPOINT = {
    "rfdetr-base",
    "rfdetr-small",
    "rfdetr-medium",
    "rfdetr-nano",
    "rfdetr-large",
    "rfdetr-xlarge",
    "rfdetr-2xlarge",
    "rfdetr-seg-preview",
    "rfdetr-seg-nano",
    "rfdetr-seg-small",
    "rfdetr-seg-medium",
    "rfdetr-seg-large",
    "rfdetr-seg-xlarge",
    "rfdetr-seg-2xlarge",
    "rfdetr-seg-xxlarge",
}
OFFLINE_CACHE_MANIFEST_VERSION = 2

DEFAULT_KWARGS_PARAMS_TO_BE_FORWARDED_TO_DEPENDENT_MODELS = [
    "owlv2_enforce_model_compilation",
    "owlv2_class_embeddings_cache",
    "owlv2_images_embeddings_cache",
]


def _canonicalize_unordered_request_values(
    value: object,
    case_insensitive: bool = False,
) -> object:
    """Stabilize values whose package-selection semantics treat them as sets."""

    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    serialized_values = set()
    for item in values:
        serialized_item = getattr(item, "value", item)
        if case_insensitive and isinstance(serialized_item, str):
            serialized_item = serialized_item.lower()
        serialized_values.add(serialized_item)
    return sorted(serialized_values)


def _runtime_compatibility_content(runtime_x_ray: object) -> dict:
    """Return stable machine-compatibility data independent of display text."""

    def stringify(value: object) -> Optional[str]:
        return None if value is None else str(value)

    available_providers = getattr(
        runtime_x_ray, "available_onnx_execution_providers", None
    )
    return {
        "version": 1,
        "gpu_available": getattr(runtime_x_ray, "gpu_available", False),
        "gpu_devices": list(getattr(runtime_x_ray, "gpu_devices", [])),
        "gpu_devices_cc": [
            str(value) for value in getattr(runtime_x_ray, "gpu_devices_cc", [])
        ],
        "driver_version": stringify(
            getattr(runtime_x_ray, "driver_version", None)
        ),
        "cuda_version": stringify(getattr(runtime_x_ray, "cuda_version", None)),
        "trt_version": stringify(getattr(runtime_x_ray, "trt_version", None)),
        "jetson_type": getattr(runtime_x_ray, "jetson_type", None),
        "l4t_version": stringify(getattr(runtime_x_ray, "l4t_version", None)),
        "os_version": getattr(runtime_x_ray, "os_version", None),
        "torch_available": getattr(runtime_x_ray, "torch_available", False),
        "torch_version": stringify(
            getattr(runtime_x_ray, "torch_version", None)
        ),
        "torchvision_version": stringify(
            getattr(runtime_x_ray, "torchvision_version", None)
        ),
        "onnxruntime_version": stringify(
            getattr(runtime_x_ray, "onnxruntime_version", None)
        ),
        "available_onnx_execution_providers": (
            sorted(available_providers)
            if available_providers is not None
            else None
        ),
        "hf_transformers_available": getattr(
            runtime_x_ray, "hf_transformers_available", False
        ),
        "trt_python_package_available": getattr(
            runtime_x_ray, "trt_python_package_available", False
        ),
    }


def _runtime_compatibility_hash(runtime_x_ray: object) -> str:
    return hash_dict_content(
        content=_runtime_compatibility_content(runtime_x_ray=runtime_x_ray)
    )


class AutoModel:

    @classmethod
    def describe_model(
        cls,
        model_id: str,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        pull_artefacts_size: bool = False,
        weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Display comprehensive metadata and available packages for a model.

        Shows detailed information about a model without loading it, including:

        - Model architecture and variant
        - Task type (object detection, classification, etc.)
        - Available model packages (different backends, quantizations, batch sizes)
        - Package requirements and compatibility
        - Model dependencies (if any)
        - Package sizes (optional, requires network requests)

        This is useful for:

        - Exploring available models before loading
        - Understanding which backends are available for a model
        - Checking model requirements and compatibility
        - Debugging model loading issues
        - Selecting the right package for your environment

        Args:
            model_id: Model identifier. Can be:
                - Pre-trained model ID (e.g., "yolov8n-640", "rfdetr-base")
                - Custom Roboflow model (e.g., "my-project/2")

            weights_provider: Source for model metadata. Options:
                - "roboflow" (default): Query Roboflow platform
                - Custom provider name (if registered)

            api_key: Roboflow API key for accessing private models. If not provided,
                uses the `ROBOFLOW_API_KEY` environment variable. Not required for
                public pre-trained models.

            pull_artefacts_size: Whether to calculate and display the total size of
                each model package. This requires making network requests to check
                file sizes, so it's slower. Default: False.

            weights_provider_extra_query_params: Extra query parameters to pass to the weights' provider. Advanced
                usage only.

            weights_provider_extra_headers: Extra headers to pass to the weights' provider. Advanced
                usage only.

        Returns:
            None. Prints formatted tables to the console showing:
                1. Model overview table with architecture, task type, and dependencies
                2. Available packages table with backend, quantization, and batch size info

        Raises:
            UnauthorizedModelAccessError: If API key is invalid or model access is denied.
            ModelNotFoundError: If the model ID doesn't exist in the weights provider.

        Examples:
            View model information:

            >>> from inference_models import AutoModel
            >>> AutoModel.describe_model("yolov8n-640")

            View with package sizes:

            >>> AutoModel.describe_model("rfdetr-base", pull_artefacts_size=True)
            # Same as above, but includes a "Size" column showing package sizes

            View private model:

            >>> AutoModel.describe_model(
            ...     "my-workspace/my-model/2",
            ...     api_key="your_api_key"
            ... )

        See Also:
            - `AutoModel.describe_model_package()`: View detailed info for a specific package
            - `AutoModel.describe_compute_environment()`: Check your runtime environment
            - `AutoModel.from_pretrained()`: Load a model after inspecting it
        """
        model_metadata = get_model_from_provider(
            provider=weights_provider,
            model_id=model_id,
            api_key=api_key,
            weights_provider_extra_query_params=weights_provider_extra_query_params,
            weights_provider_extra_headers=weights_provider_extra_headers,
        )
        model_packages_size = None
        if pull_artefacts_size:
            model_packages_size = calculate_size_of_all_model_packages_artefacts(
                model_packages=model_metadata.model_packages
            )
        console = Console()
        model_overview_table = render_table_with_model_overview(
            model_id=model_metadata.model_id,
            requested_model_id=model_id,
            model_architecture=model_metadata.model_architecture,
            model_variant=model_metadata.model_variant,
            task_type=model_metadata.task_type,
            weights_provider=weights_provider,
            registered_packages=len(model_metadata.model_packages),
            model_dependencies=model_metadata.model_dependencies,
        )
        console.print(model_overview_table)
        console.print("\n")
        packages_overview_table = render_table_with_model_packages(
            model_packages=model_metadata.model_packages,
            model_packages_size=model_packages_size,
        )
        console.print(packages_overview_table)
        text = Text.assemble(
            ("\nWant to check more details about specific package?", "bold"),
            "\nUse AutoModel.describe_model_package('model_id', 'package_id').",
        )
        console.print(text)
        if not pull_artefacts_size:
            text = Text.assemble(
                ("\nWant to verify the size of model package?", "bold"),
                "\nUse AutoModel.describe_model('model_id', pull_artefacts_size=True) - the execution will be "
                "slightly longer, as we must collect the size of all elements of model package.",
            )
            console.print(text)

    @classmethod
    def describe_model_package(
        cls,
        model_id: str,
        package_id: str,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        pull_artefacts_size: bool = True,
        weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Display detailed information about a specific model package.

        Shows comprehensive details for a single model package, including:

        - Backend type (PyTorch, ONNX, TensorRT, etc.)
        - Quantization level (FP32, FP16, INT8, etc.)
        - Batch size configuration (fixed or dynamic)
        - Required dependencies and environment
        - Package artifacts (model files, configs, etc.)
        - Total package size (optional)
        - Hardware requirements (CUDA version, TensorRT version, etc.)

        This is useful for:

        - Understanding package requirements before loading
        - Debugging compatibility issues
        - Checking package size before download
        - Verifying package contents

        Args:
            model_id: Model identifier. Can be:
                - Pre-trained model ID (e.g., "yolov8n-640", "rfdetr-base")
                - Custom Roboflow model (e.g., "my-project/2")

            package_id: Specific package identifier to inspect. Get this from
                `AutoModel.describe_model()` output.

            weights_provider: Source for model metadata. Options:
                - "roboflow" (default): Query Roboflow platform
                - Custom provider name (if registered)

            api_key: Roboflow API key for accessing private models. If not provided,
                uses the `ROBOFLOW_API_KEY` environment variable. Not required for
                public pre-trained models.

            pull_artefacts_size: Whether to calculate and display the size of each
                artifact in the package. This requires making network requests to check
                file sizes, so it's slower. Default: True.

            weights_provider_extra_query_params: Extra query parameters to pass to the weights' provider. Advanced
                usage only.

            weights_provider_extra_headers: Extra headers to pass to the weights' provider. Advanced
                usage only.

        Returns:
            None. Prints a formatted table to the console showing package details.

        Raises:
            UnauthorizedModelAccessError: If API key is invalid or model access is denied.
            ModelNotFoundError: If the model ID doesn't exist in the weights provider.
            NoModelPackagesAvailableError: If the specified package_id doesn't exist
                for this model.

        Examples:
            View package details:

            >>> from inference_models import AutoModel
            >>> # First, see available packages
            >>> AutoModel.describe_model("yolov8n-640")
            >>> # Then inspect a specific package
            >>> AutoModel.describe_model_package("yolov8n-640", "pkg-trt-fp16-1-32")

            View without artifact sizes (faster):

            >>> AutoModel.describe_model_package(
            ...     "rfdetr-base",
            ...     "pkg-torch-fp32",
            ...     pull_artefacts_size=False
            ... )

        See Also:
            - `AutoModel.describe_model()`: View all available packages for a model
            - `AutoModel.describe_compute_environment()`: Check your runtime environment
            - `AutoModel.from_pretrained()`: Load a model with a specific package
        """
        model_metadata = get_model_from_provider(
            provider=weights_provider,
            model_id=model_id,
            api_key=api_key,
            weights_provider_extra_query_params=weights_provider_extra_query_params,
            weights_provider_extra_headers=weights_provider_extra_headers,
        )
        selected_package = None
        for package in model_metadata.model_packages:
            if package.package_id == package_id:
                selected_package = package
        if selected_package is None:
            raise NoModelPackagesAvailableError(
                message=f"Selected model package {package_id} does not exist for model {model_id}. Make sure provided "
                f"value is valid.",
                help_url="https://inference-models.roboflow.com/errors/package-negotiation/#nomodelpackagesavailableerror",
            )
        artefacts_size = None
        if pull_artefacts_size:
            artefacts_size = calculate_artefacts_size(
                package_artefacts=selected_package.package_artefacts
            )
        table = render_model_package_details_table(
            model_id=model_metadata.model_id,
            requested_model_id=model_id,
            artefacts_size=artefacts_size,
            model_package=selected_package,
        )
        console = Console()
        console.print(table)
        if not pull_artefacts_size:
            text = Text.assemble(
                ("\nWant to verify the size of model package?", "bold"),
                "\nUse AutoModel.describe_model_package('model_id', 'package_id', pull_artefacts_size=True)"
                "- the execution will be slightly longer, as we must collect the size of all elements of model package.",
            )
            console.print(text)

    @classmethod
    def describe_compute_environment(cls) -> None:
        """Inspect and display the current runtime environment and available backends.

        Performs a comprehensive scan of your system to detect:

        - **Hardware**: GPU availability, GPU models, compute capability
        - **CUDA**: Driver version, CUDA toolkit version
        - **TensorRT**: TensorRT version and availability
        - **PyTorch**: PyTorch and torchvision versions
        - **ONNX Runtime**: Version and available execution providers
        - **Other backends**: Hugging Face Transformers, Ultralytics
        - **Platform**: OS version, Jetson type (if applicable), L4T version

        This is useful for:

        - Debugging model loading issues
        - Verifying backend installations
        - Checking hardware compatibility
        - Understanding which model packages will work in your environment
        - Troubleshooting performance issues

        Returns:
            None. Prints a formatted table to the console showing all detected
            environment information.

        Examples:
            Check your environment:

            >>> from inference_models import AutoModel
            >>> AutoModel.describe_compute_environment()
            # Displays output like:
                                        Compute environment details
            Detected GPUs:                      N/A
            Detected GPUs CUDA CC:              N/A
            NVIDIA driver:                      N/A
            CUDA version:                       N/A
            TRT version:                        N/A
            TRT Python package available:       False
            OS version:                         macos-26.2-arm64-arm-64bit
            torch version:                      2.6.0
            torchvision version:                0.21.0
            ONNX runtime version:               1.21.0
            Detected ONNX execution providers:  CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider

        See Also:
            - `AutoModel.describe_model()`: View model metadata and requirements
            - `AutoModel.from_pretrained()`: Load a model (uses this environment info)
        """
        runtime_x_ray = x_ray_runtime_environment()
        table = render_runtime_x_ray(runtime_x_ray=runtime_x_ray)
        console = Console()
        console.print(table)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        model_package_id: Optional[str] = None,
        backend: Optional[
            Union[str, BackendType, List[Union[str, BackendType]]]
        ] = None,
        batch_size: Optional[Union[int, Tuple[int, int]]] = None,
        quantization: Optional[
            Union[str, Quantization, List[Union[str, Quantization]]]
        ] = None,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        default_onnx_trt_options: bool = True,
        max_package_loading_attempts: Optional[int] = None,
        verbose: bool = False,
        model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
        allow_untrusted_packages: bool = False,
        trt_engine_host_code_allowed: bool = True,
        allow_local_code_packages: bool = True,
        verify_hash_while_download: bool = True,
        download_files_without_hash: bool = False,
        use_auto_resolution_cache: bool = True,
        auto_resolution_cache: Optional[AutoResolutionCache] = None,
        allow_direct_local_storage_loading: bool = True,
        model_access_manager: Optional[ModelAccessManager] = None,
        nms_fusion_preferences: Optional[Union[bool, dict]] = None,
        model_type: Optional[str] = None,
        task_type: Optional[str] = None,
        allow_loading_dependency_models: bool = True,
        dependency_models_params: Optional[dict] = None,
        point_model_directory: Optional[Callable[[str], None]] = None,
        forwarded_kwargs: Optional[List[str]] = None,
        weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AnyModel:
        """Load and initialize a computer vision model with automatic backend selection.

        This is the primary entry point for loading models in `inference-models`. It automatically:

        - Downloads model weights from the specified provider (default: Roboflow)
        - Selects the optimal backend (TensorRT > PyTorch Hugging Face> > ONNX > others)
        - Configures the model for your hardware (CPU/GPU)
        - Handles caching of atrefacts

        Args:
            model_id_or_path: Model identifier or local path. Can be:
                - Pre-trained model ID (e.g., "yolov8n-640", "rfdetr-base", "resnet50")
                - Custom Roboflow model (e.g., "my-project/2")
                - Local directory path containing model files
                - Local checkpoint file path (e.g., "/path/to/checkpoint.pth")

            weights_provider: Source for model weights. Options:
                - "roboflow" (default): Download from Roboflow platform
                - "local": Load from local filesystem
                - Custom provider name (if registered via `register_model_provider()`)

            api_key: Roboflow API key for accessing private models. If not provided,
                uses the `ROBOFLOW_API_KEY` environment variable. Not required for
                public pre-trained models.

            model_package_id: Specific model package to load (advanced). If not provided,
                automatically selects the best package based on your environment and
                requested backend/quantization. Use `AutoModel.describe_model()` to see
                available packages.

            backend: Preferred inference backend(s). Can be:
                - Single backend: "torch", "onnx", "trt" (TensorRT), "hugging-face"
                - List of allowed backends: ["trt", "torch"] (the normal
                  compatibility and package ranking rules choose among them)
                - BackendType enum value(s)
                - None (default): Automatic selection (TensorRT > PyTorch > ONNX > HF)

            batch_size: Preferred batch size for inference. Can be:
                - Single integer: Fixed batch size (e.g., 1, 8, 16)
                - Tuple: Range of batch sizes (e.g., (1, 8) for dynamic batching)
                - None (default): Use model's default batch size
                Note: Only affects models with multiple batch size variants.

            quantization: Model quantization level(s). Can be:
                - Single value: "fp32", "fp16", "bf16", "int8"
                - List of allowed values: ["fp16", "fp32"] (the normal package
                  ranking rules choose among them)
                - Quantization enum value(s)
                - None (default): Automatic selection based on device capabilities

            onnx_execution_providers: ONNX Runtime execution providers (ONNX backend only).
                Examples:
                - ["CUDAExecutionProvider", "CPUExecutionProvider"]
                - [("TensorrtExecutionProvider", {"trt_fp16_enable": True})]
                If not provided, automatically selects based on available hardware.

            device: PyTorch device for model execution. Can be:
                - String: "cpu", "cuda", "cuda:0", "cuda:1", "mps"
                - torch.device object
                Default: "cuda" if available, otherwise "cpu"

            default_onnx_trt_options: Whether to use default TensorRT optimization options
                for ONNX Runtime's TensorRT execution provider. Default: True.

            max_package_loading_attempts: Maximum number of model packages to try before
                failing. Useful when multiple packages are available. Default: Try all
                matching packages.

            verbose: Enable detailed logging during model loading. Useful for debugging
                package selection and download issues. Default: False.

            model_download_file_lock_acquire_timeout: Timeout in seconds for acquiring
                file locks during concurrent downloads. Default: FILE_LOCK_ACQUIRE_TIMEOUT (20).

            allow_untrusted_packages: Allow loading model packages with custom code that
                haven't been verified. **Security risk** - only enable for trusted sources.
                Default: False.

            trt_engine_host_code_allowed: Allow TensorRT engines to execute host code.
                Required for some TensorRT optimizations. Default: True.

            allow_local_code_packages: Allow loading models with custom Python code from
                local directories. Default: True.

            verify_hash_while_download: Verify file integrity using checksums during
                download. Recommended for production. Default: True.

            download_files_without_hash: Allow downloading files that don't have checksums.
                **Security risk** - only enable for trusted sources. Default: False.

            use_auto_resolution_cache: Enable caching of model resolution results to speed
                up subsequent loads. Default: True.

            auto_resolution_cache: Custom cache implementation. If None, uses default
                file-based cache. Advanced usage only.

            allow_direct_local_storage_loading: Allow loading models directly from local
                paths without going through the weights provider. Default: True.

            model_access_manager: Custom model access control manager. If None, uses
                permissive default. Advanced usage only.

            nms_fusion_preferences: Non-Maximum Suppression fusion preferences for ONNX
                models. Can be:
                - True: Enable NMS fusion with default settings
                - False: Disable NMS fusion
                - dict: Custom NMS fusion configuration
                - None (default): Use model's default settings

            model_type: Override model architecture type (advanced). Only needed when
                loading local models without metadata. Examples: "yolov8", "rfdetr".

            task_type: Override task type (advanced). Only needed when loading local
                models without metadata. Examples: "object-detection", "classification".

            allow_loading_dependency_models: Allow loading models that depend on other
                models (e.g., some VLMs depend on separate vision encoders). Default: True.

            dependency_models_params: Parameters to pass to dependency models. Dict mapping
                dependency names to parameter dicts. Advanced usage only.

            point_model_directory: Callback function called with the model directory path
                after loading. Advanced usage only.

            forwarded_kwargs: List of kwargs to forward to dependency models. Advanced
                usage only.

            weights_provider_extra_query_params: Extra query parameters to pass to the weights' provider. Advanced
                usage only.

            weights_provider_extra_headers: Extra headers to pass to the weights' provider. Advanced
                usage only.

            **kwargs: Additional model-specific parameters passed to the model's
                `from_pretrained()` method. Varies by model type.

        Returns:
            Loaded model instance. The specific type depends on the model's task:
                - ObjectDetectionModel: For object detection (YOLO, RF-DETR, etc.)
                - ClassificationModel: For single-label classification
                - MultiLabelClassificationModel: For multi-label classification
                - InstanceSegmentationModel: For instance segmentation
                - KeyPointsDetectionModel: For keypoint detection
                - DepthEstimationModel: For depth estimation
                - StructuredOCRModel: For OCR with structured output
                - TextImageEmbeddingModel: For vision-language embeddings (CLIP, etc.)
                - OpenVocabularyObjectDetectionModel: For open-vocabulary detection

        Raises:
            UnauthorizedModelAccessError: If API key is invalid or model access is denied.
            ModelPackageNotFoundError: If no compatible model package is found for your
                environment and requested parameters.
            CorruptedModelPackageError: If model files are corrupted or incomplete.
            InvalidParameterError: If provided parameters are invalid.
            DirectLocalStorageAccessError: If local path loading is disabled but a local
                path was provided.

        Examples:
            Basic usage with pre-trained model:

            >>> from inference_models import AutoModel
            >>> model = AutoModel.from_pretrained("yolov8n-640")
            >>> predictions = model(image)

            Load custom Roboflow model:

            >>> model = AutoModel.from_pretrained(
            ...     "my-project/2",
            ...     api_key="your_api_key"
            ... )

            Force specific backend and device:

            >>> model = AutoModel.from_pretrained(
            ...     "rfdetr-base",
            ...     backend="torch",
            ...     device="cuda:1"
            ... )

            Load with quantization:

            >>> model = AutoModel.from_pretrained(
            ...     "yolov8n-640",
            ...     quantization="fp16"
            ... )

            Load from local checkpoint:

            >>> model = AutoModel.from_pretrained(
            ...     "/path/to/checkpoint.pth",
            ...     model_type="rfdetr-base",
            ...     labels=["cat", "dog"]
            ... )

            Enable verbose logging:

            >>> model = AutoModel.from_pretrained(
            ...     "yolov8n-640",
            ...     verbose=True
            ... )

        See Also:
            - `AutoModel.describe_model()`: View model metadata before loading
            - `AutoModel.describe_model_package()`: View specific package details
            - `AutoModel.describe_compute_environment()`: Check available backends
            - `AutoModel.list_available_models()`: List all registered models
        """
        if model_access_manager is None:
            model_access_manager = LiberalModelAccessManager()
        if model_access_manager.is_model_access_forbidden(
            model_id=model_id_or_path, api_key=api_key
        ):
            raise UnauthorizedModelAccessError(
                message=f"Unauthorized not access model with ID: {model_package_id}. Are you sure you use valid "
                f"API key? The default weights provider is Roboflow - see Roboflow authentication details: "
                f"https://docs.roboflow.com/api-reference/authentication "
                f"and export key to `ROBOFLOW_API_KEY` environment variable. If you use custom weights "
                f"provider - verify access constraints relevant for the provider.",
                help_url="https://inference-models.roboflow.com/errors/model-retrieval/#unauthorizedmodelaccesserror",
            )
        if auto_resolution_cache is None:

            def register_file_created_for_model_package(
                file_path: str, model_id: str, package_id: str
            ) -> None:
                access_identifiers = AccessIdentifiers(
                    model_id=model_id,
                    package_id=package_id,
                    api_key=api_key,
                )
                model_access_manager.on_file_created(
                    file_path=file_path,
                    access_identifiers=access_identifiers,
                )

            auto_resolution_cache = BaseAutoLoadMetadataCache(
                file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                verbose=verbose,
                on_file_created=register_file_created_for_model_package,
                on_file_deleted=model_access_manager.on_file_deleted,
            )
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except RuntimeError as error:
                raise InvalidParameterError(
                    message="Could not parse `device` parameter value - make sure that it is a valid string "
                    f"representation of torch device. Valid values: 'cpu', 'cuda' or 'cuda:0'. If you see this error "
                    "while using Roboflow infrastructure - contact us to get help. Otherwise - verify your setup.",
                    help_url="https://inference-models.roboflow.com/errors/input-validation/#invalidparametererror",
                ) from error
        model_init_kwargs = {
            "onnx_execution_providers": onnx_execution_providers,
            "device": device,
            "default_onnx_trt_options": default_onnx_trt_options,
            "engine_host_code_allowed": trt_engine_host_code_allowed,
        }
        model_init_kwargs.update(kwargs)
        if not os.path.exists(model_id_or_path):
            # QUESTION: is it enough to assume presence of local dir as the intent to load
            # model from disc drive? What if we have clash of model id / model alias with
            # contents of someone's local drive - shall we then try to load from both sources?
            # that still may end up with ambiguous behavior - probably the solution would be
            # to require prefix like file://... to denote the intent of loading model from local
            # drive?
            runtime_x_ray = x_ray_runtime_environment()
            runtime_compatibility = _runtime_compatibility_content(
                runtime_x_ray=runtime_x_ray
            )
            offline_compatibility_content = {
                "provider": weights_provider,
                "model_id": model_id_or_path,
                "requested_model_package_id": model_package_id,
                "requested_backends": _canonicalize_unordered_request_values(
                    backend,
                    case_insensitive=True,
                ),
                "requested_batch_size": batch_size,
                "requested_quantization": _canonicalize_unordered_request_values(
                    quantization
                ),
                "device": str(device),
                "onnx_execution_providers": onnx_execution_providers,
                "default_onnx_trt_options": default_onnx_trt_options,
                "allow_untrusted_packages": allow_untrusted_packages,
                "trt_engine_host_code_allowed": trt_engine_host_code_allowed,
                "allow_local_code_packages": allow_local_code_packages,
                "verify_hash_while_download": verify_hash_while_download,
                "download_files_without_hash": download_files_without_hash,
                "allow_loading_dependency_models": allow_loading_dependency_models,
                "nms_fusion_preferences": nms_fusion_preferences,
                "weights_provider_extra_query_params": weights_provider_extra_query_params,
                "weights_provider_extra_headers": weights_provider_extra_headers,
                "runtime_compatibility": runtime_compatibility,
            }
            offline_compatibility_hash = hash_dict_content(
                content=offline_compatibility_content
            )
            auto_negotiation_hash = hash_dict_content(
                content={
                    **offline_compatibility_content,
                    "api_key": api_key,
                }
            )
            model_from_access_manager = model_access_manager.retrieve_model_instance(
                model_id=model_id_or_path,
                package_id=model_package_id,
                api_key=api_key,
                loading_parameter_digest=auto_negotiation_hash,
            )
            if model_from_access_manager:
                return model_from_access_manager
            if forwarded_kwargs is None:
                forwarded_kwargs = (
                    DEFAULT_KWARGS_PARAMS_TO_BE_FORWARDED_TO_DEPENDENT_MODELS
                )
            forwarded_kwargs_values = {
                name: kwargs[name] for name in forwarded_kwargs if name in kwargs
            }

            def attempt_cached_load(cache_hash: str) -> Optional[AnyModel]:
                return attempt_loading_model_with_auto_load_cache(
                    use_auto_resolution_cache=use_auto_resolution_cache,
                    auto_resolution_cache=auto_resolution_cache,
                    auto_negotiation_hash=cache_hash,
                    model_access_manager=model_access_manager,
                    model_name_or_path=model_id_or_path,
                    model_init_kwargs=dict(model_init_kwargs),
                    api_key=api_key,
                    allow_loading_dependency_models=allow_loading_dependency_models,
                    forwarded_kwargs_values=forwarded_kwargs_values,
                    verbose=verbose,
                    weights_provider=weights_provider,
                    max_package_loading_attempts=max_package_loading_attempts,
                    model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                    allow_untrusted_packages=allow_untrusted_packages,
                    trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                    allow_local_code_packages=allow_local_code_packages,
                    verify_hash_while_download=verify_hash_while_download,
                    download_files_without_hash=download_files_without_hash,
                    allow_direct_local_storage_loading=allow_direct_local_storage_loading,
                    dependency_models_params=dependency_models_params,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                )

            def attempt_compatible_cached_load() -> Optional[AnyModel]:
                if not use_auto_resolution_cache:
                    return None
                compatible_candidates = (
                    auto_resolution_cache.find_compatible_candidates(
                        offline_compatibility_hash=offline_compatibility_hash
                    )
                )
                for compatible_hash, cache_entry in compatible_candidates:
                    if compatible_hash == auto_negotiation_hash:
                        continue
                    model = attempt_cached_load(compatible_hash)
                    if model is None:
                        continue
                    if point_model_directory:
                        point_model_directory(
                            generate_model_package_cache_path(
                                model_id=cache_entry.cache_model_id
                                or cache_entry.model_id,
                                package_id=cache_entry.model_package_id,
                            )
                        )
                    return model
                return None

            def attempt_raw_cached_load() -> Optional[Tuple[AnyModel, str]]:
                return attempt_loading_model_from_offline_cache(
                    model_id=model_id_or_path,
                    model_init_kwargs=dict(model_init_kwargs),
                    requested_model_package_id=model_package_id,
                    requested_backends=backend,
                    requested_batch_size=batch_size,
                    requested_quantization=quantization,
                    model_access_manager=model_access_manager,
                    api_key=api_key,
                    allow_local_code_packages=allow_local_code_packages,
                    allow_untrusted_packages=allow_untrusted_packages,
                    allow_loading_dependency_models=allow_loading_dependency_models,
                    dependency_models_params=dependency_models_params,
                    forwarded_kwargs_values=forwarded_kwargs_values,
                    weights_provider=weights_provider,
                    auto_resolution_cache=auto_resolution_cache,
                    use_auto_resolution_cache=use_auto_resolution_cache,
                    max_package_loading_attempts=max_package_loading_attempts,
                    model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                    trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                    verify_hash_while_download=verify_hash_while_download,
                    download_files_without_hash=download_files_without_hash,
                    allow_direct_local_storage_loading=allow_direct_local_storage_loading,
                    nms_fusion_preferences=nms_fusion_preferences,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                    verbose=verbose,
                    offline_compatibility_hash=offline_compatibility_hash,
                )

            model_from_cache = attempt_cached_load(auto_negotiation_hash)
            if model_from_cache:
                return model_from_cache
            if OFFLINE_MODE and model_provider_requires_network(
                provider=weights_provider
            ):
                model_from_cache = attempt_compatible_cached_load()
                if model_from_cache is not None:
                    return model_from_cache
                offline_result = attempt_raw_cached_load()
                if offline_result is not None:
                    model, offline_cache_dir = offline_result
                    if point_model_directory:
                        point_model_directory(offline_cache_dir)
                    return model
                raise ModelRetrievalError(
                    message=f"Cannot load model {model_id_or_path} in OFFLINE_MODE - "
                    f"no compatible cached model package found in "
                    f"{INFERENCE_HOME}/models-cache/. "
                    f"Pre-populate the cache by running once with network access, "
                    f"or disable OFFLINE_MODE.",
                    help_url="https://inference-models.roboflow.com/errors/model-retrieval/#modelretrievalerror",
                )
            try:
                model_metadata = get_model_from_provider(
                    provider=weights_provider,
                    model_id=model_id_or_path,
                    api_key=api_key,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                )
                if (
                    model_metadata.model_dependencies
                    and not allow_loading_dependency_models
                ):
                    raise CorruptedModelPackageError(
                        message=f"Could not load model {model_id_or_path} as it defines another models which are "
                        f"it's dependency, but the auto-loader prevents loading dependencies at certain "
                        f"nesting depth to avoid excessive resolution procedure. This is a limitation of "
                        f"current implementation. Provide us the context of your use-case to get help.",
                        help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                    )
                if model_metadata.model_id != model_id_or_path:
                    model_access_manager.on_model_alias_discovered(
                        alias=model_id_or_path,
                        model_id=model_metadata.model_id,
                    )
                model_dependencies = model_metadata.model_dependencies or []
                for model_dependency in model_dependencies:
                    model_access_manager.on_model_dependency_discovered(
                        base_model_id=model_dependency.model_id,
                        base_model_package_id=model_dependency.model_package_id,
                        dependent_model_id=model_metadata.model_id,
                    )
                for model_package in model_metadata.model_packages:
                    package_access_identifiers = AccessIdentifiers(
                        model_id=model_metadata.model_id,
                        package_id=model_package.package_id,
                        api_key=api_key,
                    )
                    model_access_manager.on_model_package_access_granted(
                        package_access_identifiers
                    )
            except UnauthorizedModelAccessError as error:
                model_access_manager.on_model_access_forbidden(
                    model_id=model_id_or_path, api_key=api_key
                )
                raise error
            except RetryError:
                verbose_info(
                    message=f"API unreachable for model {model_id_or_path}, "
                    f"attempting offline cache fallback.",
                    verbose_requested=verbose,
                )
                model_from_cache = attempt_compatible_cached_load()
                if model_from_cache is not None:
                    return model_from_cache
                offline_result = attempt_raw_cached_load()
                if offline_result is None:
                    raise
                model, offline_cache_dir = offline_result
                if point_model_directory:
                    point_model_directory(offline_cache_dir)
                return model
            # here we verify if de-aliasing or access confirmation from auth master changed something
            model_from_access_manager = model_access_manager.retrieve_model_instance(
                model_id=model_id_or_path,
                package_id=model_package_id,
                api_key=api_key,
                loading_parameter_digest=auto_negotiation_hash,
            )
            if model_from_access_manager:
                return model_from_access_manager
            matching_model_packages = negotiate_model_packages(
                model_architecture=model_metadata.model_architecture,
                task_type=model_metadata.task_type,
                model_packages=model_metadata.model_packages,
                requested_model_package_id=model_package_id,
                requested_backends=backend,
                requested_batch_size=batch_size,
                requested_quantization=quantization,
                device=device,
                onnx_execution_providers=onnx_execution_providers,
                allow_untrusted_packages=allow_untrusted_packages,
                trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                nms_fusion_preferences=nms_fusion_preferences,
                verbose=verbose,
            )
            model_dependencies_instances = {}
            model_dependencies_directories = {}
            dependency_models_params = dependency_models_params or {}
            for model_dependency in model_dependencies:
                dependency_params = dict(
                    dependency_models_params.get(model_dependency.name, {})
                )
                dependency_params["model_id_or_path"] = model_dependency.model_id
                dependency_params["model_package_id"] = (
                    model_dependency.model_package_id
                )
                resolved_model_parameters = prepare_dependency_model_parameters(
                    model_parameters=dependency_params
                )
                verbose_info(
                    message=f"Initialising dependent model: {model_dependency.model_id}",
                    verbose_requested=verbose,
                )

                def model_directory_pointer(model_dir: str) -> None:
                    model_dependencies_directories[model_dependency.name] = model_dir

                for name, value in forwarded_kwargs_values.items():
                    if name not in resolved_model_parameters.model_extra:
                        resolved_model_parameters.model_extra[name] = value

                dependency_instance = AutoModel.from_pretrained(
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
                    model_access_manager=model_access_manager,
                    nms_fusion_preferences=resolved_model_parameters.nms_fusion_preferences,
                    model_type=resolved_model_parameters.model_type,
                    task_type=resolved_model_parameters.task_type,
                    allow_loading_dependency_models=False,
                    dependency_models_params=None,
                    point_model_directory=model_directory_pointer,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                    **resolved_model_parameters.kwargs,
                )
                model_dependencies_instances[model_dependency.name] = (
                    dependency_instance
                )

            return attempt_loading_matching_model_packages(
                model_id=model_id_or_path,
                model_architecture=model_metadata.model_architecture,
                task_type=model_metadata.task_type,
                matching_model_packages=matching_model_packages,
                model_init_kwargs=model_init_kwargs,
                model_access_manager=model_access_manager,
                auto_negotiation_hash=auto_negotiation_hash,
                offline_compatibility_hash=offline_compatibility_hash,
                api_key=api_key,
                model_dependencies=model_metadata.model_dependencies,
                model_dependencies_instances=model_dependencies_instances,
                model_dependencies_directories=model_dependencies_directories,
                recommended_parameters=model_metadata.recommended_parameters,
                max_package_loading_attempts=max_package_loading_attempts,
                model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                verify_hash_while_download=verify_hash_while_download,
                download_files_without_hash=download_files_without_hash,
                auto_resolution_cache=auto_resolution_cache,
                use_auto_resolution_cache=use_auto_resolution_cache,
                point_model_directory=point_model_directory,
                verbose=verbose,
            )
        if not allow_direct_local_storage_loading:
            raise DirectLocalStorageAccessError(
                message="Attempted to load model directly pointing local path, rather than model ID. This "
                "operation is forbidden as AutoModel.from_pretrained(...) was used with "
                "`allow_direct_local_storage_loading=False`. If you are running `inference-models` outside Roboflow "
                "hosted solutions - verify your setup. If you see this error on Roboflow platform - this "
                "feature was disabled for security reason. In rare cases when you use valid model ID, the "
                "clash of ID with local path may cause this error - we ask you to report the issue here: "
                "https://github.com/roboflow/inference/issues.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#directlocalstorageaccesserror",
            )
        return attempt_loading_model_from_local_storage(
            model_dir_or_weights_path=model_id_or_path,
            allow_local_code_packages=allow_local_code_packages,
            model_init_kwargs=model_init_kwargs,
            model_type=model_type,
            task_type=task_type,
            backend_type=backend,
        )


def attempt_loading_model_with_auto_load_cache(
    use_auto_resolution_cache: bool,
    auto_resolution_cache: AutoResolutionCache,
    auto_negotiation_hash: str,
    model_access_manager: ModelAccessManager,
    model_name_or_path: str,
    model_init_kwargs: dict,
    api_key: Optional[str],
    allow_loading_dependency_models: bool,
    forwarded_kwargs_values: Dict[str, Any],
    verbose: bool = False,
    weights_provider: str = "roboflow",
    max_package_loading_attempts: Optional[int] = None,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    allow_untrusted_packages: bool = False,
    trt_engine_host_code_allowed: bool = True,
    allow_local_code_packages: bool = True,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    allow_direct_local_storage_loading: bool = True,
    dependency_models_params: Optional[dict] = None,
    weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
    weights_provider_extra_headers: Optional[Dict[str, str]] = None,
) -> Optional[AnyModel]:
    if not use_auto_resolution_cache:
        return None
    verbose_info(
        message=f"Attempt to load model {model_name_or_path} using auto-load cache.",
        verbose_requested=verbose,
    )
    cache_entry = auto_resolution_cache.retrieve(
        auto_negotiation_hash=auto_negotiation_hash
    )
    if cache_entry is None:
        verbose_info(
            message=f"Could not find auto-load cache for model {model_name_or_path}.",
            verbose_requested=verbose,
        )
        return None
    if cache_entry.model_id != model_name_or_path:
        LOGGER.warning(
            "Ignoring auto-load cache entry for model %s while loading %s.",
            cache_entry.model_id,
            model_name_or_path,
        )
        return None
    if not allow_untrusted_packages and cache_entry.trusted_source is not True:
        verbose_info(
            message=(
                f"Auto-load cache for {model_name_or_path} does not prove that "
                "the selected package came from a trusted source."
            ),
            verbose_requested=verbose,
        )
        return None
    if not model_access_manager.is_model_package_access_granted(
        model_id=cache_entry.model_id,
        package_id=cache_entry.model_package_id,
        api_key=api_key,
    ):
        return None
    if not all_files_exist(files=cache_entry.resolved_files):
        verbose_info(
            message=f"Could not find all required files denoted in auto-load cache for model {model_name_or_path}.",
            verbose_requested=verbose,
        )
        return None
    try:
        model_dependencies = cache_entry.model_dependencies or []
        if model_dependencies and not allow_loading_dependency_models:
            raise CorruptedModelPackageError(
                message=f"Could not load model {cache_entry.model_id} as it defines another models which are "
                f"it's dependency, but the auto-loader prevents loading dependencies at certain "
                f"nesting depth to avoid excessive resolution procedure. This is a limitation of "
                f"current implementation. Provide us the context of your use-case to get help.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        model_dependencies_instances = {}
        dependency_models_params = dependency_models_params or {}
        for model_dependency in model_dependencies:
            dependency_params = dict(
                dependency_models_params.get(model_dependency.name, {})
            )
            dependency_params["model_id_or_path"] = model_dependency.model_id
            dependency_params["model_package_id"] = model_dependency.model_package_id
            resolved_model_parameters = prepare_dependency_model_parameters(
                model_parameters=dependency_params
            )

            for name, value in forwarded_kwargs_values.items():
                if name not in resolved_model_parameters.model_extra:
                    resolved_model_parameters.model_extra[name] = value
            verbose_info(
                message=f"Initialising dependent model: {model_dependency.model_id}",
                verbose_requested=verbose,
            )
            dependency_instance = AutoModel.from_pretrained(
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
                model_access_manager=model_access_manager,
                nms_fusion_preferences=resolved_model_parameters.nms_fusion_preferences,
                model_type=resolved_model_parameters.model_type,
                task_type=resolved_model_parameters.task_type,
                allow_loading_dependency_models=False,
                dependency_models_params=None,
                weights_provider_extra_query_params=weights_provider_extra_query_params,
                weights_provider_extra_headers=weights_provider_extra_headers,
                **resolved_model_parameters.kwargs,
            )
            model_dependencies_instances[model_dependency.name] = dependency_instance
        model_class = resolve_model_class(
            model_architecture=cache_entry.model_architecture,
            task_type=cache_entry.task_type,
            backend=cache_entry.backend_type,
            model_features=(
                set(cache_entry.model_features)
                if cache_entry.model_features
                else None
            ),
        )
        model_package_cache_dir = generate_model_package_cache_path(
            model_id=cache_entry.cache_model_id or cache_entry.model_id,
            package_id=cache_entry.model_package_id,
        )
        model_init_kwargs[MODEL_DEPENDENCIES_KEY] = model_dependencies_instances
        # Cache stores the already-resolved (package-vs-model) value written
        # in initialize_model — no need to re-run resolve_recommended_parameters.
        if cache_entry.recommended_parameters is not None:
            model_init_kwargs["recommended_parameters"] = (
                cache_entry.recommended_parameters
            )
        model = model_class.from_pretrained(
            model_package_cache_dir,
            **_prepare_library_model_init_kwargs(
                model_class=model_class,
                model_init_kwargs=model_init_kwargs,
            ),
        )
        verbose_info(
            message=f"Successfully loaded model {model_name_or_path} using auto-loading cache.",
            verbose_requested=verbose,
        )
        return model
    except Exception as error:
        LOGGER.warning(
            f"Encountered error {error} of type {type(error)} when attempted to load model using "
            f"auto-load cache. The resolution metadata is being preserved because model "
            f"initialization failures can be transient. Contact Roboflow submitting "
            f"issue under: https://github.com/roboflow/inference/issues/"
        )
        return None


def find_cached_model_package_dir(model_id: str) -> Optional[str]:
    """Return the path to a locally-cached model package for *model_id*, or ``None``.

    Scans the model's cache root under ``{INFERENCE_HOME}/models-cache/`` for any
    package directory that contains a ``model_config.json``. This is used as a
    fallback when the weights-provider API is unreachable (offline / air-gapped).
    """
    for package_dir in _iterate_cached_model_package_dirs(model_id=model_id):
        try:
            model_config = parse_model_config(
                config_path=os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
            )
        except Exception:
            continue
        if model_config.task_type is None:
            continue
        if not model_config.is_library_model() and (
            model_config.model_module is None or model_config.model_class is None
        ):
            continue
        return package_dir
    return None


def _iterate_cached_model_package_dirs(model_id: str) -> Generator[str, None, None]:
    # model_id may originate from request data - resolve both roots and make
    # sure scanned paths cannot escape the models cache (also guards against
    # symlinked cache entries pointing outside of it).
    models_cache_root = os.path.realpath(generate_models_cache_dir())
    try:
        cache_root = os.path.realpath(
            generate_model_cache_root_for_model_id(model_id=model_id)
        )
    except Exception:
        return
    if not cache_root.startswith(models_cache_root + os.sep):
        return
    if not os.path.isdir(cache_root):
        return
    try:
        entries = sorted(os.listdir(cache_root))
    except OSError:
        return
    for entry in entries:
        if entry.startswith("."):
            continue
        lexical_package_dir = os.path.join(cache_root, entry)
        if os.path.islink(lexical_package_dir):
            continue
        package_dir = os.path.realpath(lexical_package_dir)
        if not package_dir.startswith(cache_root + os.sep):
            continue
        if not os.path.isdir(package_dir):
            continue
        config_path = os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
        if os.path.islink(config_path) or not os.path.isfile(config_path):
            continue
        try:
            config_content = read_json(path=config_path)
        except (OSError, ValueError):
            continue
        if not isinstance(config_content, dict):
            continue
        cached_model_id = config_content.get("model_id")
        if cached_model_id is not None and cached_model_id != model_id:
            LOGGER.warning(
                f"Ignoring cached package at {package_dir} because its model ID "
                f"({cached_model_id}) does not match the requested model ID "
                f"({model_id})."
            )
            continue
        yield package_dir


def attempt_loading_model_from_offline_cache(
    model_id: str,
    model_init_kwargs: dict,
    requested_model_package_id: Optional[str] = None,
    requested_backends: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
    requested_batch_size: Optional[Union[int, Tuple[int, int]]] = None,
    requested_quantization: Optional[
        Union[str, Quantization, List[Union[str, Quantization]]]
    ] = None,
    model_access_manager: Optional[ModelAccessManager] = None,
    api_key: Optional[str] = None,
    allow_local_code_packages: bool = True,
    allow_untrusted_packages: bool = False,
    allow_loading_dependency_models: bool = True,
    dependency_models_params: Optional[dict] = None,
    forwarded_kwargs_values: Optional[Dict[str, Any]] = None,
    weights_provider: str = "roboflow",
    auto_resolution_cache: Optional[AutoResolutionCache] = None,
    use_auto_resolution_cache: bool = True,
    max_package_loading_attempts: Optional[int] = None,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    trt_engine_host_code_allowed: bool = True,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    allow_direct_local_storage_loading: bool = True,
    nms_fusion_preferences: Optional[Union[bool, dict]] = None,
    weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
    weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    offline_compatibility_hash: Optional[str] = None,
) -> Optional[Tuple[AnyModel, str]]:
    """Try to load a model from local cache when the API is unreachable.

    Scans the model's cache root for package directories containing
    ``model_config.json`` and attempts to load each until one succeeds.
    Returns ``(model, package_dir)`` on success, ``None`` if no cached
    package could be loaded.
    """
    found_any_package = False
    current_runtime_compatibility_hash = _runtime_compatibility_hash(
        runtime_x_ray=x_ray_runtime_environment()
    )
    candidates: Dict[str, Tuple[str, InferenceModelConfig, ModelPackageMetadata]] = {}
    for package_dir in _iterate_cached_model_package_dirs(model_id=model_id):
        package_id = os.path.basename(package_dir)
        if (
            requested_model_package_id is not None
            and package_id != requested_model_package_id
        ):
            continue
        if (
            model_access_manager is not None
            and not model_access_manager.is_model_package_access_granted(
                model_id=model_id,
                package_id=package_id,
                api_key=api_key,
            )
        ):
            continue
        try:
            package_config = parse_model_config(
                config_path=os.path.join(
                    package_dir,
                    MODEL_CONFIG_FILE_NAME,
                )
            )
        except Exception as error:
            LOGGER.warning(
                f"Failed to inspect cached model package from {package_dir}: {error}"
            )
            continue
        is_versioned_manifest = (
            package_config.offline_manifest_version
            == OFFLINE_CACHE_MANIFEST_VERSION
        )
        if not is_versioned_manifest and not allow_untrusted_packages:
            LOGGER.warning(
                "Ignoring legacy cached package %s because its trust provenance "
                "is unknown. Re-warm it with this inference-models version or "
                "explicitly set allow_untrusted_packages=True.",
                package_dir,
            )
            continue
        if (
            is_versioned_manifest
            and package_config.runtime_compatibility_hash
            != current_runtime_compatibility_hash
        ):
            LOGGER.warning(
                "Ignoring cached package %s because it was warmed in a different "
                "runtime environment.",
                package_dir,
            )
            continue
        if (
            offline_compatibility_hash is not None
            and package_config.offline_compatibility_hash
            != offline_compatibility_hash
        ):
            LOGGER.warning(
                "Ignoring cached package %s because it was not warmed for the "
                "current model-loading constraints.",
                package_dir,
            )
            continue
        if (
            not allow_untrusted_packages
            and package_config.trusted_source is not True
        ):
            continue
        if package_config.backend_type is None:
            continue
        if not model_implementation_exists(
            model_architecture=package_config.model_architecture,
            task_type=package_config.task_type,
            backend=package_config.backend_type,
            model_features=(
                set(package_config.model_features)
                if package_config.model_features
                else None
            ),
        ):
            continue
        try:
            package_quantization = (
                Quantization(package_config.quantization)
                if package_config.quantization is not None
                else Quantization.UNKNOWN
            )
        except ValueError:
            continue
        package_metadata = ModelPackageMetadata(
            package_id=package_id,
            backend=package_config.backend_type,
            package_artefacts=[],
            package_source=PackageSourceType.LOCAL_CACHE,
            quantization=package_quantization,
            dynamic_batch_size_supported=package_config.dynamic_batch_size_supported,
            static_batch_size=package_config.static_batch_size,
            trusted_source=package_config.trusted_source is True,
            model_features=package_config.model_features,
        )
        candidates[package_id] = (
            package_dir,
            package_config,
            package_metadata,
        )

    matching_packages = [candidate[2] for candidate in candidates.values()]
    if requested_model_package_id is None:
        feature_compatible_packages = []
        for package_metadata in matching_packages:
            package_config = candidates[package_metadata.package_id][1]
            try:
                compatible = [package_metadata]
                if requested_backends is not None:
                    compatible, _ = filter_model_packages_by_requested_backend(
                        model_packages=compatible,
                        requested_backends=requested_backends,
                        verbose=verbose,
                    )
                if requested_batch_size is not None:
                    compatible, _ = filter_model_packages_by_requested_batch_size(
                        model_packages=compatible,
                        requested_batch_size=requested_batch_size,
                        verbose=verbose,
                    )
                effective_quantization = requested_quantization
                default_quantization_used = False
                if effective_quantization is None:
                    default_quantization_used = True
                    effective_quantization = determine_default_allowed_quantization(
                        device=model_init_kwargs.get("device")
                    )
                if effective_quantization:
                    compatible, _ = (
                        filter_model_packages_by_requested_quantization(
                            model_packages=compatible,
                            requested_quantization=effective_quantization,
                            default_quantization_used=default_quantization_used,
                            verbose=verbose,
                        )
                    )
                compatible, _ = filter_model_packages_based_on_model_features(
                    model_packages=compatible,
                    nms_fusion_preferences=nms_fusion_preferences,
                    model_architecture=package_config.model_architecture,
                    task_type=package_config.task_type,
                )
                feature_compatible_packages.extend(compatible)
            except Exception as error:
                LOGGER.warning(
                    "Ignoring malformed offline package metadata in %s: %s",
                    candidates[package_metadata.package_id][0],
                    error,
                )
        matching_packages = rank_model_packages(
            model_packages=feature_compatible_packages,
            selected_device=model_init_kwargs.get("device"),
            nms_fusion_preferences=nms_fusion_preferences,
        )
    if max_package_loading_attempts is not None:
        matching_packages = matching_packages[:max_package_loading_attempts]

    dependency_models_params = dependency_models_params or {}
    forwarded_kwargs_values = forwarded_kwargs_values or {}
    for package_metadata in matching_packages:
        package_id = package_metadata.package_id
        package_dir, package_config, _ = candidates[package_id]
        found_any_package = True
        try:
            raw_dependencies = package_config.model_dependencies
            if (
                raw_dependencies is None
                and not allow_loading_dependency_models
            ):
                raise CorruptedModelPackageError(
                    message=(
                        f"Cannot verify whether cached package {package_id} "
                        "has dependencies while dependency loading is disabled."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            model_dependencies = [
                ModelDependency.model_validate(dependency)
                for dependency in (raw_dependencies or [])
            ]
            if model_dependencies and not allow_loading_dependency_models:
                raise CorruptedModelPackageError(
                    message=(
                        f"Cannot load cached package {package_id} because it "
                        "requires dependency models and dependency loading is disabled."
                    ),
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            dependency_instances = {}
            for dependency in model_dependencies:
                dependency_params = dict(
                    dependency_models_params.get(dependency.name, {})
                )
                dependency_params["model_id_or_path"] = dependency.model_id
                dependency_params["model_package_id"] = dependency.model_package_id
                resolved_parameters = prepare_dependency_model_parameters(
                    model_parameters=dependency_params
                )
                for name, value in forwarded_kwargs_values.items():
                    if name not in resolved_parameters.model_extra:
                        resolved_parameters.model_extra[name] = value
                dependency_instances[dependency.name] = AutoModel.from_pretrained(
                    model_id_or_path=resolved_parameters.model_id_or_path,
                    weights_provider=weights_provider,
                    api_key=api_key,
                    model_package_id=resolved_parameters.model_package_id,
                    backend=resolved_parameters.backend,
                    batch_size=resolved_parameters.batch_size,
                    quantization=resolved_parameters.quantization,
                    onnx_execution_providers=resolved_parameters.onnx_execution_providers,
                    device=resolved_parameters.device,
                    default_onnx_trt_options=resolved_parameters.default_onnx_trt_options,
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
                    model_access_manager=model_access_manager,
                    nms_fusion_preferences=resolved_parameters.nms_fusion_preferences,
                    model_type=resolved_parameters.model_type,
                    task_type=resolved_parameters.task_type,
                    allow_loading_dependency_models=False,
                    dependency_models_params=None,
                    weights_provider_extra_query_params=weights_provider_extra_query_params,
                    weights_provider_extra_headers=weights_provider_extra_headers,
                    **resolved_parameters.kwargs,
                )
            package_init_kwargs = dict(model_init_kwargs)
            package_init_kwargs[MODEL_DEPENDENCIES_KEY] = dependency_instances
            if package_config.recommended_parameters is not None:
                package_init_kwargs["recommended_parameters"] = (
                    RecommendedParameters.model_validate(
                        package_config.recommended_parameters
                    )
                )
            model = attempt_loading_model_from_local_storage(
                model_dir_or_weights_path=package_dir,
                allow_local_code_packages=allow_local_code_packages,
                model_init_kwargs=package_init_kwargs,
            )
            verbose_info(
                message=f"Loaded model {model_id} from offline cache at {package_dir}.",
                verbose_requested=verbose,
            )
            if model_access_manager is not None:
                model_access_manager.on_model_loaded(
                    model=model,
                    access_identifiers=AccessIdentifiers(
                        model_id=model_id,
                        package_id=package_id,
                        api_key=api_key,
                    ),
                    model_storage_path=package_dir,
                )
            return model, package_dir
        except Exception as error:
            LOGGER.warning(
                f"Failed to load cached model package from {package_dir}: {error}"
            )
    if not found_any_package:
        verbose_info(
            message=f"No offline cache packages found for model {model_id}.",
            verbose_requested=verbose,
        )
    else:
        verbose_info(
            message=f"No usable cached model package found for {model_id}.",
            verbose_requested=verbose,
        )
    return None


def all_files_exist(files: List[str]) -> bool:
    return all(os.path.exists(f) for f in files)


def _prepare_library_model_init_kwargs(
    model_class: Any, model_init_kwargs: dict
) -> dict:
    if not OFFLINE_MODE:
        return model_init_kwargs
    try:
        loader_parameters = inspect.signature(
            model_class.from_pretrained
        ).parameters
    except (TypeError, ValueError):
        return model_init_kwargs
    local_files_only_parameter = loader_parameters.get("local_files_only")
    if local_files_only_parameter is None or local_files_only_parameter.kind is (
        inspect.Parameter.POSITIONAL_ONLY
    ):
        return model_init_kwargs
    return {**model_init_kwargs, "local_files_only": True}


def attempt_loading_matching_model_packages(
    model_id: str,
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    matching_model_packages: List[ModelPackageMetadata],
    model_init_kwargs: dict,
    model_access_manager: ModelAccessManager,
    auto_resolution_cache: AutoResolutionCache,
    auto_negotiation_hash: str,
    api_key: Optional[str],
    model_dependencies: Optional[List[ModelDependency]],
    model_dependencies_instances: Dict[str, AnyModel],
    model_dependencies_directories: Dict[str, str],
    recommended_parameters: Optional[RecommendedParameters] = None,
    max_package_loading_attempts: Optional[int] = None,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    verbose: bool = True,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    use_auto_resolution_cache: bool = True,
    point_model_directory: Optional[Callable[[str], None]] = None,
    offline_compatibility_hash: Optional[str] = None,
) -> AnyModel:
    if max_package_loading_attempts is not None:
        matching_model_packages = matching_model_packages[:max_package_loading_attempts]
    if not matching_model_packages:
        raise NoModelPackagesAvailableError(
            message=f"Cannot load model {model_id} - no matching model package candidates for given model "
            f"running in this environment.",
            help_url="https://inference-models.roboflow.com/errors/package-negotiation/#nomodelpackagesavailableerror",
        )
    failed_load_attempts: List[Tuple[str, Exception]] = []
    for idx, model_package in enumerate(matching_model_packages):
        access_identifiers = AccessIdentifiers(
            model_id=model_id,
            package_id=model_package.package_id,
            api_key=api_key,
        )
        verbose_info(
            message=f"Attempt to load model package: {model_package.get_summary()}",
            verbose_requested=verbose,
        )
        try:
            model, model_package_cache_dir = initialize_model(
                model_id=model_id,
                model_architecture=model_architecture,
                task_type=task_type,
                model_package=model_package,
                model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                model_init_kwargs=model_init_kwargs,
                auto_resolution_cache=auto_resolution_cache,
                auto_negotiation_hash=auto_negotiation_hash,
                offline_compatibility_hash=offline_compatibility_hash,
                model_dependencies=model_dependencies,
                model_dependencies_instances=model_dependencies_instances,
                model_dependencies_directories=model_dependencies_directories,
                recommended_parameters=recommended_parameters,
                verify_hash_while_download=verify_hash_while_download,
                download_files_without_hash=download_files_without_hash,
                on_file_created=partial(
                    model_access_manager.on_file_created,
                    access_identifiers=access_identifiers,
                ),
                on_file_renamed=partial(
                    model_access_manager.on_file_renamed,
                    access_identifiers=access_identifiers,
                ),
                on_symlink_created=partial(
                    model_access_manager.on_symlink_created,
                    access_identifiers=access_identifiers,
                ),
                on_symlink_deleted=model_access_manager.on_symlink_deleted,
                use_auto_resolution_cache=use_auto_resolution_cache,
            )
            LOGGER.info(
                "Loaded model %s with backend %s (package %s)",
                model_id,
                model_package.backend.value,
                model_package.package_id,
            )
            model_access_manager.on_model_loaded(
                model=model,
                access_identifiers=access_identifiers,
                model_storage_path=model_package_cache_dir,
            )
            if point_model_directory:
                point_model_directory(model_package_cache_dir)
            return model
        except Exception as error:
            LOGGER.warning(
                f"Model package with id {model_package.package_id} that was selected to be loaded "
                f"failed to load with error: {error} of type {error.__class__.__name__}. This may "
                f"be caused several issues. If you see this warning after manually specifying model "
                f"package to be loaded - make sure that all required dependencies are installed. If "
                f"that warning is displayed when the model package was auto-selected - there is most "
                f"likely a bug in `inference-models` and you should raise an issue providing full context of "
                f"the event. https://github.com/roboflow/inference/issues"
            )
            next_idx = idx + 1
            if next_idx < len(matching_model_packages):
                next_backend = matching_model_packages[next_idx].backend.value
                LOGGER.warning(
                    "Falling back from %s to %s backend for model %s",
                    model_package.backend.value,
                    next_backend,
                    model_id,
                )
            failed_load_attempts.append((model_package.package_id, error))

    summary_of_errors = "\n".join(
        f"\t* model_package_id={model_package_id} error={error} error_type={error.__class__.__name__}"
        for model_package_id, error in failed_load_attempts
    )
    raise ModelPackageAlternativesExhaustedError(
        message=f"Could not load any of model package candidate for model {model_id}. This may "
        f"be caused several issues. If you see this warning after manually specifying model "
        f"package to be loaded - make sure that all required dependencies are installed. If "
        f"that warning is displayed when the model package was auto-selected - there is most "
        f"likely a bug in `inference-models` and you should raise an issue providing full context of "
        f"the event. https://github.com/roboflow/inference/issues\n\n"
        f"Here is the summary of errors for specific model packages:\n{summary_of_errors}\n\n",
        help_url="https://inference-models.roboflow.com/errors/model-loading/#modelpackagealternativesexhaustederror",
        alternatives_errors=[summary[1] for summary in failed_load_attempts],
    )


def initialize_model(
    model_id: str,
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    model_package: ModelPackageMetadata,
    model_init_kwargs: dict,
    auto_resolution_cache: AutoResolutionCache,
    auto_negotiation_hash: str,
    model_dependencies: Optional[List[ModelDependency]],
    model_dependencies_instances: Dict[str, AnyModel],
    model_dependencies_directories: Dict[str, str],
    recommended_parameters: Optional[RecommendedParameters] = None,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    on_file_created: Optional[Callable[[str], None]] = None,
    on_file_renamed: Optional[Callable[[str, str], None]] = None,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
    on_symlink_deleted: Optional[Callable[[str], None]] = None,
    use_auto_resolution_cache: bool = True,
    offline_compatibility_hash: Optional[str] = None,
) -> Tuple[AnyModel, str]:
    model_features = None
    if model_package.model_features:
        model_features = set(model_package.model_features.keys())
    model_class = resolve_model_class(
        model_architecture=model_architecture,
        task_type=task_type,
        backend=model_package.backend,
        model_features=model_features,
    )
    for artefact in model_package.package_artefacts:
        if artefact.file_handle == MODEL_CONFIG_FILE_NAME:
            raise CorruptedModelPackageError(
                message=f"For model with id=`{model_id}` and package={model_package.package_id} discovered "
                f"artefact named `{MODEL_CONFIG_FILE_NAME}` which collides with the config file that "
                f"inference is supposed to create for a model in order for compatibility with offline "
                f"loaders. This problem indicate a violation of model package contract and requires change in "
                f"model package structure. If you experience this issue using hosted Roboflow solution, contact "
                f"us to solve the problem.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    cache_model_id = model_package.cache_model_id or model_id
    model_package_cache_dir = generate_model_package_cache_path(
        model_id=cache_model_id,
        package_id=model_package.package_id,
    )
    os.makedirs(model_package_cache_dir, exist_ok=True)
    if model_package.package_source == PackageSourceType.LOCAL_CACHE:
        shared_files_mapping = _resolve_local_cache_package_files(
            model_package_cache_dir=model_package_cache_dir,
            package_artefacts=model_package.package_artefacts,
        )
        model_specific_files_mapping: Dict[str, str] = {}
        symlinks_mapping = {
            handle: os.path.join(model_package_cache_dir, handle)
            for handle in shared_files_mapping
        }
    else:
        files_specs = [
            (artefact.file_handle, artefact.download_url, artefact.md5_hash)
            for artefact in model_package.package_artefacts
            if isinstance(artefact, FileDownloadSpecs)
        ]
        file_specs_with_hash = [spec for spec in files_specs if spec[2] is not None]
        file_specs_without_hash = [spec for spec in files_specs if spec[2] is None]
        shared_blobs_dir = generate_shared_blobs_path()
        shared_files_mapping = download_files_to_directory(
            target_dir=shared_blobs_dir,
            files_specs=file_specs_with_hash,
            file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            verify_hash_while_download=verify_hash_while_download,
            download_files_without_hash=download_files_without_hash,
            name_after="md5_hash",
            on_file_created=on_file_created,
            on_file_renamed=on_file_renamed,
        )
        model_specific_files_mapping = download_files_to_directory(
            target_dir=model_package_cache_dir,
            files_specs=file_specs_without_hash,
            file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            verify_hash_while_download=verify_hash_while_download,
            download_files_without_hash=download_files_without_hash,
            on_file_created=on_file_created,
            on_file_renamed=on_file_renamed,
        )
        symlinks_mapping = create_symlinks_to_shared_blobs(
            model_dir=model_package_cache_dir,
            shared_files_mapping=shared_files_mapping,
            model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            on_symlink_created=on_symlink_created,
            on_symlink_deleted=on_symlink_deleted,
        )
    resolved_recommended_parameters = resolve_recommended_parameters(
        package_level=model_package.recommended_parameters,
        model_level=recommended_parameters,
    )
    config_path = os.path.join(model_package_cache_dir, MODEL_CONFIG_FILE_NAME)
    resolved_files = set(shared_files_mapping.values())
    resolved_files.update(model_specific_files_mapping.values())
    resolved_files.update(symlinks_mapping.values())
    dependencies_resolved_files = handle_dependencies_directories_creation(
        model_package_cache_dir=model_package_cache_dir,
        model_dependencies_directories=model_dependencies_directories,
        model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
        on_symlink_created=on_symlink_created,
        on_symlink_deleted=on_symlink_deleted,
    )
    resolved_files.update(dependencies_resolved_files)
    model_init_kwargs[MODEL_DEPENDENCIES_KEY] = model_dependencies_instances
    if resolved_recommended_parameters is not None:
        model_init_kwargs["recommended_parameters"] = resolved_recommended_parameters
    model = model_class.from_pretrained(
        model_package_cache_dir,
        **_prepare_library_model_init_kwargs(
            model_class=model_class,
            model_init_kwargs=model_init_kwargs,
        ),
    )
    # The versioned manifest is the marker that a package is eligible for raw
    # offline discovery.  Do not publish it until the package has initialized
    # successfully: a failed online candidate may leave downloaded artefacts
    # behind, but those partial artefacts must not be advertised as a warmed
    # offline package on the next restart.
    dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture=model_architecture,
        task_type=task_type,
        backend_type=model_package.backend,
        file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
        model_id=cache_model_id,
        on_file_created=on_file_created,
        model_features=model_package.model_features,
        trusted_source=model_package.trusted_source,
        model_dependencies=[
            dependency.model_dump(mode="json")
            for dependency in (model_dependencies or [])
        ],
        recommended_parameters=(
            resolved_recommended_parameters.model_dump(mode="json")
            if resolved_recommended_parameters is not None
            else None
        ),
        quantization=(
            model_package.quantization.value
            if model_package.quantization is not None
            else Quantization.UNKNOWN.value
        ),
        dynamic_batch_size_supported=model_package.dynamic_batch_size_supported,
        static_batch_size=model_package.static_batch_size,
        runtime_compatibility_hash=_runtime_compatibility_hash(
            runtime_x_ray=x_ray_runtime_environment()
        ),
        offline_compatibility_hash=offline_compatibility_hash,
    )
    resolved_files.add(config_path)
    dump_auto_resolution_cache(
        use_auto_resolution_cache=use_auto_resolution_cache,
        auto_resolution_cache=auto_resolution_cache,
        auto_negotiation_hash=auto_negotiation_hash,
        offline_compatibility_hash=offline_compatibility_hash,
        model_id=model_id,
        cache_model_id=cache_model_id,
        model_package_id=model_package.package_id,
        model_architecture=model_architecture,
        task_type=task_type,
        backend_type=model_package.backend,
        resolved_files=resolved_files,
        model_dependencies=model_dependencies,
        model_features=model_package.model_features,
        recommended_parameters=resolved_recommended_parameters,
        trusted_source=model_package.trusted_source,
    )
    return model, model_package_cache_dir


def create_symlinks_to_shared_blobs(
    model_dir: str,
    shared_files_mapping: Dict[FileHandle, str],
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
    on_symlink_deleted: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    # this function will not override existing files
    os.makedirs(model_dir, exist_ok=True)
    result = {}
    for file_handle, source_path in shared_files_mapping.items():
        link_name = os.path.join(model_dir, file_handle)
        target_path = shared_files_mapping[file_handle]
        result[file_handle] = link_name
        if os.path.exists(link_name) and (
            not os.path.islink(link_name) or os.path.realpath(link_name) == target_path
        ):
            continue
        handle_symlink_creation(
            target_path=target_path,
            link_name=link_name,
            model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            on_symlink_created=on_symlink_created,
            on_symlink_deleted=on_symlink_deleted,
        )
    return result


def handle_symlink_creation(
    target_path: str,
    link_name: str,
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
    on_symlink_deleted: Optional[Callable[[str], None]] = None,
) -> None:
    link_dir, link_file_name = os.path.split(os.path.abspath(link_name))
    os.makedirs(link_dir, exist_ok=True)
    lock_path = os.path.join(link_dir, f".{link_file_name}.lock")
    with FileLock(lock_path, timeout=model_download_file_lock_acquire_timeout):
        if os.path.islink(link_name):
            # file does not exist, but is link = broken symlink - we should purge
            os.remove(link_name)
            if on_symlink_deleted:
                on_symlink_deleted(link_name)
        elif os.path.exists(link_name):
            # regular file exists at link location - do not overwrite
            LOGGER.debug(
                f"Regular file already exists at {link_name}, skipping symlink creation."
            )
            return
        try:
            os.symlink(target_path, link_name)
            if on_symlink_created:
                on_symlink_created(target_path, link_name)
        except FileExistsError:
            # Another process created the file/link between our check and symlink call
            LOGGER.debug(
                f"Symlink target {link_name} was created by another process, skipping."
            )
            return


def dump_model_config_for_offline_use(
    config_path: str,
    model_architecture: Optional[ModelArchitecture],
    task_type: TaskType,
    backend_type: Optional[BackendType],
    file_lock_acquire_timeout: int,
    on_file_created: Optional[Callable[[str], None]] = None,
    model_id: Optional[str] = None,
    model_features: Optional[dict] = None,
    trusted_source: Optional[bool] = None,
    model_dependencies: Optional[List[dict]] = None,
    recommended_parameters: Optional[dict] = None,
    quantization: Optional[str] = None,
    dynamic_batch_size_supported: Optional[bool] = None,
    static_batch_size: Optional[int] = None,
    runtime_compatibility_hash: Optional[str] = None,
    offline_compatibility_hash: Optional[str] = None,
) -> None:
    """Persist a versioned manifest used for safe offline package loading."""
    target_file_dir, target_file_name = os.path.split(config_path)
    target_file_dir = target_file_dir or "."
    lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
    if os.path.islink(target_file_dir):
        raise CorruptedModelPackageError(
            message="Refusing to write model metadata through a symbolic link.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    os.makedirs(target_file_dir, exist_ok=True)
    if (
        os.path.islink(target_file_dir)
        or os.path.islink(config_path)
        or os.path.islink(lock_path)
    ):
        raise CorruptedModelPackageError(
            message="Refusing to write model metadata through a symbolic link.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    with FileLock(lock_path, timeout=file_lock_acquire_timeout):
        created = not os.path.exists(config_path)
        if os.path.exists(config_path):
            try:
                content = read_json(path=config_path)
            except ValueError:
                content = {}
            if not isinstance(content, dict):
                content = {}
        else:
            content = {}
        content.update(
            {
                "offline_manifest_version": OFFLINE_CACHE_MANIFEST_VERSION,
                "model_architecture": model_architecture,
                "task_type": task_type,
                "backend_type": backend_type,
                "model_features": model_features,
                "trusted_source": trusted_source,
                "model_dependencies": model_dependencies,
                "recommended_parameters": recommended_parameters,
                "quantization": quantization,
                "dynamic_batch_size_supported": dynamic_batch_size_supported,
                "static_batch_size": static_batch_size,
                "runtime_compatibility_hash": runtime_compatibility_hash,
                "offline_compatibility_hash": offline_compatibility_hash,
            }
        )
        existing_model_id = content.get("model_id")
        if model_id is not None and (
            not isinstance(existing_model_id, str) or not existing_model_id
        ):
            content["model_id"] = model_id

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=target_file_dir,
                prefix=f".{target_file_name}.",
                suffix=".tmp",
                delete=False,
            ) as file_handle:
                temporary_path = file_handle.name
                json.dump(content, file_handle)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            if os.path.islink(config_path):
                raise CorruptedModelPackageError(
                    message="Refusing to replace model metadata through a symbolic link.",
                    help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                )
            os.replace(temporary_path, config_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        if created and on_file_created:
            on_file_created(config_path)


def handle_dependencies_directories_creation(
    model_package_cache_dir: str,
    model_dependencies_directories: Dict[str, str],
    model_download_file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
    on_symlink_deleted: Optional[Callable[[str], None]] = None,
) -> Set[str]:
    resolved_files = set()
    if not model_dependencies_directories:
        return resolved_files
    for dependency_name, dependency_directory in model_dependencies_directories.items():
        dependency_files = scan_dependency_directory_for_resolved_files(
            dependency_directory=dependency_directory
        )
        resolved_files.update(dependency_files)
        dependencies_sub_dir = os.path.join(
            model_package_cache_dir, MODEL_DEPENDENCIES_SUB_DIR
        )
        target_dependency_dir = os.path.join(dependencies_sub_dir, dependency_name)
        os.makedirs(dependencies_sub_dir, exist_ok=True)
        dependency_lock_path = os.path.join(
            dependencies_sub_dir, f".{dependency_name}.lock"
        )
        with FileLock(
            dependency_lock_path, timeout=model_download_file_lock_acquire_timeout
        ):
            if os.path.exists(target_dependency_dir) and os.path.islink(
                target_dependency_dir
            ):
                os.remove(target_dependency_dir)
                if on_symlink_deleted:
                    on_symlink_deleted(target_dependency_dir)
            if not os.path.exists(target_dependency_dir):
                # Question: is it ok to only try to remove symlink and avoid doing anything else
                # if we encounter actual file / dir there?
                os.symlink(dependency_directory, target_dependency_dir)
                if on_symlink_created:
                    on_symlink_created(dependency_directory, target_dependency_dir)
    return resolved_files


def scan_dependency_directory_for_resolved_files(
    dependency_directory: str,
) -> List[str]:
    # we do not follow symlinks here, as the assumption is that we only support one level of nesting
    # for packages, wo when we have dependency - this model must not have dependencies, so
    # we will not encounter directories which are symlinks to be followed.
    results = []
    for current_dir, _, files in os.walk(dependency_directory):
        for file in files:
            if file.startswith(".") and file.endswith(".lock"):
                continue
            full_path = os.path.abspath(os.path.join(current_dir, file))
            results.append(full_path)
            if os.path.islink(full_path):
                results.append(os.readlink(full_path))
    return results


def dump_auto_resolution_cache(
    use_auto_resolution_cache: bool,
    auto_resolution_cache: AutoResolutionCache,
    auto_negotiation_hash: str,
    model_id: str,
    model_package_id: str,
    model_architecture: Optional[ModelArchitecture],
    task_type: TaskType,
    backend_type: Optional[BackendType],
    resolved_files: Set[str],
    model_dependencies: Optional[List[ModelDependency]],
    model_features: Optional[dict],
    recommended_parameters: Optional[RecommendedParameters] = None,
    cache_model_id: Optional[str] = None,
    trusted_source: Optional[bool] = None,
    offline_compatibility_hash: Optional[str] = None,
) -> None:
    if not use_auto_resolution_cache:
        return None
    cache_content = AutoResolutionCacheEntry(
        model_id=model_id,
        cache_model_id=cache_model_id,
        model_package_id=model_package_id,
        resolved_files=resolved_files,
        model_architecture=model_architecture,
        task_type=task_type,
        backend_type=backend_type,
        created_at=datetime.now(),
        model_dependencies=model_dependencies,
        model_features=model_features,
        recommended_parameters=recommended_parameters,
        offline_compatibility_hash=offline_compatibility_hash,
        trusted_source=trusted_source,
    )
    auto_resolution_cache.register(
        auto_negotiation_hash=auto_negotiation_hash, cache_entry=cache_content
    )


def _resolve_local_cache_package_files(
    model_package_cache_dir: str,
    package_artefacts: List[LocalFileArtefactSpecs],
) -> Dict[str, str]:
    shared_blobs_dir = generate_shared_blobs_path()
    shared_files_mapping: Dict[str, str] = {}
    for artefact in package_artefacts:
        if not isinstance(artefact, LocalFileArtefactSpecs):
            raise CorruptedModelPackageError(
                message=(
                    "Local cache model package contains non-local artefact specs. "
                    "All artefacts must be LocalFileArtefactSpecs."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if not is_valid_md5_hash(artefact.md5_hash):
            raise CorruptedModelPackageError(
                message=(
                    f"Local cache model package artefact `{artefact.file_handle}` has an "
                    f"invalid md5 hash `{artefact.md5_hash}`."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if artefact.file_handle != os.path.basename(artefact.file_handle):
            raise CorruptedModelPackageError(
                message=(
                    f"Local cache model package artefact `{artefact.file_handle}` has an "
                    f"unsafe file handle."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        package_file_path = os.path.join(model_package_cache_dir, artefact.file_handle)
        if not os.path.isfile(package_file_path):
            raise CorruptedModelPackageError(
                message=(
                    f"Local cache model package is missing artefact `{artefact.file_handle}` "
                    f"at `{package_file_path}`."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        # File content md5 is verified during discovery (discover_local_trt_packages),
        # so we only validate presence and hash format here to avoid re-hashing the
        # full engine on the load path.
        shared_files_mapping[artefact.file_handle] = os.path.join(
            shared_blobs_dir, artefact.md5_hash
        )
    return shared_files_mapping


def attempt_loading_model_from_local_storage(
    model_dir_or_weights_path: str,
    allow_local_code_packages: bool,
    model_init_kwargs: dict,
    model_type: Optional[str] = None,
    task_type: Optional[str] = None,
    backend_type: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
) -> AnyModel:
    if os.path.isfile(model_dir_or_weights_path):
        return attempt_loading_model_from_checkpoint(
            checkpoint_path=model_dir_or_weights_path,
            model_init_kwargs=model_init_kwargs,
            model_type=model_type,
            task_type=task_type,
            backend_type=backend_type,
        )
    config_path = os.path.join(model_dir_or_weights_path, MODEL_CONFIG_FILE_NAME)
    model_config = parse_model_config(config_path=config_path)
    if model_config.is_library_model():
        return load_library_model_from_local_dir(
            model_dir=model_dir_or_weights_path,
            model_config=model_config,
            model_init_kwargs=model_init_kwargs,
        )
    if not allow_local_code_packages:
        raise ForbiddenLocalCodePackageAccessError(
            message=f"Attempted to load model from local package with arbitrary code. This is not allowed in "
            f"this environment. To let inference loading such models, use `allow_local_code_packages=True` "
            f"parameter of `AutoModel.from_pretrained(...)`. If you see this error while using one of Roboflow "
            f"hosted solution - contact us to solve the problem.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#forbiddenlocalcodepackageaccesserror",
        )
    return load_model_from_local_package_with_arbitrary_code(
        model_dir=model_dir_or_weights_path,
        model_config=model_config,
        model_init_kwargs=model_init_kwargs,
    )


def attempt_loading_model_from_checkpoint(
    checkpoint_path: str,
    model_init_kwargs: dict,
    model_type: Optional[str] = None,
    task_type: Optional[str] = None,
    backend_type: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
) -> AnyModel:
    model_architecture, task_type, backend_type = resolve_models_registry_entry(
        model_type=model_type,
        task_type=task_type,
        backend_type=backend_type,
    )
    model_init_kwargs["model_type"] = model_type
    model_class = resolve_model_class(
        model_architecture=model_architecture,
        task_type=task_type,
        backend=backend_type,
    )
    return model_class.from_pretrained(
        checkpoint_path,
        **_prepare_library_model_init_kwargs(
            model_class=model_class,
            model_init_kwargs=model_init_kwargs,
        ),
    )


def resolve_models_registry_entry(
    model_type: Optional[str],
    task_type: Optional[str] = None,
    backend_type: Optional[
        Union[str, BackendType, List[Union[str, BackendType]]]
    ] = None,
) -> Tuple[str, str, BackendType]:
    #  TODO: in the future this check will grow in size
    if not model_type:
        raise MissingModelInitParameterError(
            message="When loading model directly from checkpoint path, `model_type` parameter must be specified. "
            "Use one of the supported value, for example `rfdetr-nano` in case you refer checkpoint of "
            "RFDetr Nano model.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#missingmodelinitparametererror",
        )
    if model_type not in MODEL_TYPES_TO_LOAD_FROM_CHECKPOINT:
        raise InvalidModelInitParameterError(
            message="When loading model directly from checkpoint path, `model_type` parameter must define "
            "one of the type of model that support loading directly from the checkpoints. "
            f"Models supported in current version: {MODEL_TYPES_TO_LOAD_FROM_CHECKPOINT}",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
        )
    # a bit of hard coding here, over time we must maintain
    model_architecture = "rfdetr"
    if task_type is None:
        if model_type == "rfdetr-seg-preview":
            task_type = INSTANCE_SEGMENTATION_TASK
        else:
            task_type = OBJECT_DETECTION_TASK
    if task_type not in {OBJECT_DETECTION_TASK, INSTANCE_SEGMENTATION_TASK}:
        raise InvalidModelInitParameterError(
            message=f"When loading model directly from checkpoint path, set `model_type` as {model_type} and "
            f"`task_type` as {task_type}, whereas selected model do only support `{OBJECT_DETECTION_TASK}` "
            f"task while loading from checkpoint file.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
        )
    if backend_type is None:
        backend_type = BackendType.TORCH
    if isinstance(backend_type, list) and len(backend_type) != 1:
        if len(backend_type) != 1:
            raise InvalidModelInitParameterError(
                message=f"When loading model directly from checkpoint path, set `backend` parameter to be {backend_type}, "
                f"whereas it is only supported to pass a single value.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
            )
        backend_type = backend_type[0]
    if isinstance(backend_type, str):
        backend_type = parse_backend_type(value=backend_type)
    if backend_type is not BackendType.TORCH:
        raise InvalidModelInitParameterError(
            message=f"When loading model directly from checkpoint path, selected the following backend {backend_type}, "
            f"but the backend supported for model {model_type} is {BackendType.TORCH}",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
        )
    return model_architecture, task_type, backend_type


def parse_model_config(config_path: str) -> InferenceModelConfig:
    if not os.path.isfile(config_path):
        raise CorruptedModelPackageError(
            message=f"Could not find model config while attempting to load model from "
            f"local directory. This error may be caused by misconfiguration of model package (lack of config "
            f"file), as well as by clash between model_id or model alias and contents of local disc drive which "
            f"is possible when you have local directory in current dir which has the name colliding with the "
            f"model you attempt to load. If your intent was to load model from remote backend (not local "
            f"storage) - verify the contents of $PWD. If you see this problem while using one of Roboflow "
            f"hosted solutions - contact us to get help.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    try:
        raw_config = read_json(path=config_path)
    except ValueError as error:
        raise CorruptedModelPackageError(
            message=f"Could not decode model config while attempting to load model from "
            f"local directory. This error may be caused by corrupted config file. Validate the content of your "
            f"model package and check in documentation the required format of model config file. "
            f"If you see this problem while using one of Roboflow hosted solutions - contact us to get help.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error
    if not isinstance(raw_config, dict):
        raise CorruptedModelPackageError(
            message=f"While loading the model from local directory encountered corrupted model config file - config is "
            f"supposed to be a dictionary, instead decoded object of type: "
            f"{type(raw_config)}. If you see this problem while using one of Roboflow hosted solutions - "
            f"contact us to get help. Otherwise - verify the content of your model config.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    backend_type = None
    if "backend_type" in raw_config:
        raw_backend_type = raw_config["backend_type"]
        try:
            backend_type = BackendType(raw_backend_type)
        except (TypeError, ValueError) as e:
            raise CorruptedModelPackageError(
                message=f"While loading the model from local directory encountered corrupted model config "
                f"- declared `backend_type` ({raw_backend_type}) is not supported by inference. "
                f"Supported values: {list(t.value for t in BackendType)}. If you see this problem while using "
                f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify the content "
                f"of your model config.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from e
    model_features = raw_config.get("model_features")
    model_dependencies = raw_config.get("model_dependencies")
    recommended_parameters = raw_config.get("recommended_parameters")
    trusted_source = raw_config.get("trusted_source")
    optional_string_fields = (
        "model_architecture",
        "task_type",
        "model_module",
        "model_class",
        "quantization",
        "runtime_compatibility_hash",
        "offline_compatibility_hash",
    )
    for field_name in optional_string_fields:
        field_value = raw_config.get(field_name)
        if field_value is not None and not isinstance(field_value, str):
            raise CorruptedModelPackageError(
                message=f"Cached model config contains invalid {field_name} metadata.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    for hash_field_name in (
        "runtime_compatibility_hash",
        "offline_compatibility_hash",
    ):
        hash_field_value = raw_config.get(hash_field_name)
        if hash_field_value is not None and re.fullmatch(
            r"[0-9a-f]{64}", hash_field_value
        ) is None:
            raise CorruptedModelPackageError(
                message=(
                    f"Cached model config contains invalid {hash_field_name} "
                    "metadata."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
    if model_features is not None and not isinstance(model_features, dict):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid model_features metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if model_dependencies is not None and (
        not isinstance(model_dependencies, list)
        or not all(isinstance(item, dict) for item in model_dependencies)
    ):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid model_dependencies metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if recommended_parameters is not None and not isinstance(
        recommended_parameters, dict
    ):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid recommended_parameters metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if trusted_source is not None and not isinstance(trusted_source, bool):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid trusted_source metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    dynamic_batch_size_supported = raw_config.get(
        "dynamic_batch_size_supported"
    )
    if dynamic_batch_size_supported is not None and not isinstance(
        dynamic_batch_size_supported, bool
    ):
        raise CorruptedModelPackageError(
            message=(
                "Cached model config contains invalid "
                "dynamic_batch_size_supported metadata."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    static_batch_size = raw_config.get("static_batch_size")
    if static_batch_size is not None and (
        not isinstance(static_batch_size, int)
        or isinstance(static_batch_size, bool)
        or static_batch_size < 1
    ):
        raise CorruptedModelPackageError(
            message="Cached model config contains invalid static_batch_size metadata.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    offline_manifest_version = raw_config.get("offline_manifest_version")
    if offline_manifest_version is not None and (
        not isinstance(offline_manifest_version, int)
        or isinstance(offline_manifest_version, bool)
    ):
        raise CorruptedModelPackageError(
            message=(
                "Cached model config contains invalid offline_manifest_version "
                "metadata."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return InferenceModelConfig(
        model_architecture=raw_config.get("model_architecture"),
        task_type=raw_config.get("task_type"),
        backend_type=backend_type,
        model_module=raw_config.get("model_module"),
        model_class=raw_config.get("model_class"),
        model_features=model_features,
        trusted_source=trusted_source,
        model_dependencies=model_dependencies,
        recommended_parameters=recommended_parameters,
        quantization=raw_config.get("quantization"),
        dynamic_batch_size_supported=dynamic_batch_size_supported,
        static_batch_size=static_batch_size,
        runtime_compatibility_hash=raw_config.get(
            "runtime_compatibility_hash"
        ),
        offline_compatibility_hash=raw_config.get(
            "offline_compatibility_hash"
        ),
        offline_manifest_version=offline_manifest_version,
    )


def load_library_model_from_local_dir(
    model_dir: str,
    model_config: InferenceModelConfig,
    model_init_kwargs: dict,
) -> AnyModel:
    model_class = resolve_model_class(
        model_architecture=model_config.model_architecture,
        task_type=model_config.task_type,
        backend=model_config.backend_type,
        model_features=(
            set(model_config.model_features)
            if model_config.model_features
            else None
        ),
    )
    return model_class.from_pretrained(
        model_dir,
        **_prepare_library_model_init_kwargs(
            model_class=model_class,
            model_init_kwargs=model_init_kwargs,
        ),
    )


def load_model_from_local_package_with_arbitrary_code(
    model_dir: str,
    model_config: InferenceModelConfig,
    model_init_kwargs: dict,
) -> AnyModel:
    if model_config.model_module is None or model_config.model_class is None:
        raise CorruptedModelPackageError(
            message=f"While loading the model from local directory encountered corrupted model config file. "
            f"Config does not specify neither `model_module` name nor `model_class`, which are both  "
            f"required to load models provided with arbitrary code. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify the content "
            f"of your model config.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    model_module_path = os.path.join(model_dir, model_config.model_module)
    if not os.path.isfile(model_module_path):
        raise CorruptedModelPackageError(
            message=f"While loading the model from local directory encountered corrupted model config file. "
            f"Config pointed module {model_config.model_module}, but there is no file under "
            f"{model_module_path}. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify the content "
            f"of your model config.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    model_class = load_class_from_path(
        module_path=model_module_path, class_name=model_config.model_class
    )
    return model_class.from_pretrained(model_dir, **model_init_kwargs)


def load_class_from_path(module_path: str, class_name: str) -> AnyModel:
    if not os.path.exists(module_path):
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            "Could find the module under the path specified in model config. If you see this problem "
            f"while using one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            "Could not build module specification. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None or not hasattr(loader, "exec_module"):
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            "Could not execute module loader. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    try:
        loader.exec_module(module)
    except Exception as error:
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue executing the module code "
            f"to retrieve model class. Details of the error: {error}. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if not hasattr(module, class_name):
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            f"Module `{module_name}` has no class `{class_name}`. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment. It may also be the case that configuration file of the model points "
            f"to invalid class name.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return getattr(module, class_name)


def resolve_recommended_parameters(
    package_level: Optional[RecommendedParameters],
    model_level: Optional[RecommendedParameters],
) -> Optional[RecommendedParameters]:
    """Package-level recommended_parameters take priority over model-level."""
    return package_level if package_level is not None else model_level
