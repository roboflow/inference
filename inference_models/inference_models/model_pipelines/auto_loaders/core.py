from typing import Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.tree import Tree

from inference_models.errors import ModelPipelineInitializationError
from inference_models.logger import verbose_info
from inference_models.model_pipelines.auto_loaders.pipelines_registry import (
    REGISTERED_PIPELINES,
    get_default_pipeline_parameters,
    resolve_pipeline_class,
)
from inference_models.models.auto_loaders.access_manager import ModelAccessManager
from inference_models.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCache,
)
from inference_models.models.auto_loaders.core import AutoModel
from inference_models.models.auto_loaders.dependency_models import (
    DependencyModelParameters,
    prepare_dependency_model_parameters,
)
from inference_models.models.auto_loaders.entities import AnyModel


class AutoModelPipeline:

    @classmethod
    def list_available_pipelines(cls) -> None:
        """Display all registered model pipelines available for loading.

        Shows a tree view of all pipeline identifiers that can be used with
        `AutoModelPipeline.from_pretrained()`. Pipelines are multi-model workflows
        that combine multiple models to solve complex computer vision tasks.

        Returns:
            None. Prints a formatted tree to the console showing all registered
            pipeline identifiers.

        Examples:
            List available pipelines:

            >>> from inference_models import AutoModelPipeline
            >>> AutoModelPipeline.list_available_pipelines()
            # Displays:
            # Available Model Pipelines:
            # ├── face-and-gaze-detection
            # └── ... (other registered pipelines)

        See Also:
            - `AutoModelPipeline.from_pretrained()`: Load a pipeline
        """
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
            List[Optional[Union[str, dict, DependencyModelParameters]]]
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
        model_access_manager: Optional[ModelAccessManager] = None,
        weights_provider_extra_query_params: Optional[List[Tuple[str, str]]] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AnyModel:
        """Load and initialize a multi-model pipeline.

        Pipelines are pre-configured workflows that combine multiple models to solve
        complex computer vision tasks. For example, the "face-and-gaze-detection"
        pipeline combines a face detector with a gaze estimation model.

        Each pipeline has default model configurations, but you can override them by
        providing custom `models_parameters`. All models in the pipeline are loaded
        using `AutoModel.from_pretrained()` with the same loading parameters.

        Args:
            pipline_id: Pipeline identifier. Use `list_available_pipelines()` to see
                available options. Examples: "face-and-gaze-detection".

            models_parameters: Optional list of parameters for each model in the pipeline.
                Can be:
                - None (default): Use pipeline's default models
                - List of model IDs (strings): ["model1-id", "model2-id"]
                - List of parameter dicts: [{"model_id_or_path": "model1", "device": "cuda"}, ...]
                - List of DependencyModelParameters objects (advanced)
                - Mix of the above (None entries use defaults)

                The list length should match the number of models in the pipeline.

            weights_provider: Source for model weights. Default: "roboflow".

            api_key: Roboflow API key for accessing private models. If not provided,
                uses the `ROBOFLOW_API_KEY` environment variable.

            max_package_loading_attempts: Maximum number of model packages to try before
                failing. Default: Try all matching packages.

            verbose: Enable detailed logging during pipeline and model loading.
                Default: False.

            model_download_file_lock_acquire_timeout: Timeout in seconds for acquiring
                file locks during concurrent downloads. Default: 10.

            allow_untrusted_packages: Allow loading model packages with custom code that
                haven't been verified. **Security risk**. Default: False.

            trt_engine_host_code_allowed: Allow TensorRT engines to execute host code.
                Default: True.

            allow_local_code_packages: Allow loading models with custom Python code from
                local directories. Default: True.

            verify_hash_while_download: Verify file integrity using checksums during
                download. Default: True.

            download_files_without_hash: Allow downloading files without checksums.
                **Security risk**. Default: False.

            use_auto_resolution_cache: Enable caching of model resolution results.
                Default: True.

            auto_resolution_cache: Custom cache implementation. Advanced usage only.

            allow_direct_local_storage_loading: Allow loading models directly from local
                paths. Default: True.

            model_access_manager: Custom model access control manager. Advanced usage only.

            weights_provider_extra_query_params: Extra query parameters to pass to the weights' provider. Advanced
                usage only.

            weights_provider_extra_headers: Extra headers to pass to the weights' provider. Advanced
                usage only.

            **kwargs: Additional pipeline-specific parameters passed to the pipeline's
                `with_models()` method.

        Returns:
            Initialized pipeline instance. The specific type depends on the pipeline.

        Raises:
            ModelPipelineInitializationError: If pipeline initialization fails, models
                cannot be loaded, or required parameters are missing.
            UnauthorizedModelAccessError: If API key is invalid or model access is denied.

        Examples:
            Load pipeline with default models:

            >>> from inference_models import AutoModelPipeline
            >>> pipeline = AutoModelPipeline.from_pretrained("face-and-gaze-detection")
            >>> results = pipeline(image)

            Load pipeline with custom model parameters:

            >>> pipeline = AutoModelPipeline.from_pretrained(
            ...     "face-and-gaze-detection",
            ...     models_parameters=[
            ...         "mediapipe/face-detector",  # Use specific face detector
            ...         {"model_id_or_path": "l2cs-net/rn50", "device": "cuda"}  # Custom gaze model
            ...     ]
            ... )

            Load with verbose logging:

            >>> pipeline = AutoModelPipeline.from_pretrained(
            ...     "face-and-gaze-detection",
            ...     verbose=True
            ... )

        See Also:
            - `AutoModelPipeline.list_available_pipelines()`: List all available pipelines
            - `AutoModel.from_pretrained()`: Load individual models
        """
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
                help_url="https://inference-models.roboflow.com/errors/model-loading/#modelpipelineinitializationerror",
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
            resolved_model_parameters = prepare_dependency_model_parameters(
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
                model_access_manager=model_access_manager,
                nms_fusion_preferences=resolved_model_parameters.nms_fusion_preferences,
                model_type=resolved_model_parameters.model_type,
                task_type=resolved_model_parameters.task_type,
                weights_provider_extra_query_params=weights_provider_extra_query_params,
                weights_provider_extra_headers=weights_provider_extra_headers,
                **resolved_model_parameters.kwargs,
            )
            models.append(model)
        return pipeline_class.with_models(models, **kwargs)
