from typing import Callable, Dict, Optional

from inference_models.errors import ModelRetrievalError
from inference_models.weights_providers.entities import ModelMetadata
from inference_models.weights_providers.roboflow import get_roboflow_model

ModelId = str
ApiKey = Optional[str]
WeightsProvider = Callable[[ModelId, ApiKey, ...], ModelMetadata]

WEIGHTS_PROVIDERS: Dict[str, WeightsProvider] = {  # type: ignore
    "roboflow": get_roboflow_model,
}


def get_model_from_provider(
    model_id: ModelId, provider: str, api_key: ApiKey = None, **kwargs
) -> ModelMetadata:
    """Retrieve model metadata from a registered weights provider.

    Fetches model metadata (including available packages, download URLs, and
    configuration) from a registered weights provider. The default provider is
    "roboflow", but custom providers can be registered using `register_model_provider()`.

    Args:
        model_id: Model identifier. Format depends on the provider:
            - Roboflow: "workspace/project/version" or "model-id"
            - Custom providers: Provider-specific format

        provider: Name of the weights provider to use. Default provider is "roboflow".

        api_key: API key for authentication. Required for private models on Roboflow.
            If not provided, uses the `ROBOFLOW_API_KEY` environment variable.

        **kwargs: Additional provider-specific parameters.

    Returns:
        ModelMetadata object containing:
            - model_id: Canonical model identifier
            - model_packages: List of available model packages with different backends
            - dependencies: Model dependencies and requirements
            - Additional provider-specific metadata

    Raises:
        ModelRetrievalError: If the provider is not registered or model retrieval fails.
        UnauthorizedModelAccessError: If API key is invalid or model access is denied.

    Examples:
        Get model metadata from Roboflow:

        >>> from inference_models.developer_tools import get_model_from_provider
        >>>
        >>> metadata = get_model_from_provider(
        ...     model_id="yolov8n-640",
        ...     provider="roboflow"
        ... )
        >>>
        >>> print(f"Model ID: {metadata.model_id}")
        >>> print(f"Available packages: {len(metadata.model_packages)}")
        >>> for package in metadata.model_packages:
        ...     print(f"  - {package.backend_type} ({package.quantization})")

        Get private model with API key:

        >>> metadata = get_model_from_provider(
        ...     model_id="my-workspace/my-project/2",
        ...     provider="roboflow",
        ...     api_key="your_api_key_here"
        ... )

        Use with custom provider:

        >>> # After registering a custom provider
        >>> metadata = get_model_from_provider(
        ...     model_id="custom-model-id",
        ...     provider="my_custom_provider",
        ...     custom_param="value"
        ... )

    See Also:
        - `register_model_provider()`: Register a custom weights provider
        - `AutoModel.from_pretrained()`: Load models using the provider system
    """
    if provider not in WEIGHTS_PROVIDERS:
        raise ModelRetrievalError(
            message=f"Requested model to be retrieved using '{provider}' provider which is not implemented.",
            help_url="https://todo",
        )
    return WEIGHTS_PROVIDERS[provider](model_id, api_key, **kwargs)


def register_model_provider(
    provider_name: str, provider_handler: WeightsProvider
) -> None:
    """Register a custom weights provider for model retrieval.

    Allows registration of custom model providers that can be used with
    `AutoModel.from_pretrained()` and `get_model_from_provider()`. This enables
    loading models from custom sources beyond the default Roboflow provider.

    Args:
        provider_name: Unique name for the provider. Will be used as the
            `weights_provider` parameter in `AutoModel.from_pretrained()`.

        provider_handler: Callable that implements the provider interface.
            Must accept (model_id: str, api_key: Optional[str], **kwargs) and
            return a ModelMetadata object.

    Examples:
        Register a custom provider:

        >>> from inference_models.developer_tools import (
        ...     register_model_provider,
        ...     ModelMetadata,
        ...     ModelPackageMetadata,
        ...     ONNXPackageDetails
        ... )
        >>> from inference_models import BackendType, Quantization
        >>>
        >>> def my_custom_provider(model_id: str, api_key: str = None, **kwargs):
        ...     # Fetch model metadata from your custom source
        ...     return ModelMetadata(
        ...         model_id=model_id,
        ...         model_packages=[
        ...             ModelPackageMetadata(
        ...                 backend_type=BackendType.ONNX,
        ...                 quantization=Quantization.FP32,
        ...                 package_details=ONNXPackageDetails(
        ...                     download_url="https://my-server.com/model.onnx",
        ...                     md5_hash="abc123...",
        ...                     # ... other details
        ...                 ),
        ...                 # ... other metadata
        ...             )
        ...         ],
        ...         dependencies=[]
        ...     )
        >>>
        >>> # Register the provider
        >>> register_model_provider("my_provider", my_custom_provider)
        >>>
        >>> # Now use it with AutoModel
        >>> from inference_models import AutoModel
        >>> model = AutoModel.from_pretrained(
        ...     "my-model-id",
        ...     weights_provider="my_provider"
        ... )

        Register a provider for local file system:

        >>> def local_file_provider(model_id: str, api_key: str = None, **kwargs):
        ...     base_path = kwargs.get("base_path", "/models")
        ...     model_path = f"{base_path}/{model_id}"
        ...
        ...     # Return metadata pointing to local files
        ...     return ModelMetadata(
        ...         model_id=model_id,
        ...         model_packages=[...],  # Configure packages
        ...         dependencies=[]
        ...     )
        >>>
        >>> register_model_provider("local", local_file_provider)
        >>>
        >>> model = AutoModel.from_pretrained(
        ...     "yolov8n",
        ...     weights_provider="local",
        ...     base_path="/my/models"
        ... )

    Note:
        - Provider handlers must return a `ModelMetadata` object
        - The provider name must be unique (will override existing providers)
        - Provider handlers should handle authentication and error cases

    See Also:
        - `get_model_from_provider()`: Retrieve models using registered providers
        - `ModelMetadata`: Structure for model metadata
        - `ModelPackageMetadata`: Structure for package metadata
    """
    WEIGHTS_PROVIDERS[provider_name] = provider_handler
