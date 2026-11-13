"""Read an action recognition model's contract without downloading its weights."""

import tempfile
from typing import List, Optional, Tuple

from inference_models.errors import (
    ModelImplementationNotFoundError,
    ModelRetrievalError,
)
from inference_models.models.auto_loaders.models_registry import (
    ACTION_RECOGNITION_TASK,
    model_implementation_exists,
    resolve_model_class,
)
from inference_models.models.base.action_recognition import (
    ActionRecognitionContract,
    ActionRecognitionModel,
)
from inference_models.utils.download import download_files_to_directory
from inference_models.weights_providers.core import get_model_from_provider
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    ModelMetadata,
    ModelPackageMetadata,
)


def get_action_recognition_contract(
    model_id: str,
    *,
    api_key: Optional[str] = None,
    weights_provider: str = "roboflow",
) -> ActionRecognitionContract:
    """Read what a caller must know before an action recognition model's first call.

    Resolves the model class the auto-loader would serve the model with, then
    downloads only the small files that class reads its contract from (for
    example ``inference_config.json`` and ``class_names.txt``) and parses them
    with the class's own ``contract_from_package``. Weights are never
    downloaded, so this suits a client that runs the model elsewhere.

    Args:
        model_id (str): Model identifier, for example ``"my-project/3"``.
        api_key (Optional[str]): Key with access to the model.
        weights_provider (str): Weights provider to read package metadata from.

    Returns:
        ActionRecognitionContract: Classes, sampling and span conventions.

    Raises:
        ModelImplementationNotFoundError: No registered action recognition
            implementation serves the model's architecture.
        ModelRetrievalError: A contract file cannot be downloaded.
        CorruptedModelPackageError: The package files are invalid.

    Examples:
        >>> from inference_models import get_action_recognition_contract
        >>> contract = get_action_recognition_contract("my-project/3", api_key="...")
        >>> contract.video_sampling.sample_fps
        12.0
    """
    metadata = get_model_from_provider(
        model_id=model_id, provider=weights_provider, api_key=api_key
    )
    model_class, package = _resolve_action_recognition_package(metadata=metadata)

    with tempfile.TemporaryDirectory() as package_dir:
        download_files_to_directory(
            target_dir=package_dir,
            files_specs=_contract_file_specs(
                package=package, contract_files=model_class.contract_files
            ),
            verbose=False,
            download_files_without_hash=True,
        )
        contract = model_class.contract_from_package(package_dir=package_dir)

    return contract


def _resolve_action_recognition_package(
    metadata: ModelMetadata,
) -> Tuple[type, ModelPackageMetadata]:
    # Runtime negotiation is skipped on purpose: the caller does not run the
    # model, so its own hardware must not rule packages out.
    for package in metadata.model_packages:
        if not model_implementation_exists(
            model_architecture=metadata.model_architecture,
            task_type=ACTION_RECOGNITION_TASK,
            backend=package.backend,
        ):
            continue
        model_class = resolve_model_class(
            model_architecture=metadata.model_architecture,
            task_type=ACTION_RECOGNITION_TASK,
            backend=package.backend,
        )
        if issubclass(model_class, ActionRecognitionModel):
            return model_class, package

    raise ModelImplementationNotFoundError(
        message=(
            f"Model {metadata.model_id} has architecture "
            f"{metadata.model_architecture!r} and no package that a registered "
            f"action recognition implementation serves."
        ),
        help_url="https://inference-models.roboflow.com/errors/model-loading/#modelimplementationnotfounderror",
    )


def _contract_file_specs(
    package: ModelPackageMetadata, contract_files: Tuple[str, ...]
) -> List[Tuple[str, str, Optional[str]]]:
    specs = []
    for artefact in package.package_artefacts:
        if artefact.file_handle not in contract_files:
            continue
        if not isinstance(artefact, FileDownloadSpecs):
            # Skipping it would read a default contract the package overrides.
            raise ModelRetrievalError(
                message=(
                    f"Package {package.package_id} holds {artefact.file_handle} "
                    f"only as a local file, so its contract cannot be read remotely."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-retrieval/#modelretrievalerror",
            )
        specs.append((artefact.file_handle, artefact.download_url, artefact.md5_hash))

    return specs
