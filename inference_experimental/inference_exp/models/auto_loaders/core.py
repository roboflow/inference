import importlib
import importlib.util
import os.path
from copy import copy
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from filelock import FileLock
from inference_exp.configuration import (
    AUTO_LOADER_CACHE_EXPIRATION_MINUTES,
    DEFAULT_DEVICE,
    INFERENCE_HOME,
)
from inference_exp.errors import CorruptedModelPackageError, ModelLoadingError
from inference_exp.logger import LOGGER, verbose_info
from inference_exp.models.auto_loaders.auto_negotiation import negotiate_model_packages
from inference_exp.models.auto_loaders.entities import (
    MODEL_CONFIG_FILE_NAME,
    AutoResolutionCacheEntry,
    InferenceModelConfig,
    ModelArchitecture,
    TaskType,
)
from inference_exp.models.auto_loaders.models_registry import resolve_model_class
from inference_exp.models.base.classification import (
    ClassificationModel,
    MultiLabelClassificationModel,
)
from inference_exp.models.base.depth_estimation import DepthEstimationModel
from inference_exp.models.base.documents_parsing import DocumentParsingModel
from inference_exp.models.base.embeddings import TextImageEmbeddingModel
from inference_exp.models.base.instance_segmentation import InstanceSegmentationModel
from inference_exp.models.base.keypoints_detection import KeyPointsDetectionModel
from inference_exp.models.base.object_detection import ObjectDetectionModel
from inference_exp.utils.download import FileHandle, download_files_to_directory
from inference_exp.utils.file_system import dump_json, read_json, remove_file_if_exists
from inference_exp.weights_providers.core import get_model_from_provider
from inference_exp.weights_providers.entities import (
    BackendType,
    ModelPackageMetadata,
    Quantization,
)

AnyModel = Union[
    ClassificationModel,
    MultiLabelClassificationModel,
    DepthEstimationModel,
    DocumentParsingModel,
    TextImageEmbeddingModel,
    InstanceSegmentationModel,
    KeyPointsDetectionModel,
    ObjectDetectionModel,
]


class AutoModel:

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        weights_provider: str = "roboflow",
        api_key: Optional[str] = None,
        model_package_id: Optional[str] = None,
        backends: Optional[
            Union[str, BackendType, List[Union[str, BackendType]]]
        ] = None,
        batch_size: Optional[Union[int, Tuple[int, int]]] = None,
        quantization: Optional[
            Union[str, Quantization, List[Union[str, Quantization]]]
        ] = None,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        device: torch.device = DEFAULT_DEVICE,
        default_onnx_trt_options: bool = True,
        max_package_loading_attempts: Optional[int] = None,
        verbose: bool = False,
        model_download_file_lock_acquire_timeout: int = 10,
        allow_untrusted_packages: bool = False,
        trt_engine_host_code_allowed: bool = True,
        allow_local_code_packages: bool = False,
        verify_hash_while_download: bool = True,
        download_files_without_hash: bool = False,
        use_auto_resolution_cache: bool = True,
        **kwargs,
    ) -> AnyModel:
        # TODO: implement authorized cache
        model_init_kwargs = {
            "onnx_execution_providers": onnx_execution_providers,
            "device": device,
            "default_onnx_trt_options": default_onnx_trt_options,
            "engine_host_code_allowed": trt_engine_host_code_allowed,
        }
        model_init_kwargs.update(kwargs)
        if not os.path.isdir(model_name_or_path):
            # QUESTION: is it enough to assume presence of local dir as the intent to load
            # model from disc drive? What if we have clash of model id / model alias with
            # contents of someone's local drive - shall we then try to load from both sources?
            # that still may end up with ambiguous behaviour - probably the solution would be
            # to require prefix like file://... to denote the intent of loading model from local
            # drive?
            model_from_cache = attempt_loading_model_with_auto_load_cache(
                use_auto_resolution_cache=use_auto_resolution_cache,
                model_name_or_path=model_name_or_path,
                model_init_kwargs=model_init_kwargs,
                verbose=verbose,
            )
            if model_from_cache:
                return model_from_cache
            model_metadata = get_model_from_provider(
                provider=weights_provider,
                model_id=model_name_or_path,
                api_key=api_key,
            )
            matching_model_packages = negotiate_model_packages(
                model_packages=model_metadata.model_packages,
                requested_model_package_id=model_package_id,
                requested_backends=backends,
                requested_batch_size=batch_size,
                requested_quantization=quantization,
                device=device,
                onnx_execution_providers=onnx_execution_providers,
                allow_untrusted_packages=allow_untrusted_packages,
                trt_engine_host_code_allowed=trt_engine_host_code_allowed,
                verbose=verbose,
            )
            return attempt_loading_matching_model_packages(
                model_id=model_metadata.model_id,
                model_alias=model_name_or_path,
                model_architecture=model_metadata.model_architecture,
                task_type=model_metadata.task_type,
                matching_model_packages=matching_model_packages,
                model_init_kwargs=model_init_kwargs,
                max_package_loading_attempts=max_package_loading_attempts,
                model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                verify_hash_while_download=verify_hash_while_download,
                download_files_without_hash=download_files_without_hash,
                use_auto_resolution_cache=use_auto_resolution_cache,
            )
        return attempt_loading_model_from_local_storage(
            model_dir=model_package_id,
            allow_local_code_packages=allow_local_code_packages,
            model_init_kwargs=model_init_kwargs,
        )


def attempt_loading_model_with_auto_load_cache(
    use_auto_resolution_cache: bool,
    model_name_or_path: str,
    model_init_kwargs: dict,
    verbose: bool = False,
) -> Optional[AnyModel]:
    if not use_auto_resolution_cache:
        return None
    verbose_info(
        message=f"Attempt to load model {model_name_or_path} using auto-load cache.",
        verbose_requested=verbose,
    )
    cache_path = generate_auto_resolution_cache_path(model_id=model_name_or_path)
    if not os.path.exists(cache_path):
        verbose_info(
            message=f"Could not find auto-load cache for model {model_name_or_path}.",
            verbose_requested=verbose,
        )
        return None
    try:
        cache_content = read_json(path=cache_path)
        cache_entry = AutoResolutionCacheEntry.model_validate(cache_content)
        minutes_since_entry_created = (
            datetime.now() - cache_entry.created_at
        ).total_seconds() / 60
        if minutes_since_entry_created > AUTO_LOADER_CACHE_EXPIRATION_MINUTES:
            remove_file_if_exists(path=cache_path)
            verbose_info(
                message=f"Auto-load cache for model {model_name_or_path} expired.",
                verbose_requested=verbose,
            )
            return None
        if not all_files_exist(cache_entry.resolved_files):
            return None
        model_class = resolve_model_class(
            model_architecture=cache_entry.model_architecture,
            task_type=cache_entry.task_type,
            backend=cache_entry.backend_type,
        )
        model_package_cache_dir = generate_model_package_cache_path(
            model_id=cache_entry.model_id,
            package_id=cache_entry.model_package_id,
        )
        model = model_class.from_pretrained(
            model_package_cache_dir, **model_init_kwargs
        )
        verbose_info(
            message=f"Successfully loaded model {model_name_or_path} using auto-loading cache.",
            verbose_requested=verbose,
        )
        return model
    except Exception as error:
        LOGGER.warning(
            f"Encountered error {error} of type {type(error)} when attempted to load model using "
            f"auto-load cache. This may indicate corrupted cache of inference bug. Contact Roboflow submitting "
            f"issue under: https://github.com/roboflow/inference/issues/"
        )
        remove_file_if_exists(path=cache_path)
        return None


def all_files_exist(files: List[str]) -> bool:
    return all(os.path.exists(f) for f in files)


class FileLoadingMiddleware:

    def __init__(
        self,
        upstream_on_file_allocated: Optional[Callable[[str], None]] = None,
        upstream_on_file_renamed: Optional[Callable[[str, str], None]] = None,
        upstream_on_symlink_created: Optional[Callable[[str, str], None]] = None,
    ):
        self._upstream_on_file_allocated = upstream_on_file_allocated
        self._upstream_on_file_renamed = upstream_on_file_renamed
        self._upstream_on_symlink_created = upstream_on_symlink_created
        self._directory_content = set()

    def on_file_allocated(self, file: str) -> None:
        if self._upstream_on_file_allocated:
            self._upstream_on_file_allocated(file)
        self._directory_content.add(file)

    def on_file_renamed(self, old: str, new: str) -> None:
        if self._upstream_on_file_renamed:
            self._upstream_on_file_renamed(old, new)
        self._directory_content.remove(old)
        self._directory_content.add(new)

    def on_symlink_created(self, source: str, target: str) -> None:
        if self._upstream_on_symlink_created:
            self._upstream_on_symlink_created(source, target)
        self._directory_content.add(target)

    def get_all_files_in_directory(self) -> Set[str]:
        return copy(self._directory_content)


def attempt_loading_matching_model_packages(
    model_id: str,
    model_alias: str,
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    matching_model_packages: List[ModelPackageMetadata],
    model_init_kwargs: dict,
    max_package_loading_attempts: Optional[int] = None,
    model_download_file_lock_acquire_timeout: int = 10,
    verbose: bool = True,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    use_auto_resolution_cache: bool = True,
    on_file_allocated: Optional[Callable[[str], None]] = None,
    on_file_renamed: Optional[Callable[[str, str], None]] = None,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
) -> AnyModel:
    if max_package_loading_attempts is not None:
        matching_model_packages = matching_model_packages[:max_package_loading_attempts]
    if not matching_model_packages:
        raise ModelLoadingError(
            message=f"Cannot load model {model_id} - no matching model package candidates for given model "
            f"running in this environment.",
            help_url="https://todo",
        )
    failed_load_attempts: List[Tuple[str, Exception]] = []
    for model_package in matching_model_packages:
        verbose_info(
            message=f"Attempt to load model package: {model_package.get_summary()}",
            verbose_requested=verbose,
        )
        try:
            loading_middleware = FileLoadingMiddleware(
                upstream_on_file_allocated=on_file_allocated,
                upstream_on_file_renamed=on_file_renamed,
                upstream_on_symlink_created=on_symlink_created,
            )
            model = initialize_model(
                model_id=model_id,
                model_architecture=model_architecture,
                task_type=task_type,
                model_package=model_package,
                model_download_file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
                model_init_kwargs=model_init_kwargs,
                verify_hash_while_download=verify_hash_while_download,
                download_files_without_hash=download_files_without_hash,
                on_file_allocated=loading_middleware.on_file_allocated,
                on_file_renamed=loading_middleware.on_file_renamed,
                on_symlink_created=loading_middleware.on_symlink_created,
            )
            dump_auto_resolution_cache(
                use_auto_resolution_cache=use_auto_resolution_cache,
                model_id=model_id,
                model_alias=model_alias,
                model_package_id=model_package.package_id,
                model_architecture=model_architecture,
                task_type=task_type,
                backend_type=model_package.backend,
                resolved_files=loading_middleware.get_all_files_in_directory(),
                file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
            )
            return model
        except Exception as error:
            LOGGER.warning(
                f"Model package with id {model_package.package_id} that was selected to be loaded "
                f"failed to load with error: {error} of type {error.__class__.__name__}. This may "
                f"be caused several issues. If you see this warning after manually specifying model "
                f"package to be loaded - make sure that all required dependencies are installed. If "
                f"that warning is displayed when the model package was auto-selected - there is most "
                f"likely a bug in `inference` and you should raise an issue providing full context of "
                f"the event. https://github.com/roboflow/inference/issues"
            )
            failed_load_attempts.append((model_package.package_id, error))

    summary_of_errors = "\n".join(
        f"\t* model_package_id={model_package_id} error={error} error_type={error.__class__.__name__}"
        for model_package_id, error in failed_load_attempts
    )
    raise ModelLoadingError(
        message=f"Could not load any of model package candidate for model {model_id}. This may "
        f"be caused several issues. If you see this warning after manually specifying model "
        f"package to be loaded - make sure that all required dependencies are installed. If "
        f"that warning is displayed when the model package was auto-selected - there is most "
        f"likely a bug in `inference` and you should raise an issue providing full context of "
        f"the event. https://github.com/roboflow/inference/issues\n\n"
        f"Here is the summary of errors for specific model packages:\n{summary_of_errors}",
        help_url="https://todo",
    )


def initialize_model(
    model_id: str,
    model_architecture: ModelArchitecture,
    task_type: Optional[TaskType],
    model_package: ModelPackageMetadata,
    model_init_kwargs: dict,
    model_download_file_lock_acquire_timeout: int = 10,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    on_file_allocated: Optional[Callable[[str], None]] = None,
    on_file_renamed: Optional[Callable[[str, str], None]] = None,
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
) -> AnyModel:
    model_class = resolve_model_class(
        model_architecture=model_architecture,
        task_type=task_type,
        backend=model_package.backend,
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
                help_url="https://todo",
            )
    files_specs = [
        (a.file_handle, a.download_url, a.md5_hash)
        for a in model_package.package_artefacts
    ]
    file_specs_with_hash = [f for f in files_specs if f[2] is not None]
    file_specs_without_hash = [f for f in files_specs if f[2] is None]
    shared_blobs_dir = os.path.join(INFERENCE_HOME, "shared-blobs")
    model_package_cache_dir = generate_model_package_cache_path(
        model_id=model_id,
        package_id=model_package.package_id,
    )
    shared_files_mapping = download_files_to_directory(
        target_dir=shared_blobs_dir,
        files_specs=file_specs_with_hash,
        file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
        verify_hash_while_download=verify_hash_while_download,
        download_files_without_hash=download_files_without_hash,
        name_after="md5_hash",
        on_file_allocated=on_file_allocated,
        on_file_renamed=on_file_renamed,
    )
    _ = download_files_to_directory(
        target_dir=model_package_cache_dir,
        files_specs=file_specs_without_hash,
        file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
        verify_hash_while_download=verify_hash_while_download,
        download_files_without_hash=download_files_without_hash,
        on_file_allocated=on_file_allocated,
        on_file_renamed=on_file_renamed,
    )
    create_symlinks_to_shared_blobs(
        model_dir=model_package_cache_dir,
        shared_files_mapping=shared_files_mapping,
        on_symlink_created=on_symlink_created,
    )
    config_path = os.path.join(model_package_cache_dir, MODEL_CONFIG_FILE_NAME)
    dump_model_config_for_offline_use(
        config_path=config_path,
        model_architecture=model_architecture,
        task_type=task_type,
        backend_type=model_package.backend,
        file_lock_acquire_timeout=model_download_file_lock_acquire_timeout,
        on_file_allocated=on_file_allocated,
    )
    return model_class.from_pretrained(model_package_cache_dir, **model_init_kwargs)


def create_symlinks_to_shared_blobs(
    model_dir: str,
    shared_files_mapping: Dict[FileHandle, str],
    on_symlink_created: Optional[Callable[[str, str], None]] = None,
) -> None:
    # this function will not override existing files
    os.makedirs(model_dir, exist_ok=True)
    for file_handle, source_path in shared_files_mapping.items():
        link_name = os.path.join(model_dir, file_handle)
        target_path = shared_files_mapping[file_handle]
        if os.path.exists(link_name):
            continue
        if os.path.islink(link_name):
            # file does not exist, but is link = broken symlink - we should purge
            os.remove(link_name)
        os.symlink(target_path, link_name)
        if on_symlink_created:
            on_symlink_created(shared_files_mapping[file_handle], link_name)


def dump_model_config_for_offline_use(
    config_path: str,
    model_architecture: Optional[ModelArchitecture],
    task_type: TaskType,
    backend_type: Optional[BackendType],
    file_lock_acquire_timeout: int,
    on_file_allocated: Optional[Callable[[str], None]] = None,
) -> None:
    if os.path.exists(config_path):
        return None
    target_file_dir, target_file_name = os.path.split(config_path)
    lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
    with FileLock(lock_path, timeout=file_lock_acquire_timeout):
        dump_json(
            path=config_path,
            content={
                "model_architecture": model_architecture,
                "task_type": task_type,
                "backend_type": backend_type,
            },
        )
        if on_file_allocated:
            on_file_allocated(config_path)


def dump_auto_resolution_cache(
    use_auto_resolution_cache: bool,
    model_id: str,
    model_alias: str,
    model_package_id: str,
    model_architecture: Optional[ModelArchitecture],
    task_type: TaskType,
    backend_type: Optional[BackendType],
    resolved_files: Set[str],
    file_lock_acquire_timeout: int,
) -> None:
    if not use_auto_resolution_cache:
        return None
    cache_content = AutoResolutionCacheEntry(
        model_id=model_id,
        model_package_id=model_package_id,
        resolved_files=resolved_files,
        model_architecture=model_architecture,
        task_type=task_type,
        backend_type=backend_type,
        created_at=datetime.now(),
    )
    path_for_model_id = generate_auto_resolution_cache_path(model_id=model_id)
    _dump_auto_resolution_cache(
        path=path_for_model_id,
        content=cache_content.model_dump(mode="json"),
        file_lock_acquire_timeout=file_lock_acquire_timeout,
    )
    if model_alias != model_id:
        path_for_alias = generate_auto_resolution_cache_path(model_id=model_alias)
        _dump_auto_resolution_cache(
            path=path_for_alias,
            content=cache_content.model_dump(mode="json"),
            file_lock_acquire_timeout=file_lock_acquire_timeout,
        )


def generate_auto_resolution_cache_path(model_id: str) -> str:
    return os.path.join(INFERENCE_HOME, "auto-resolution-cache", f"{model_id}.json")


def generate_model_package_cache_path(model_id: str, package_id: str) -> str:
    return os.path.join(INFERENCE_HOME, "models-cache", model_id, package_id)


def _dump_auto_resolution_cache(
    path: str,
    content: dict,
    file_lock_acquire_timeout: int,
) -> None:
    target_file_dir, target_file_name = os.path.split(path)
    lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
    with FileLock(lock_path, timeout=file_lock_acquire_timeout):
        dump_json(path=path, content=content)


def attempt_loading_model_from_local_storage(
    model_dir: str,
    allow_local_code_packages: bool,
    model_init_kwargs: dict,
) -> AnyModel:
    config_path = os.path.join(model_dir, MODEL_CONFIG_FILE_NAME)
    model_config = parse_model_config(config_path=config_path)
    if model_config.is_library_model():
        return load_library_model_from_local_dir(
            model_dir=model_dir,
            model_config=model_config,
            model_init_kwargs=model_init_kwargs,
        )
    if not allow_local_code_packages:
        raise ModelLoadingError(
            message=f"Attempted to load model from local package with arbitrary code. This is not allowed in "
            f"this environment. To let inference loading such models, use `allow_local_code_packages=True` "
            f"parameter of `AutoModel.from_pretrained(...)`. If you see this error while using one of Roboflow "
            f"hosted solution - contact us to solve the problem.",
            help_url="https://todo",
        )
    return load_model_from_local_package_with_arbitrary_code(
        model_dir=model_dir,
        model_config=model_config,
        model_init_kwargs=model_init_kwargs,
    )


def parse_model_config(config_path: str) -> InferenceModelConfig:
    if not os.path.isfile(config_path):
        raise ModelLoadingError(
            message=f"Could not find model config while attempting to load model from "
            f"local directory. This error may be caused by misconfiguration of model package (lack of config "
            f"file), as well as by clash between model_id or model alias and contents of local disc drive which "
            f"is possible when you have local directory in current dir which has the name colliding with the "
            f"model you attempt to load. If your intent was to load model from remote backend (not local "
            f"storage) - verify the contents of $PWD. If you see this problem while using one of Roboflow "
            f"hosted solutions - contact us to get help.",
            help_url="https://todo",
        )
    try:
        raw_config = read_json(path=config_path)
    except ValueError as error:
        raise CorruptedModelPackageError(
            message=f"Could not decode model config while attempting to load model from "
            f"local directory. This error may be caused by corrupted config file. Validate the content of your "
            f"model package and check in documentation the required format of model config file. "
            f"If you see this problem while using one of Roboflow hosted solutions - contact us to get help.",
            help_url="https://todo",
        ) from error
    if not isinstance(raw_config, dict):
        raise CorruptedModelPackageError(
            message=f"While loading the model from local directory encountered corrupted model config file - config is "
            f"supposed to be a dictionary, instead decoded object of type: "
            f"{type(raw_config)}. If you see this problem while using one of Roboflow hosted solutions - "
            f"contact us to get help. Otherwise - verify the content of your model config.",
            help_url="https://todo",
        )
    backend_type = None
    if "backend_type" in raw_config:
        raw_backend_type = raw_config["backend_type"]
        try:
            backend_type = BackendType(raw_backend_type)
        except ValueError as e:
            raise CorruptedModelPackageError(
                message=f"While loading the model from local directory encountered corrupted model config "
                "- declared `backend_type` ({raw_backend_type}) is not supported by inference. "
                f"Supported values: {list(t.value for t in BackendType)}. If you see this problem while using "
                f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify the content "
                f"of your model config.",
                help_url="https://todo",
            ) from e
    return InferenceModelConfig(
        model_architecture=raw_config.get("model_architecture"),
        task_type=raw_config.get("task_type"),
        backend_type=backend_type,
        model_module=raw_config.get("model_module"),
        model_class=raw_config.get("model_class"),
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
    )
    return model_class.from_pretrained(model_dir, **model_init_kwargs)


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
            help_url="https://todo",
        )
    model_module_path = os.path.join(model_dir, model_config.model_module)
    if not os.path.isfile(model_module_path):
        raise CorruptedModelPackageError(
            message=f"While loading the model from local directory encountered corrupted model config file. "
            f"Config pointed module {model_config.model_module}, but there is no file under "
            f"{model_module_path}. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify the content "
            f"of your model config.",
            help_url="https://todo",
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
            help_url="https://todo",
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
            help_url="https://todo",
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
            help_url="https://todo",
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
            help_url="https://todo",
        )
    if not hasattr(module, class_name):
        raise CorruptedModelPackageError(
            message=f"When loading local model with arbitrary code, encountered issue with loading the module. "
            f"Module `{module_name}` has no class `{class_name}`. If you see this problem while using "
            f"one of Roboflow hosted solutions - contact us to get help. Otherwise - verify your "
            f"model package checking if you can load the module with model implementation within your "
            f"python environment. It may also be the case that configuration file of the model points "
            f"to invalid class name.",
            help_url="https://todo",
        )
    return getattr(module, class_name)
