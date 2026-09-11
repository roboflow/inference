"""Injected configuration for `inference/core/workflows`.

Every value the module used to take from `inference.core.env` is a field of
`WorkflowsConfiguration`, which the host builds from its own already-resolved
settings and installs once per process with `configure_process`.

The configuration is PROCESS-WIDE in its entirety. Every one of the 67 values
is read from a module constant frozen at import - `core_steps/loader.py`
branches on the tensor flag while it is being imported, `offline.py:34` reads
`SECURE_GATEWAY`, `block_assembler.py:100` reads the custom-Python
authorization flag - so there is no field a second, differing configuration
could change after the fact. `ensure_process_configuration_matches` therefore
refuses ANY difference rather than accepting a value nothing would honour.

Per-engine variance lives where it already did: the named init parameters
`workflows_core.{step_execution_mode, api_key, disable_sinks,
allow_access_to_file_system, allowed_write_directory,
allow_access_to_environmental_variables}`, resolved by
`steps_initialiser.retrieve_init_parameter_values`.

`default_configuration()` is the STANDALONE fallback, not the production path.
Its values are the ones `inference/core/env.py` resolves for an EMPTY
environment. Every normalisation `env.py` performs - `str2bool`, `.lower()`,
the `OFFLINE_MODE` / `SECURE_GATEWAY` rewrites, the `PROJECT`-dependent hosted
URLs, the api-key-transport validation - stays in `env.py`; the host builder
copies the RESOLVED attributes. `resolve_image_tensor_device` is the single
copied piece of logic and is pinned by a differential test against `env.py`.
"""

import threading
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, List, Optional, Tuple

from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError


@dataclass(frozen=True)
class EngineConfiguration:
    step_execution_mode: str = "local"
    async_future_result_timeout: float = 60.0
    max_inner_workflow_depth: int = 4
    max_inner_workflow_count: int = 32
    allow_custom_python_execution: bool = True
    custom_python_execution_mode: str = "local"
    allow_blocks_accessing_local_storage: bool = True
    allow_blocks_accessing_environmental_variables: bool = True
    blocks_write_directory: Optional[str] = None
    # Tuples, not lists: the configuration is frozen and compared by value. The
    # facade re-materialises the `list` the call sites see today.
    disabled_block_types: Tuple[str, ...] = ()
    disabled_block_patterns: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TensorConfiguration:
    """Selects which block classes `core_steps/loader.py` imports and how
    `WorkflowImageData` stores pixels. Block loading is process-global and
    cached (`blocks_loader.py:252`), so these can never be per-engine."""

    representation_enabled: bool = False
    # `torch.device` when the flag is on and torch is importable, else None.
    # Typed `Any` so this module never imports torch (an OPTIONAL dependency).
    image_tensor_device: Optional[Any] = None
    visualisation_validate_owners: bool = False
    sam_video_mask_representation: str = "rle"
    enforce_dense_instance_masks: bool = False


@dataclass(frozen=True)
class RemoteExecutionConfiguration:
    api_target: str = "hosted"
    api_key_transport: str = "both"
    local_inference_api_url: str = "http://127.0.0.1:9001"
    hosted_detect_url: str = "https://detect.roboflow.com"
    hosted_classification_url: str = "https://classify.roboflow.com"
    hosted_instance_segmentation_url: str = "https://outline.roboflow.com"
    hosted_semantic_segmentation_url: str = "https://segment.roboflow.com"
    hosted_core_model_url: str = "https://infer.roboflow.com"
    max_step_batch_size: int = 1
    max_step_concurrent_requests: int = 8


@dataclass(frozen=True)
class PlatformConfiguration:
    api_base_url: str = "https://api.roboflow.com"
    offline_mode: bool = False
    secure_gateway: Optional[str] = None
    gcp_serverless: bool = False


@dataclass(frozen=True)
class FontsConfiguration:
    allow_download: bool = True
    model_cache_dir: str = "/tmp/cache"


@dataclass(frozen=True)
class ModelsConfiguration:
    lmm_enabled: bool = False
    clip_version_id: str = "ViT-B-16"
    core_model_sam2_enabled: bool = True
    core_model_sam3_enabled: bool = True
    core_model_pe_enabled: bool = True
    core_model_gaze_enabled: bool = True
    sam3_exec_mode: str = "local"
    sam3_3d_objects_enabled: bool = False
    florence2_enabled: bool = True
    qwen_2_5_enabled: bool = True
    qwen_3_enabled: bool = True
    qwen_3_5_enabled: bool = True
    smolvlm2_enabled: bool = True
    moondream2_enabled: bool = True
    depth_estimation_enabled: bool = True
    cosmos3_enabled: bool = True
    glm_ocr_enabled: bool = True


@dataclass(frozen=True)
class ModalConfiguration:
    token_id: Optional[str] = None
    token_secret: Optional[str] = field(default=None, repr=False)
    workspace_name: str = "roboflow"
    allow_anonymous_execution: bool = False
    anonymous_workspace_name: str = "anonymous"
    app_name: str = "webexec-roboflow-platform"
    executor_idle_ttl_seconds: int = 1800
    jpeg_quality: int = 95
    transport: str = "http"
    ws_connect_timeout_seconds: int = 30
    ws_read_timeout_seconds: int = 720
    ws_connection_pool_size: int = 1
    ws_fail_on_session_loss: bool = False
    ws_idle_release_seconds: int = 120


@dataclass(frozen=True)
class SecretsConfiguration:
    api_key: Optional[str] = field(default=None, repr=False)
    roboflow_internal_service_name: Optional[str] = None
    roboflow_internal_service_secret: Optional[str] = field(default=None, repr=False)


@dataclass(frozen=True)
class DebugConfiguration:
    output_dir: Optional[str] = None


@dataclass(frozen=True)
class WorkflowsConfiguration:
    engine: EngineConfiguration = field(default_factory=EngineConfiguration)
    tensor: TensorConfiguration = field(default_factory=TensorConfiguration)
    remote: RemoteExecutionConfiguration = field(
        default_factory=RemoteExecutionConfiguration
    )
    platform: PlatformConfiguration = field(default_factory=PlatformConfiguration)
    fonts: FontsConfiguration = field(default_factory=FontsConfiguration)
    models: ModelsConfiguration = field(default_factory=ModelsConfiguration)
    modal: ModalConfiguration = field(default_factory=ModalConfiguration)
    secrets: SecretsConfiguration = field(default_factory=SecretsConfiguration)
    debug: DebugConfiguration = field(default_factory=DebugConfiguration)


def default_configuration() -> WorkflowsConfiguration:
    """The standalone fallback: every field at its documented default."""
    return WorkflowsConfiguration()


def resolve_image_tensor_device(
    representation_enabled: bool, device: Optional[str] = None
) -> Optional[Any]:
    """Materialise the tensor device exactly as `inference/core/env.py:1535-1547`.

    `torch` is an OPTIONAL dependency, so both the import and the device are
    deferred behind the flag AND guarded on torch's presence. Off-flag or
    torch-less the value is `None`. An invalid device string raises whatever
    `torch.device` raises, which is what `env.py` does too - it does not guard
    that call either.

    The `try` deliberately spans the import, the cuda autodetect AND the
    materialisation, catching only `ImportError`, because that is exactly
    `env.py:1536-1547`'s boundary (round-2 defect 5: the round-1 copy protected
    only the import, so a `torch.cuda.is_available()` raising `ImportError`
    would have propagated from the copy while `env.py` swallowed it).
    `test_the_copy_preserves_the_originals_exception_boundary` pins both shapes.
    """
    if not representation_enabled:
        return None
    try:
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    except ImportError:
        return None


def describe_configuration_difference(
    installed: WorkflowsConfiguration, candidate: WorkflowsConfiguration
) -> List[str]:
    """`['group.field: <installed> != <candidate>', ...]`, empty when equal.

    Secret fields are compared but never rendered - the message says the field
    differs, not what either value is.
    """
    differences = []
    for group in fields(WorkflowsConfiguration):
        installed_group = getattr(installed, group.name)
        candidate_group = getattr(candidate, group.name)
        if installed_group == candidate_group:
            continue
        if not is_dataclass(candidate_group):
            differences.append(f"{group.name}: differs")
            continue
        for member in fields(installed_group):
            left = getattr(installed_group, member.name)
            right = getattr(candidate_group, member.name)
            if left == right:
                continue
            if member.repr is False:
                differences.append(f"{group.name}.{member.name}: differs (redacted)")
            else:
                differences.append(f"{group.name}.{member.name}: {left!r} != {right!r}")
    return differences


_CONFIGURATION: Optional[WorkflowsConfiguration] = None
# One lock guards installation, the sticky fallback initialisation in
# `get_configuration` (which also assigns) and `reset_configuration`. Round-1
# review reproduced two conflicting installs both succeeding through an
# unlocked check-then-assign; Phase 10's image-codec registry locks for the
# same reason.
_INSTALL_LOCK = threading.RLock()


def configure_process(configuration: WorkflowsConfiguration) -> None:
    """Install the process-wide configuration. Set-once, by value.

    Installing the identical object, or an EQUAL one, is a no-op - frozen
    dataclasses compare by value and the server memoises its instance.
    Installing a DIFFERING one raises, naming every group and field that
    differs.
    """
    global _CONFIGURATION
    with _INSTALL_LOCK:
        if _CONFIGURATION is not None and _CONFIGURATION != configuration:
            differences = describe_configuration_difference(
                _CONFIGURATION, configuration
            )
            raise WorkflowEnvironmentConfigurationError(
                public_message=(
                    "A different WorkflowsConfiguration is already installed in "
                    "this process. Workflows configuration is process-level: "
                    "block registration and the image representation are decided "
                    "once, at import time, and every other value is frozen into "
                    "module constants. Install the configuration before importing "
                    "any workflows module, and install the same one everywhere. "
                    f"Differences: {differences}"
                ),
                context="workflow_configuration | process_installation",
            )
        _CONFIGURATION = configuration


def get_configuration() -> WorkflowsConfiguration:
    """The installed configuration, falling back to the standalone default.

    The fallback is STICKY: once read, it becomes the process configuration, so
    a later `configure_process` with different values raises instead of
    silently disagreeing with values already frozen into module constants.
    """
    global _CONFIGURATION
    with _INSTALL_LOCK:
        if _CONFIGURATION is None:
            _CONFIGURATION = default_configuration()
        return _CONFIGURATION


def reset_configuration() -> None:
    """Test-only hook: forget the installed configuration."""
    global _CONFIGURATION
    with _INSTALL_LOCK:
        _CONFIGURATION = None


def ensure_process_configuration_matches(configuration: Any) -> None:
    """Refuse an engine-supplied configuration that differs from the process one.

    `init_parameters["workflows_core.configuration"]` is a CONSISTENCY
    ASSERTION, not an override channel: the composition roots pass the object
    the host installed, and a mismatch means the process is mis-wired. Because
    every value is read from a module constant frozen at import, accepting a
    differing value would mean accepting a value nothing honours.

    The caller decides PRESENCE (the engine calls this only when the key is in
    `init_parameters`); this function judges the VALUE, and `None` is not a
    valid value: an explicit `None` would be handed to every block that
    declares `configuration` ahead of the registered default
    (`steps_initialiser.py:124-125` prefers explicit values), so it is refused
    like any other non-`WorkflowsConfiguration` (round-4 defect 1).
    """
    if not isinstance(configuration, WorkflowsConfiguration):
        # Only ever reached through the DEDICATED init-parameter keys, so a
        # foreign value there is a mis-wiring, never a plugin's own parameter
        # (`v1/core.py` deliberately does not fall back to the bare name).
        raise WorkflowEnvironmentConfigurationError(
            public_message=(
                "`workflows_core.configuration` must be a WorkflowsConfiguration, "
                f"got {type(configuration).__name__}."
            ),
            context="workflow_compilation | engine_initialisation",
        )
    installed = get_configuration()
    if configuration == installed:
        return
    differences = describe_configuration_difference(installed, configuration)
    raise WorkflowEnvironmentConfigurationError(
        public_message=(
            "`workflows_core.configuration` differs from the WorkflowsConfiguration "
            "installed in this process. Workflows configuration is process-level "
            "and cannot be varied per engine; use the named init parameters "
            "(`workflows_core.step_execution_mode`, `workflows_core.api_key`, "
            "`workflows_core.disable_sinks`, "
            "`workflows_core.allow_access_to_file_system`, "
            "`workflows_core.allowed_write_directory`, "
            "`workflows_core.allow_access_to_environmental_variables`) for "
            f"per-engine values. Differences: {differences}"
        ),
        context="workflow_compilation | engine_initialisation",
    )
