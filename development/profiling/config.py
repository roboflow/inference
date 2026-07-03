from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, field_validator
import yaml


DEFAULT_CAPTURE_RANGE = "profile-target"
DEFAULT_OUTPUT_DIR = Path("inference_profiling/snippets")


class CudaSyncConfig(BaseModel):
    """CUDA synchronization settings for a profiling run."""

    synchronize_before_warmup: bool = Field(
        default=True,
        description="Whether to synchronize CUDA before warmup passes.",
    )
    synchronize_after_warmup: bool = Field(
        default=True,
        description="Whether to synchronize CUDA after warmup passes.",
    )
    synchronize_before_capture: bool = Field(
        default=True,
        description="Whether to synchronize CUDA before the measured capture range.",
    )
    synchronize_after_capture: bool = Field(
        default=True,
        description="Whether to synchronize CUDA after the measured capture range.",
    )
    synchronize_each_iteration: bool = Field(
        default=False,
        description="Whether to synchronize CUDA after each target invocation.",
    )


class TargetConfig(BaseModel):
    """Profile target configuration."""

    name: str = Field(
        description="Profile target name.",
        examples=["smoke-tensor"],
    )
    import_path: str | None = Field(
        default=None,
        description="Optional file import path for generated targets.",
        examples=["inference_profiling/snippets/example/target.py:target"],
    )
    profile_prepare: bool = Field(
        default=False,
        description="Whether to include target preparation in warmup and capture.",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate target name.

        Args:
            value (str): Provided target name.

        Returns:
            Normalized target name.

        Raises:
            ValueError: If the target name is empty.
        """
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("target.name must not be empty.")

        return stripped_value


class DataSourceConfig(BaseModel):
    """Data source configuration."""

    name: str = Field(
        description="Data source name.",
        examples=["dummy", "images"],
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Data-source-specific configuration parameters.",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate data source name.

        Args:
            value (str): Provided data source name.

        Returns:
            Normalized data source name.

        Raises:
            ValueError: If the data source name is empty.
        """
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("data_source.name must not be empty.")

        return stripped_value


class ProfileConfig(BaseModel):
    """Complete configuration for a profiling run."""

    profile_name: str = Field(
        description="Profile name used in generated run paths and manifests.",
        examples=["smoke-tensor"],
    )
    target: TargetConfig = Field(
        description="Target selection and target execution settings.",
    )
    data_source: DataSourceConfig = Field(
        description="Data source selection and source-specific settings.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device string used for target preparation.",
        examples=["cpu", "cuda"],
    )
    warmup: int = Field(
        default=1,
        description="Number of warmup passes to run before capture.",
    )
    iterations: int = Field(
        default=1,
        description="Number of measured passes to run inside capture.",
    )
    record_loading: Literal["eager", "lazy"] = Field(
        default="eager",
        description="Whether records are materialized once or iterated lazily per pass.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional reproducibility seed.",
    )
    validate_output: bool = Field(
        default=True,
        description="Whether to validate target outputs during warmup.",
    )
    capture_range: str = Field(
        default=DEFAULT_CAPTURE_RANGE,
        description="Top-level NVTX capture range name.",
    )
    output_dir: Path = Field(
        default=DEFAULT_OUTPUT_DIR,
        description="Base directory for generated run manifests and traces.",
    )
    write_manifest: bool = Field(
        default=True,
        description="Whether to write a run manifest after execution.",
    )
    cuda: CudaSyncConfig = Field(
        default_factory=CudaSyncConfig,
        description="CUDA synchronization policy.",
    )

    @field_validator("profile_name", "device", "capture_range")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Validate required string fields.

        Args:
            value (str): Provided string value.

        Returns:
            Normalized string value.

        Raises:
            ValueError: If the value is empty.
        """
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("String config values must not be empty.")

        return stripped_value

    @field_validator("warmup")
    @classmethod
    def validate_warmup(cls, value: int) -> int:
        """Validate warmup pass count.

        Args:
            value (int): Provided warmup count.

        Returns:
            Validated warmup count.

        Raises:
            ValueError: If warmup count is negative.
        """
        if value < 0:
            raise ValueError("warmup must be greater than or equal to 0.")

        return value

    @field_validator("iterations")
    @classmethod
    def validate_iterations(cls, value: int) -> int:
        """Validate measured iteration count.

        Args:
            value (int): Provided measured iteration count.

        Returns:
            Validated iteration count.

        Raises:
            ValueError: If measured iteration count is less than 1.
        """
        if value < 1:
            raise ValueError("iterations must be greater than or equal to 1.")

        return value

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize this config for manifest output.

        Returns:
            Dictionary representation with path fields converted to strings.
        """
        result = self.model_dump(mode="json")

        return result


def load_profile_config(path: str | Path) -> ProfileConfig:
    """Load a YAML profiling config.

    Args:
        path (str | Path): Path to the YAML config file.

    Returns:
        Parsed profile configuration.

    Raises:
        ValueError: If the YAML root is not a mapping.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, Mapping):
        raise ValueError("Profile config must be a YAML mapping.")

    profile_config = parse_profile_config(raw_config)

    return profile_config


def parse_profile_config(raw_config: Mapping[str, Any]) -> ProfileConfig:
    """Parse raw YAML config values into a profile config.

    Args:
        raw_config (Mapping[str, Any]): Raw mapping loaded from YAML.

    Returns:
        Parsed profile configuration.

    Raises:
        ValueError: If required sections or values are missing.
    """
    target_raw = _required_mapping(raw_config, "target")
    data_raw = _get_data_source_config(raw_config)
    cuda_raw = _optional_mapping(raw_config, "cuda")

    profile_name = (
        raw_config.get("profile_name")
        or _optional_mapping(raw_config, "profile").get("name")
        or target_raw.get("name")
    )
    if not profile_name:
        raise ValueError("Profile config requires 'profile_name' or 'profile.name'.")

    workload_raw = _optional_mapping(raw_config, "workload")
    data_source_name = data_raw.get("name") or data_raw.get("source")
    if data_source_name is None:
        raise ValueError("Data source config requires 'name' or 'source'.")

    profile_config = ProfileConfig(
        profile_name=str(profile_name),
        target=TargetConfig(
            name=str(_required_value(target_raw, "name")),
            import_path=_optional_str(target_raw.get("import_path")),
            profile_prepare=bool(target_raw.get("profile_prepare", False)),
        ),
        data_source=DataSourceConfig(
            name=str(data_source_name),
            parameters=_data_source_parameters(data_raw),
        ),
        device=str(_coalesce(raw_config, workload_raw, "device", default="cpu")),
        warmup=int(_coalesce(raw_config, workload_raw, "warmup", default=1)),
        iterations=int(_coalesce(raw_config, workload_raw, "iterations", default=1)),
        record_loading=_parse_record_loading(
            _coalesce(raw_config, workload_raw, "record_loading", default="eager")
        ),
        seed=_optional_int(_coalesce(raw_config, workload_raw, "seed", default=None)),
        validate_output=bool(
            _coalesce(raw_config, workload_raw, "validate_output", default=True)
        ),
        capture_range=str(
            _coalesce(
                raw_config,
                workload_raw,
                "capture_range",
                default=DEFAULT_CAPTURE_RANGE,
            )
        ),
        output_dir=Path(str(raw_config.get("output_dir", DEFAULT_OUTPUT_DIR))),
        write_manifest=bool(raw_config.get("write_manifest", True)),
        cuda=CudaSyncConfig(
            synchronize_before_warmup=bool(
                cuda_raw.get("synchronize_before_warmup", True)
            ),
            synchronize_after_warmup=bool(
                cuda_raw.get("synchronize_after_warmup", True)
            ),
            synchronize_before_capture=bool(
                cuda_raw.get("synchronize_before_capture", True)
            ),
            synchronize_after_capture=bool(
                cuda_raw.get("synchronize_after_capture", True)
            ),
            synchronize_each_iteration=bool(
                cuda_raw.get("synchronize_each_iteration", False)
            ),
        ),
    )

    return profile_config


def apply_overrides(
    config: ProfileConfig,
    *,
    target: str | None = None,
    data_source: str | None = None,
    device: str | None = None,
    warmup: int | None = None,
    iterations: int | None = None,
    capture_range: str | None = None,
    record_loading: Literal["eager", "lazy"] | None = None,
    seed: int | None = None,
) -> ProfileConfig:
    """Apply CLI overrides to a parsed profile config.

    Args:
        config (ProfileConfig): Base profile configuration.
        target (str | None): Optional target name override.
        data_source (str | None): Optional data source name override.
        device (str | None): Optional device override.
        warmup (int | None): Optional warmup count override.
        iterations (int | None): Optional measured iteration count override.
        capture_range (str | None): Optional capture range override.
        record_loading (Literal["eager", "lazy"] | None): Optional record loading
            mode override.
        seed (int | None): Optional reproducibility seed override.

    Returns:
        New profile configuration with overrides applied.
    """
    overridden_config = ProfileConfig(
        profile_name=config.profile_name,
        target=TargetConfig(
            name=target or config.target.name,
            import_path=config.target.import_path,
            profile_prepare=config.target.profile_prepare,
        ),
        data_source=DataSourceConfig(
            name=data_source or config.data_source.name,
            parameters=config.data_source.parameters,
        ),
        device=device or config.device,
        warmup=config.warmup if warmup is None else warmup,
        iterations=config.iterations if iterations is None else iterations,
        record_loading=(
            config.record_loading if record_loading is None else record_loading
        ),
        seed=config.seed if seed is None else seed,
        validate_output=config.validate_output,
        capture_range=capture_range or config.capture_range,
        output_dir=config.output_dir,
        write_manifest=config.write_manifest,
        cuda=config.cuda,
    )

    return overridden_config


def _get_data_source_config(raw_config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "data_source" in raw_config:
        return _required_mapping(raw_config, "data_source")
    if "data" in raw_config:
        return _required_mapping(raw_config, "data")
    raise ValueError("Profile config requires 'data_source' or 'data'.")


def _data_source_parameters(raw_config: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = dict(raw_config.get("parameters", {}))
    for key, value in raw_config.items():
        if key not in {"name", "source", "parameters"}:
            parameters[key] = value
    return parameters


def _required_mapping(raw_config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Profile config requires mapping '{key}'.")
    return value


def _optional_mapping(raw_config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw_config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Profile config field '{key}' must be a mapping.")
    return value


def _required_value(raw_config: Mapping[str, Any], key: str) -> Any:
    value = raw_config.get(key)
    if value is None:
        raise ValueError(f"Profile config requires '{key}'.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_record_loading(value: Any) -> Literal["eager", "lazy"]:
    record_loading = str(value)
    if record_loading == "eager":
        return "eager"

    if record_loading == "lazy":
        return "lazy"

    raise ValueError("record_loading must be either 'eager' or 'lazy'.")


def _coalesce(
    raw_config: Mapping[str, Any],
    workload_raw: Mapping[str, Any],
    key: str,
    *,
    default: Any,
) -> Any:
    if key in raw_config:
        return raw_config[key]
    if key in workload_raw:
        return workload_raw[key]
    return default
