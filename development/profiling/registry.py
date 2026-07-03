import importlib.util
import hashlib
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from development.profiling.data.base import DataRecord


@runtime_checkable
class ProfileTarget(Protocol):
    """Interface that profiling targets must implement."""

    name: str

    def prepare(self, record: DataRecord, *, device: torch.device) -> Any:
        """Prepare a data record for measured target execution.

        Args:
            record (DataRecord): Data record emitted by a data source.
            device (torch.device): Device selected for this profiling run.

        Returns:
            Prepared object consumed by ``run``.
        """
        ...

    def run(self, prepared: Any) -> Any:
        """Run the profiled target operation.

        Args:
            prepared (Any): Prepared object returned by ``prepare``.

        Returns:
            Target output.
        """
        ...

    def validate(self, output: Any) -> None:
        """Validate target output.

        Args:
            output (Any): Output returned by ``run``.
        """
        ...

    def summarize(self, output: Any) -> dict[str, Any]:
        """Summarize target output for manifest metadata.

        Args:
            output (Any): Output returned by ``run``.

        Returns:
            Lightweight manifest-safe output summary.
        """
        ...


TargetFactory = Callable[[], ProfileTarget]
_BUILTIN_TARGETS: dict[str, ProfileTarget | TargetFactory] = {}


def register_target(name: str, target: ProfileTarget | TargetFactory) -> None:
    """Register a built-in profiling target.

    Args:
        name (str): Target lookup name.
        target (ProfileTarget | TargetFactory): Target instance or factory.
    """
    _BUILTIN_TARGETS[name] = target


def resolve_target(name: str, import_path: str | None = None) -> ProfileTarget:
    """Resolve a built-in or explicitly imported profiling target.

    Args:
        name (str): Target lookup name.
        import_path (str | None): Optional ``path.py:attribute`` import path.

    Returns:
        Resolved profiling target.

    Raises:
        ValueError: If a built-in target is unknown or import path is malformed.
        FileNotFoundError: If an import path file does not exist.
        TypeError: If the resolved object is not a profile target.
    """
    if import_path:
        imported_candidate = _load_from_file_import(import_path)
        resolved_target = _coerce_target(imported_candidate, import_path)

        return resolved_target

    try:
        target = _BUILTIN_TARGETS[name]
    except KeyError as error:
        available = ", ".join(sorted(_BUILTIN_TARGETS))
        raise ValueError(
            f"Unknown target '{name}'. Provide target.import_path or use one of: "
            f"{available}"
        ) from error

    resolved_target = _coerce_target(target, name)

    return resolved_target


def registered_target_names() -> tuple[str, ...]:
    """List registered built-in target names.

    Returns:
        Sorted tuple of registered target names.
    """
    target_names = tuple(sorted(_BUILTIN_TARGETS))

    return target_names


def _load_from_file_import(import_path: str) -> Any:
    file_path_text, separator, attribute_name = import_path.partition(":")
    if not separator or not attribute_name:
        raise ValueError(
            "Generated target import_path must use '<path-to-file.py>:<attribute>'."
        )

    file_path = Path(file_path_text)
    if not file_path.exists():
        raise FileNotFoundError(f"Target import file does not exist: {file_path}")

    digest = hashlib.sha1(str(file_path.resolve()).encode()).hexdigest()
    module_name = f"_profiling_target_{digest}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import profiling target from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        imported_attribute = getattr(module, attribute_name)
    except AttributeError as error:
        raise AttributeError(
            f"Target import '{import_path}' does not expose '{attribute_name}'."
        ) from error

    return imported_attribute


def _coerce_target(candidate: Any, source: str) -> ProfileTarget:
    if isinstance(candidate, type):
        candidate = candidate()

    if _is_profile_target(candidate):
        return candidate

    if callable(candidate):
        candidate = candidate()
        if _is_profile_target(candidate):
            return candidate

    raise TypeError(
        f"Resolved profiling target from '{source}' must implement ProfileTarget "
        "or be a zero-argument factory that returns one."
    )


def _is_profile_target(candidate: Any) -> bool:
    is_profile_target = all(
        hasattr(candidate, attribute)
        for attribute in ("name", "prepare", "run", "validate", "summarize")
    )

    return is_profile_target


def _register_builtin_targets() -> None:
    from development.profiling.targets import BUILTIN_TARGETS

    for name, target in BUILTIN_TARGETS.items():
        register_target(name, target)


_register_builtin_targets()
