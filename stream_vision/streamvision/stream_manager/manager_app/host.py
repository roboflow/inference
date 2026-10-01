"""The host contract of the stream manager's pipeline processes.

The manager is host-neutral: it does not know how a named workflow definition
is fetched, which models provider a workflow runs with, or which API key
applies. A `PipelineHost` supplies all of that. It is built inside each
pipeline process, from a `PipelineHostDescriptor` - the import path of a
trusted factory plus plain settings - chosen when the manager is launched and
pickled across every process boundary. Live resources never cross a boundary.

A launcher passes its descriptor explicitly. A host that configures a whole
process may also install a process default here, so that callers of the
historical `start()` keep working without naming one. This module never
imports or guesses a host itself: without either, resolution fails.

This module must stay import-light: it is imported by the process bootstrap,
before any configuration-consuming stream runtime module.
"""

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple


class PipelineHostNotConfiguredError(RuntimeError):
    """No pipeline host descriptor was passed or installed for this process."""


class PipelineHost(Protocol):
    """What a pipeline process needs from its host to run a workflow."""

    def prepare_workflow(
        self,
        *,
        workflow_specification: Optional[dict],
        workspace_name: Optional[str],
        workflow_id: Optional[str],
        workflow_version_id: Optional[str],
        api_key: Optional[str],
        profiler: Any,
    ) -> Tuple[dict, Dict[str, Any], Any]:
        """Resolve a workflow and the Execution Engine bindings it runs with.

        Args:
            workflow_specification: Inline workflow definition, if any.
            workspace_name: Workspace of a registered workflow.
            workflow_id: Identifier of a registered workflow.
            workflow_version_id: Version of a registered workflow.
            api_key: API key sent with the request, if any.
            profiler: Workflows profiler recording the host's own work; the
                manager passes the same instance to the pipeline.

        Returns:
            The workflow specification, the Execution Engine init parameters
            and the step error handler.
        """

    def close(self) -> None:
        """Release what the host holds. Called once the host is no longer used."""


@dataclass(frozen=True)
class PipelineHostDescriptor:
    """A picklable reference to the factory of a pipeline host.

    Attributes:
        factory: `"package.module:attribute"` path of a trusted callable
            returning a `PipelineHost`. It is fixed by the launcher, never
            taken from a request.
        settings: Plain, picklable keyword arguments for the factory.
    """

    factory: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        module_name, separator, attribute = self.factory.partition(":")
        if not (module_name and separator and attribute):
            raise ValueError(
                "Pipeline host factory must be given as 'package.module:attribute', "
                f"got {self.factory!r}."
            )


_DEFAULT_DESCRIPTOR: Optional[PipelineHostDescriptor] = None


def install_default_host_descriptor(descriptor: PipelineHostDescriptor) -> None:
    """Set the descriptor used when a caller does not pass one explicitly.

    Args:
        descriptor: Descriptor of this process's host.
    """
    global _DEFAULT_DESCRIPTOR
    _DEFAULT_DESCRIPTOR = descriptor


def get_default_host_descriptor() -> Optional[PipelineHostDescriptor]:
    """Return the installed default descriptor, if any."""
    return _DEFAULT_DESCRIPTOR


def resolve_host_descriptor(
    descriptor: Optional[PipelineHostDescriptor],
) -> PipelineHostDescriptor:
    """Return `descriptor`, or the installed default when it is `None`.

    Args:
        descriptor: Explicitly passed descriptor, if any.

    Returns:
        The descriptor to use.

    Raises:
        PipelineHostNotConfiguredError: Neither is available.
    """
    if descriptor is not None:
        return descriptor

    if _DEFAULT_DESCRIPTOR is None:
        raise PipelineHostNotConfiguredError(
            "The stream manager needs a pipeline host: pass `host_descriptor` "
            "explicitly or install a default with `install_default_host_descriptor` "
            "before starting it."
        )

    return _DEFAULT_DESCRIPTOR


def import_attribute(path: str) -> Any:
    """Import `"package.module:qualified.name"` and return the named object.

    Args:
        path: Module and qualified attribute name, separated by a colon.

    Returns:
        The attribute.
    """
    module_name, _, qualified_name = path.partition(":")
    target = importlib.import_module(module_name)
    for name in qualified_name.split("."):
        target = getattr(target, name)

    return target


def create_pipeline_host(descriptor: PipelineHostDescriptor) -> PipelineHost:
    """Build a host from its descriptor, in the process that will use it.

    Args:
        descriptor: Descriptor of the host.

    Returns:
        A new host; the caller closes it.
    """
    factory = import_attribute(descriptor.factory)
    host = factory(**descriptor.settings)

    return host
