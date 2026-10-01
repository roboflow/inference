"""User-facing switch of bounded pipelining.

Serial execution stays the default and the reference. Passing options turns
pipelining on::

    run = session.start(inputs, handlers=handlers, pipeline=PipelineOptions())
    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        future = pipeline.submit({"image": image})
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, get_args

from roboflow_workflows.execution_engine.v2.errors import ContractError

__all__ = ["OverloadPolicy", "PipelineOptions"]

OverloadPolicy = Literal["block", "latest"]
"""What a source reader does when its source has no free admission slot.

``block``: wait for a slot; every read value is processed (lossless).
``latest``: keep only the newest read value pending; replaced values are
counted as ``dropped``. Admitted work is never dropped.
"""

_POLICIES = get_args(OverloadPolicy)


@dataclass(frozen=True)
class PipelineOptions:
    """Bounds and overload policy of a pipelined run.

    Args:
        max_in_flight: Pulses (or passive submissions) executing at once; the
            number of pipeline workers. At least 1.
        overload: Default overload policy of every source of an active run.
        source_overload: Overload policy per source name, overriding
            ``overload``. Names are checked against the plan at ``start``.
            Passive pipelines ignore overload policies: the caller submits.

    Raises:
        ContractError: When a value has the wrong type or range.
    """

    max_in_flight: int = 4
    overload: OverloadPolicy = "block"
    source_overload: Mapping[str, OverloadPolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.max_in_flight, bool) or not isinstance(
            self.max_in_flight, int
        ):
            raise ContractError(
                f"PipelineOptions max_in_flight must be an int, got "
                f"{self.max_in_flight!r}"
            )
        if self.max_in_flight < 1:
            raise ContractError(
                f"PipelineOptions max_in_flight must be at least 1, got "
                f"{self.max_in_flight}"
            )

        _check_policy(self.overload, where="overload")
        if not isinstance(self.source_overload, Mapping):
            raise ContractError(
                f"PipelineOptions source_overload must map source names to "
                f"{list(_POLICIES)}, got {type(self.source_overload).__name__}"
            )
        for name, policy in self.source_overload.items():
            if not isinstance(name, str) or not name:
                raise ContractError(
                    f"PipelineOptions source_overload keys are source names, got "
                    f"{name!r}"
                )
            _check_policy(policy, where=f"source_overload[{name!r}]")

        frozen = MappingProxyType(dict(self.source_overload))
        object.__setattr__(self, "source_overload", frozen)

    def overload_for(self, source: str) -> OverloadPolicy:
        """Return the overload policy of one source.

        Args:
            source: Declared source name.

        Returns:
            The source's own policy, else ``overload``.
        """
        policy = self.source_overload.get(source, self.overload)

        return policy

    def check_sources(self, sources: Iterable[str]) -> None:
        """Reject ``source_overload`` names that are not declared sources.

        Args:
            sources: Source names the plan declares.

        Raises:
            ContractError: When a configured name is not among ``sources``.
        """
        declared = set(sources)
        unknown = sorted(set(self.source_overload) - declared)
        if unknown:
            raise ContractError(
                f"PipelineOptions source_overload names unknown source(s) "
                f"{unknown}; declared sources: {sorted(declared)}"
            )


def _check_policy(policy: object, *, where: str) -> None:
    if policy not in _POLICIES:
        raise ContractError(
            f"PipelineOptions {where} must be one of {list(_POLICIES)}, got "
            f"{policy!r}"
        )
