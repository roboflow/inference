"""User-facing switch of bounded pipelining.

Serial execution stays the default and the reference. Passing options turns
pipelining on::

    run = session.start(inputs, handlers=handlers, pipeline=PipelineOptions())
    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        future = pipeline.submit({"image": image})

Thread budget: ``max_in_flight`` (P) teams of ``max_workflow_parallelism`` (W)
workers each, P x W engine workers in total. A team executes one pulse or
submission; with W > 1 its workers run independent branches of that pulse.
Source readers, an active run's dispatcher and native-library threads are
not part of this budget.

``max_pending_operator_pulses`` is a safeguard, not a fourth scheduling
control: it fails an active run whose operators return more pulses than the
engine should hold waiting for a team.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, get_args

from roboflow_workflows.execution_engine.v2.errors import ContractError

__all__ = ["DEFAULT_MAX_PENDING_OPERATOR_PULSES", "OverloadPolicy", "PipelineOptions"]

DEFAULT_MAX_PENDING_OPERATOR_PULSES = 4096
"""Default of ``PipelineOptions.max_pending_operator_pulses``.

Above what supported graphs hold: operator domains deeper in the graph are
dispatched first, so a chain of operators returning ``f`` pulses per call
holds about ``f`` pulses per level and worker (512 x 512 at P = 1: 1023).
"""

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
            number of worker teams. At least 1.
        overload: Default overload policy of every source of an active run.
        source_overload: Overload policy per source name, overriding
            ``overload``. Names are checked against the plan at ``start``.
            Passive pipelines ignore overload policies: the caller submits.
        max_workflow_parallelism: Workers of each team: block calls one pulse
            may run at once on independent branches of its workflow. The
            worker continuing a branch counts. ``1`` (default) runs the steps
            of a pulse one after another in plan order. At least 1.
            W schedules independent graph units, not members within one step's
            batch; ordinary serial ``session.run()`` does not use it. Keyword-only.
        max_pending_operator_pulses: Operator pulses an active run may hold
            returned but not yet started, summed over its operators. An
            operator call whose returned pulses would exceed it fails the
            run, attributed to that operator; none of them runs and none is
            dropped silently. Counts pulses, not bytes: the list an operator
            returned is already allocated. Raise it for operators that
            legitimately return longer lists, e.g. a ``finish`` flushing a
            long buffer. Serial runs and passive pipelines ignore it. At
            least 1; keyword-only.

    Raises:
        ContractError: When a value has the wrong type or range.
    """

    max_in_flight: int = 4
    overload: OverloadPolicy = "block"
    source_overload: Mapping[str, OverloadPolicy] = field(default_factory=dict)
    # Keyword-only and last: the earlier positional order stays valid.
    max_workflow_parallelism: int = field(default=1, kw_only=True)
    max_pending_operator_pulses: int = field(
        default=DEFAULT_MAX_PENDING_OPERATOR_PULSES, kw_only=True
    )

    def __post_init__(self) -> None:
        _check_count(self.max_in_flight, where="max_in_flight")
        _check_count(self.max_workflow_parallelism, where="max_workflow_parallelism")
        _check_count(
            self.max_pending_operator_pulses, where="max_pending_operator_pulses"
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


def _check_count(value: object, *, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"PipelineOptions {where} must be an int, got {value!r}")
    if value < 1:
        raise ContractError(f"PipelineOptions {where} must be at least 1, got {value}")


def _check_policy(policy: object, *, where: str) -> None:
    if policy not in _POLICIES:
        raise ContractError(
            f"PipelineOptions {where} must be one of {list(_POLICIES)}, got "
            f"{policy!r}"
        )
