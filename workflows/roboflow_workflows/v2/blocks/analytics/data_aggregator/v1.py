"""Data aggregator block: statistics of workflow values over an interval.

Port of V1 ``roboflow_core/data_aggregator@v1``::

    each call:  data[name] ── data_operations ──▶ value ─▶ every state of name
    interval elapsed (time, or number of calls)?
        no  ─▶ interval_complete=False, every statistic None
        yes ─▶ interval_complete=True, every statistic set; states and
               interval restart

Outputs are ``interval_complete`` plus ``<name>_<mode>`` for each mode in
``aggregation_mode[name]``, e.g. ``count_sum``. Gate a sink on
``interval_complete`` (``roboflow_core/continue_if_boolean@v1``) to write only completed intervals.

Scope is global, as in V1: one aggregation state per step combines every
input and every video source, and ``runs`` counts calls from all of them. A
processing reset starts a new state. The block keeps that state, so its steps
are never pruned.

Values: Python numbers, strings and sequences as in V1. A zero-dimensional
tensor or array becomes its Python number and a larger one its nested list
(a device-to-host copy); other objects, e.g. ``Detections``, are kept and
only ``count`` (their length) accepts them.

Differences from V1: aggregation_mode keys must be data keys, a mode may
appear once per name and output names may not collide (V1 ignored, double
counted or merged these); ``count`` of a string is 1, not its length;
``values_counts`` expands tuples and sets like ``distinct``; time intervals
use a monotonic clock; the ``interval_complete`` output is new, so no
statistic may be named that.
"""

import numbers
import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple
from pydantic import Field, StrictInt, ValidationInfo, field_validator
from roboflow_workflows.v2.blocks.query import (
    CompiledOperations,
    Operation,
    compile_operations,
)
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


BLOCK_TYPE = "roboflow_core/data_aggregator@v1"

AggregationMode = Literal[
    "sum",
    "avg",
    "max",
    "min",
    "count",
    "distinct",
    "count_distinct",
    "values_counts",
    "values_difference",
]

INTERVAL_COMPLETE = "interval_complete"

INTERVAL_COMPLETE_OUTPUT = Output(
    BOOLEAN_KIND,
    description=(
        "True on the call that ends an interval and carries the statistics; "
        "False on every other call. Gate sinks on it."
    ),
)

SECONDS_PER_UNIT = {"seconds": 1, "minutes": 60, "hours": 60 * 60}

NUMERIC_MODES = frozenset({"sum", "avg", "max", "min", "values_difference"})

MODE_KINDS = {
    "count": INTEGER_KIND,
    "count_distinct": INTEGER_KIND,
    "distinct": LIST_OF_VALUES_KIND,
    "values_counts": DICTIONARY_KIND,
}

class DataAggregatorBlock(Block):
    """Aggregate values over a time interval or a number of calls.

    One interval is shared by all inputs and video sources of the step: two
    cameras feed the same totals, and ``runs`` counts calls from both.

    Example: ``data={"count": "$steps.counter.count"}``,
    ``aggregation_mode={"count": ["sum", "max"]}``, ``interval_unit="runs"``
    and ``interval=3`` output ``count_sum`` and ``count_max`` on every third
    call (``None`` on the others): for counts 2, 5, 1 they are 8 and 5.
    ``interval_complete`` is True on that third call only.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Data Aggregator",
        "port_of": "roboflow_core/data_aggregator@v1",
        "section": "data_storage",
    }
    outputs = {INTERVAL_COMPLETE: INTERVAL_COMPLETE_OUTPUT}
    output_fields = ("aggregation_mode",)

    class Params(BlockParams):
        data: Dict[str, Ref()] = Field(  # noqa: F821
            description="Values to aggregate, by name; each a selector.",
            examples=[{"predictions": "$steps.model.predictions"}],
        )
        data_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description=(
                "UQL operations applied to a data value first, by data name; "
                "e.g. extract class names from predictions."
            ),
            examples=[
                {
                    "predictions": [
                        {
                            "type": "DetectionsPropertyExtract",
                            "property_name": "class_name",
                        }
                    ]
                }
            ],
        )
        aggregation_mode: Dict[str, List[AggregationMode]] = Field(
            description=(
                "Statistics per data name. Each makes the output "
                "<name>_<mode>: sum, avg, max, min and values_difference need "
                "numbers; count adds a sequence's length or 1; distinct, "
                "count_distinct and values_counts expand sequences."
            ),
            examples=[{"predictions": ["distinct", "count_distinct"]}],
        )
        interval_unit: Literal["seconds", "minutes", "hours", "runs"] = Field(
            default="seconds",
            description=(
                "Unit of interval: elapsed time, or 'runs', the number of "
                "calls (e.g. video frames). The interval is global: 'runs' "
                "counts calls from all sources together."
            ),
            examples=["seconds"],
        )
        interval: StrictInt = Field(
            gt=0,
            description="Length of one aggregation interval in interval_unit.",
            examples=[10],
        )

        @field_validator("data_operations")
        @classmethod
        def _operations_compile(
            cls, value: Dict[str, List[Operation]], info: ValidationInfo
        ) -> Dict[str, List[Operation]]:
            if "data" in info.data:
                compile_data_operations(value, data_names=tuple(info.data["data"]))

            return value

        @field_validator("aggregation_mode")
        @classmethod
        def _modes_name_data(
            cls, value: Dict[str, List[str]], info: ValidationInfo
        ) -> Dict[str, List[str]]:
            if "data" in info.data:
                output_names(value, data_names=tuple(info.data["data"]))

            return value

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Mapping[str, Output]:
        """Declare ``interval_complete`` and every configured statistic.

        Args:
            params: Validated step parameters.

        Returns:
            ``interval_complete``, then ``<name>_<mode>`` outputs in
            ``aggregation_mode`` order.
        """
        statistics = {
            output: Output(
                MODE_KINDS.get(mode, FLOAT_KIND),
                description=f"{mode} of {name} over the interval; None before its end.",
            )
            for output, (name, mode) in output_names(params.aggregation_mode).items()
        }
        outputs = {INTERVAL_COMPLETE: INTERVAL_COMPLETE_OUTPUT, **statistics}

        return outputs

    def __init__(
        self, *, aggregation_clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._clock = aggregation_clock
        self._lock = threading.Lock()
        self._states: Dict[str, "_State"] = {}
        self._runs = 0
        self._started = aggregation_clock()
        self._compiled: Optional[Tuple[Any, Dict[str, CompiledOperations]]] = None

    def run(
        self,
        *,
        data: Dict[str, Any],
        data_operations: Dict[str, List[Any]],
        aggregation_mode: Dict[str, List[str]],
        interval_unit: str,
        interval: int,
    ) -> Dict[str, Any]:
        """Add this call's values; output the statistics when the interval ends.

        Args:
            data: Values by name.
            data_operations: Operations by data name.
            aggregation_mode: Statistics by data name.
            interval_unit: ``seconds``, ``minutes``, ``hours`` or ``runs``.
            interval: Interval length in that unit.

        Returns:
            ``interval_complete`` and every ``<name>_<mode>`` output: True and
            the statistics at the interval's end, else False and ``None``.

        Raises:
            TypeError: When a value does not suit a statistic, e.g. a string
                for ``sum`` or an unhashable value for ``distinct``.
            QueryRuntimeError: When an operation cannot process a value.
        """
        values = dict(data)
        for name, chain in self._operations(data, data_operations).items():
            values[name] = chain.apply(values[name], parameters={})
        outputs = output_names(aggregation_mode)

        with self._lock:
            for output, (name, mode) in outputs.items():
                state = self._states.setdefault(output, _State(mode))
                state.add(plain(values[name]), name=name)
            self._runs += 1
            if interval_unit == "runs":
                ended = self._runs >= interval
            else:
                elapsed = self._clock() - self._started
                ended = elapsed >= interval * SECONDS_PER_UNIT[interval_unit]
            if not ended:
                return {INTERVAL_COMPLETE: False, **dict.fromkeys(outputs)}

            result = {INTERVAL_COMPLETE: True}
            for output in outputs:
                result[output] = self._states[output].result()
            self._states = {}
            self._runs = 0
            self._started = self._clock()

        return result

    def _operations(
        self, data: Dict[str, Any], data_operations: Dict[str, List[Any]]
    ) -> Dict[str, CompiledOperations]:
        key = (tuple(data), data_operations)
        cached = self._compiled
        if cached is None or cached[0] != key:
            cached = (key, compile_data_operations(data_operations, data_names=key[0]))
            self._compiled = cached

        return cached[1]

def compile_data_operations(
    data_operations: Dict[str, List[Any]], *, data_names: Tuple[str, ...]
) -> Dict[str, CompiledOperations]:
    """Compile per-name operation chains that name no runtime values.

    Args:
        data_operations: Operations by data name.
        data_names: Keys of ``data``.

    Returns:
        Compiled chains by data name.

    Raises:
        ValueError: When a key is not a data name.
        QueryDefinitionError: On an invalid chain.
    """
    compiled = {}
    for name, operations in data_operations.items():
        if name not in data_names:
            raise ValueError(
                f"{BLOCK_TYPE}: data_operations names `{name}`, which is not a "
                f"data name (names: {list(data_names)})"
            )
        compiled[name] = compile_operations(
            operations, path=f"$.data_operations.{name}", declared_names=frozenset()
        )

    return compiled

def output_names(
    aggregation_mode: Mapping[str, List[str]],
    *,
    data_names: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Tuple[str, str]]:
    """Map each output name to its data name and statistic.

    Args:
        aggregation_mode: Statistics by data name.
        data_names: Keys of ``data`` to check against; ``None`` skips that.

    Returns:
        ``{"<name>_<mode>": (name, mode)}`` in configuration order.

    Raises:
        ValueError: On an unknown data name, a repeated statistic, two
            statistics producing the same output name, or a statistic named
            ``interval_complete``.
    """
    outputs: Dict[str, Tuple[str, str]] = {}
    for name, modes in aggregation_mode.items():
        if data_names is not None and name not in data_names:
            raise ValueError(
                f"{BLOCK_TYPE}: aggregation_mode names `{name}`, which is not a "
                f"data name (names: {list(data_names)})"
            )
        for mode in modes:
            output = f"{name}_{mode}"
            if output == INTERVAL_COMPLETE:
                raise ValueError(
                    f"{BLOCK_TYPE}: output `{output}` is reserved for the "
                    "interval flag; rename the data"
                )
            if output in outputs:
                raise ValueError(
                    f"{BLOCK_TYPE}: output `{output}` is produced twice; "
                    f"repeat no statistic and rename data so names stay distinct"
                )
            outputs[output] = (name, mode)

    return outputs

def plain(value: Any) -> Any:
    """Turn a tensor or array into Python values; keep anything else.

    Args:
        value: One data value after its operations.

    Returns:
        A Python number for a zero-dimensional tensor or array, a nested
        list for a larger one, else ``value`` itself.
    """
    tolist = getattr(value, "tolist", None)
    if tolist is None or not hasattr(value, "ndim") or isinstance(value, str):
        return value

    converted = tolist()

    return converted

class _State:
    """Running statistic of one output over the current interval."""

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._count = 0
        self._total: Any = 0
        self._low: Any = None
        self._high: Any = None
        self._seen: Counter = Counter()

    def add(self, value: Any, *, name: str) -> None:
        mode = self._mode
        if mode in NUMERIC_MODES:
            if not isinstance(value, numbers.Real):
                raise TypeError(
                    f"{BLOCK_TYPE}: {mode} of `{name}` needs numbers, got "
                    f"{type(value).__name__}; add data_operations to extract one"
                )
            self._count += 1
            self._total += value
            self._low = value if self._low is None else min(self._low, value)
            self._high = value if self._high is None else max(self._high, value)
        elif mode == "count":
            sized = hasattr(value, "__len__") and not isinstance(value, str)
            self._count += len(value) if sized else 1
        else:
            items = value if isinstance(value, (list, tuple, set)) else [value]
            try:
                self._seen.update(items)
            except TypeError as error:
                raise TypeError(
                    f"{BLOCK_TYPE}: {mode} of `{name}` needs hashable values: {error}"
                ) from error

    def result(self) -> Any:
        mode = self._mode
        if mode == "sum":
            return self._total
        if mode == "avg":
            return self._total / self._count if self._count else None
        if mode == "max":
            return self._high
        if mode == "min":
            return self._low
        if mode == "values_difference":
            return None if self._low is None else self._high - self._low
        if mode == "count":
            return self._count
        if mode == "distinct":
            return list(self._seen)
        if mode == "count_distinct":
            return len(self._seen)

        # values_counts, the last mode
        counts = dict(self._seen)

        return counts
