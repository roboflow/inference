from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Data Aggregator",
            "version": "v1",
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )
    type: Literal["roboflow_core/data_aggregator@v1"]
    data: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References data to be used to construct each and every column",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
    )
    data_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL definitions of operations to be performed on defined data w.r.t. element of the data",
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
        default_factory=lambda: {},
    )
    aggregation_mode: Dict[
        str,
        List[
            Literal[
                "sum",
                "avg",
                "max",
                "min",
                "count",
                "distinct",
                "count_distinct",
                "values_counts",
            ]
        ],
    ]
    rolling_window: int = Field(description="Number of seconds to aggregate.")
    interval: int = Field(
        description="Aggregation results interval trigger (in seconds).",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="*")]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        result = []
        for variable_name in self.data.keys():
            for aggregation_mode in self.aggregation_mode.get(variable_name, []):
                if aggregation_mode == "distinct":
                    kind = [LIST_OF_VALUES_KIND]
                elif aggregation_mode == "values_counts":
                    kind = [DICTIONARY_KIND]
                else:
                    kind = [FLOAT_KIND, INTEGER_KIND]
                result.append(
                    OutputDefinition(
                        name=f"{variable_name}_{aggregation_mode}", kind=kind
                    )
                )
        return result

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class DataAggregatorBlockV1(WorkflowBlock):

    def __init__(self):
        self._open_aggregation_windows: Dict[datetime, Dict[str, AggregationState]] = {}
        self._start_timestamp = datetime.now()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        data: Dict[str, Any],
        data_operations: Dict[str, List[AllOperationsType]],
        aggregation_mode: Dict[
            str,
            List[
                Literal[
                    "sum",
                    "avg",
                    "max",
                    "min",
                    "count",
                    "distinct",
                    "count_distinct",
                    "values_counts",
                ]
            ],
        ],
        rolling_window: int,
        interval: int,
    ) -> BlockResult:
        if rolling_window % interval != 0:
            raise ValueError("Rolling window must be multiplier of interval")
        affected_windows = rolling_window // interval
        data = copy(data)
        for variable_name, operations in data_operations.items():
            operations_chain = build_operations_chain(operations=operations)
            data[variable_name] = operations_chain(
                data[variable_name], global_parameters={}
            )
        opened_windows_timestamps = sorted(self._open_aggregation_windows.keys())
        affected_timestamps = copy(opened_windows_timestamps)
        missing_windows = affected_windows - len(opened_windows_timestamps)
        now_timestamp = datetime.now()
        if opened_windows_timestamps:
            last_window_timestamp = opened_windows_timestamps[-1]
        else:
            time_elapsed = now_timestamp - self._start_timestamp
            elapsed_intervals = time_elapsed.total_seconds() // interval
            last_window_timestamp = self._start_timestamp + timedelta(
                seconds=elapsed_intervals * interval
            )
        for i in range(missing_windows):
            new_window_timestamp = last_window_timestamp + timedelta(
                seconds=(i + 1) * interval
            )
            self._open_aggregation_windows[new_window_timestamp] = initialize_states(
                data_names=data.keys(),
                aggregation_mode=aggregation_mode,
            )
            affected_timestamps.append(new_window_timestamp)
        for window in affected_timestamps:
            for variable_name, value in data.items():
                for mode in aggregation_mode.get(variable_name, []):
                    state_key = f"{variable_name}_{mode}"
                    print("Saving", window, state_key, value)
                    self._open_aggregation_windows[window][state_key].on_data(value)
        last_affected_timestamp_results = None
        for window in affected_timestamps:
            if window <= now_timestamp:
                last_affected_timestamp_results = {
                    k: v.get_result()
                    for k, v in self._open_aggregation_windows[window].items()
                }
                del self._open_aggregation_windows[window]
        print("last_affected_timestamp_results", last_affected_timestamp_results)
        if last_affected_timestamp_results is None:
            return {
                f"{variable_name}_{mode}": None
                for variable_name, value in data.items()
                for mode in aggregation_mode.get(variable_name, [])
            }
        return last_affected_timestamp_results


class AggregationState(ABC):

    @abstractmethod
    def on_data(self, value: Any):
        pass

    @abstractmethod
    def get_result(self) -> Any:
        pass


class SumState(AggregationState):

    def __init__(self):
        self._sum = 0

    def on_data(self, value: Any):
        self._sum += value

    def get_result(self) -> Any:
        return self._sum


class AVGState(AggregationState):

    def __init__(self):
        self._sum = 0
        self._n_items = 0

    def on_data(self, value: Any):
        self._sum += value
        self._n_items += 1

    def get_result(self) -> Any:
        if self._n_items > 0:
            return self._sum / self._n_items
        return None


class MaxState(AggregationState):

    def __init__(self):
        self._max: Optional[int] = None

    def on_data(self, value: Any):
        if self._max is None:
            self._max = value
        else:
            self._max = max(self._max, value)

    def get_result(self) -> Any:
        return self._max


class MinState(AggregationState):

    def __init__(self):
        self._min: Optional[int] = None

    def on_data(self, value: Any):
        if self._min is None:
            self._min = value
        else:
            self._min = min(self._min, value)

    def get_result(self) -> Any:
        return self._min


class CountState(AggregationState):

    def __init__(self):
        self._count = 0

    def on_data(self, value: Any):
        if hasattr(value, "__len__"):
            self._count += len(value)
        else:
            self._count += 1

    def get_result(self) -> Any:
        return self._count


class DistinctState(AggregationState):

    def __init__(self):
        self._distinct = set()

    def on_data(self, value: Any):
        if isinstance(value, list):
            for v in value:
                self._distinct.add(v)
            return None
        self._distinct.add(value)

    def get_result(self) -> Any:
        return list(self._distinct)


class CountDistinctState(AggregationState):

    def __init__(self):
        self._distinct = set()

    def on_data(self, value: Any):
        self._distinct.add(value)

    def get_result(self) -> Any:
        return list(self._distinct)


class ValuesCountState(AggregationState):

    def __init__(self):
        self._counts = defaultdict(int)

    def on_data(self, value: Any):
        if isinstance(value, list):
            for v in value:
                self._counts[v] += 1
            return None
        self._counts[value] += 1

    def get_result(self) -> Any:
        return dict(self._counts)


STATE_INITIALIZERS = {
    "sum": SumState,
    "avg": AVGState,
    "max": MaxState,
    "min": MinState,
    "count": CountState,
    "distinct": DistinctState,
    "count_distinct": CountDistinctState,
    "values_counts": ValuesCountState,
}


def initialize_states(
    data_names: Iterable[str],
    aggregation_mode: Dict[
        str,
        List[
            Literal[
                "sum",
                "avg",
                "max",
                "min",
                "count",
                "distinct",
                "count_distinct",
                "values_counts",
            ]
        ],
    ],
) -> Dict[str, AggregationState]:
    result = {}
    for data_name in data_names:
        for mode in aggregation_mode.get(data_name, []):
            state = STATE_INITIALIZERS[mode]()
            result[f"{data_name}_{mode}"] = state
    return result
