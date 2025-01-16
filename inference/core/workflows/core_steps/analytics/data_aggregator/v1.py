from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from datetime import datetime
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
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **Data Aggregator** block collects and processes data from Workflows to generate time-based statistical 
summaries. It allows users to define custom aggregation strategies over specified intervals, making it suitable 
for creating **analytics on data streams**.

The block enables:

* feeding it with data from other Workflow blocks and applying in-place operations (for instance to extract 
desired values out of model predictions)

* using multiple aggregation modes, including `sum`, `avg`, `max`, `min`, `count` and others

* specifying aggregation interval flexibly

### Feeding Data Aggregator

You can specify the data to aggregate by referencing input sources using the `data` field. Optionally,
for each specified `data` input you can apply chain of UQL operations with `data_operations` property.

For example, the following configuration:

```
data = {
    "predictions_model_a": "$steps.model_a.predictions",
    "predictions_model_b": "$steps.model_b.predictions",
}
data_operations = { 
    "predictions_model_a": [
        {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
    ],
    "predictions_model_b": [{"type": "SequenceLength"}]
}
```

on each step run will at first take `predictions_model_a` to extract list of detected classes
and calculate the number of predicted bounding boxes for `predictions_model_b`.

### Specifying data aggregations

For each input data referenced by `data` property you can specify list of aggregation operations, that
include:

* **`sum`**: Taking the sum of values (requires data to be numeric)

* **`avg`**: Taking the average of values (requires data to be numeric)

* **`max`**: Taking the max of values (requires data to be numeric)

* **`min`**: Taking the min of values (requires data to be numeric)

* **`count`**: Counting the values - if provided value is list - operation will add length of the list into 
aggregated state

* **`distinct`:** deduplication of encountered values - providing list of unique values in the output. If 
aggregation data is list - operation will add each element of the list into aggregated state.

* **`count_distinct`:** counting occurrences of distinct values - providing number of different values that were 
encountered. If aggregation data is list - operation will add each element of the list into aggregated state.

* **`count_distinct`:** counting distinct values - providing number of different values that were 
encountered. If aggregation data is list - operation will add each element of the list into aggregated state.

* **`values_counts`:** counting occurrences of each distinct value - providing dictionary mapping each unique value 
encountered into the number of observations. If aggregation data is list - operation will add each element of the list 
into aggregated state.

* **`values_difference`:** calculates the difference between max and min observed value (requires data to be numeric)

If we take the `data` and `data_operations` from the example above and specify `aggregation_mode` in the following way:

```
aggregation_mode = {
    "predictions_model_a": ["distinct", "count_distinct"],
    "predictions_model_b": ["avg"],
}
``` 

Our aggregation report will contain the following values:

```
{
    "predictions_model_a_distinct": ["car", "person", "dog"],
    "predictions_model_a_count_distinct": {"car": 378, "person": 128, "dog": 37},
    "predictions_model_b_avg": 7.35,
}
``` 

where:

* `predictions_model_a_distinct` provides distinct classes predicted by model A in aggregation window

* `predictions_model_a_count_distinct` provides number of classes instances predicted by model A in aggregation 
window

* `predictions_model_b_avg` provides average number of bounding boxes predicted by model B in aggregation window

### Interval nature of the block

!!! warning "Block behaviour is dictated by internal 'clock'"

    Behaviour of this block differs from other, more classical blocks which output the data for each input.
    **Data Aggregator** block maintains its internal state that dictates when the data will be produced, 
    flushing internal aggregation state of the block. 
    
    You can expect that most of the times, once fed with data, the block will produce empty outputs,
    effectively terminating downstream processing:
    
    ```
    --- input_batch[0] ----> ┌───────────────────────┐ ---->  <Empty>
    --- input_batch[1] ----> │                       │ ---->  <Empty>
            ...              │     Data Aggregator   │ ---->  <Empty>
            ...              │                       │ ---->  <Empty>           
    --- input_batch[n] ----> └───────────────────────┘ ---->  <Empty>
    ```  
    
    But once for a while, the block will yield aggregated data and flush its internal state:
    
    ```
    --- input_batch[0] ----> ┌───────────────────────┐ ---->  <Empty>
    --- input_batch[1] ----> │                       │ ---->  <Empty>
            ...              │     Data Aggregator   │ ---->  {<aggregated_report>}
            ...              │                       │ ---->  <Empty> # first datapoint added to new state          
    --- input_batch[n] ----> └───────────────────────┘ ---->  <Empty>
    ```
     
Setting the aggregation interval is possible with `interval` and `interval_unit` property.
`interval` specifies the length of aggregation window and `interval_unit` bounds the `interval` value 
into units. You can specify the interval based on:

* **time elapse:** using `["seconds", "minutes", "hours"]` as `interval_unit` will make the 
**Data Aggregator** to yield the aggregated report based on time that elapsed since last report 
was released - this setting is relevant for **processing of video streams**.

* **number of runs:** using `runs` as `interval_unit` - this setting is relevant for 
**processing of video files**, as in this context wall-clock time elapse is not the proper way of getting
meaningful reports.
"""


AggregationType = Literal[
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


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Data Aggregator",
            "version": "v1",
            "short_description": "Aggregate workflow data to produce time-based statistics.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-database",
                "blockPriority": 4,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/data_aggregator@v1"]
    data: Dict[str, Selector()] = Field(
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
        json_schema_extra={
            "uses_uql": True,
            "keys_bound_in": "data",
        },
    )
    aggregation_mode: Dict[str, List[AggregationType]] = Field(
        description="Lists of aggregation operations to apply on each input data",
        examples=[{"predictions": ["distinct", "count_distinct"]}],
        json_schema_extra={
            "keys_bound_in": "data",
            "values_metadata": {
                "sum": {
                    "name": "Sum",
                    "description": "Sums values in aggregation interval (requires data to be numeric)",
                },
                "avg": {
                    "name": "Average",
                    "description": "Averages out values in aggregation interval (requires data to be numeric)",
                },
                "max": {
                    "name": "Take Maximum Value",
                    "description": "Takes maximum value encountered in aggregation interval "
                    "(requires data to be numeric)",
                },
                "min": {
                    "name": "Take Minimum Value",
                    "description": "Takes minimum value encountered in aggregation interval "
                    "(requires data to be numeric)",
                },
                "count": {
                    "name": "Count Values",
                    "description": "Counts the encountered values - for list-like data, "
                    "operation will add length of the list into aggregated state",
                },
                "distinct": {
                    "name": "Take Distinct Values",
                    "description": "Provides list of unique values encountered - for list-like data, "
                    "operation will add each value from the list into aggregated state",
                },
                "count_distinct": {
                    "name": "Number Of Distinct Values",
                    "description": "Provides the number of distinct values encountered - for list-like data, "
                    "operation will add each value from the list into aggregated state",
                },
                "values_counts": {
                    "name": "Count Distinct Values Occurrences",
                    "description": "Counts occurrences of each unique value encountered - for list-like data,"
                    "operation will add each element of the list into aggregated state.",
                },
                "values_difference": {
                    "name": "Observed Values Difference",
                    "description": "Calculate difference between max and min value observed.",
                },
            },
        },
    )
    interval_unit: Literal["seconds", "minutes", "hours", "runs"] = Field(
        default="seconds",
        description="Unit to measure `interval`",
        examples=["seconds", "hours"],
        json_schema_extra={
            "always_visible": True,
            "values_metadata": {
                "seconds": {
                    "name": "Seconds",
                    "description": "Interval based on number of seconds elapsed",
                },
                "minutes": {
                    "name": "Minutes",
                    "description": "Interval based on number of minutes elapsed",
                },
                "hours": {
                    "name": "Hours",
                    "description": "Interval based on number of hours elapsed",
                },
                "runs": {
                    "name": "Step runs",
                    "description": "Interval based on number of data elements flowing though the "
                    "step (for example: number of processed video frames).",
                },
            },
        },
    )
    interval: int = Field(
        description="Length of aggregation interval",
        examples=[10, 100],
        gt=0,
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
        return ">=1.3.0,<2.0.0"


INTERVAL_UNIT_TO_SECONDS = {
    "seconds": 1,
    "minutes": 60,
    "hours": 60 * 60,
}


class DataAggregatorBlockV1(WorkflowBlock):

    def __init__(self):
        self._aggregation_cache: Dict[str, AggregationState] = {}
        self._aggregation_start_timestamp = datetime.now()
        self._runs = 0

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        data: Dict[str, Any],
        data_operations: Dict[str, List[AllOperationsType]],
        aggregation_mode: Dict[str, List[AggregationType]],
        interval_unit: Literal["seconds", "minutes", "hours", "runs"],
        interval: int,
    ) -> BlockResult:
        self._aggregation_cache = ensure_states_initialised(
            aggregation_cache=self._aggregation_cache,
            data_names=data.keys(),
            aggregation_mode=aggregation_mode,
        )
        self._runs += 1
        data = copy(data)
        for variable_name, operations in data_operations.items():
            operations_chain = build_operations_chain(operations=operations)
            data[variable_name] = operations_chain(
                data[variable_name], global_parameters={}
            )
        for variable_name, value in data.items():
            for mode in aggregation_mode.get(variable_name, []):
                state_key = generate_state_key(
                    field_name=variable_name, aggregation_mode=mode
                )
                self._aggregation_cache[state_key].on_data(value=value)
        seconds_since_last_dump = (
            datetime.now() - self._aggregation_start_timestamp
        ).total_seconds()
        if interval_unit == "runs":
            it_is_time_to_flush = self._runs >= interval
        else:
            interval_seconds = interval * INTERVAL_UNIT_TO_SECONDS[interval_unit]
            it_is_time_to_flush = seconds_since_last_dump >= interval_seconds
        if not it_is_time_to_flush:
            return {k: None for k in self._aggregation_cache.keys()}
        result = {k: v.get_result() for k, v in self._aggregation_cache.items()}
        self._aggregation_cache = {}
        self._runs = 0
        self._aggregation_start_timestamp = datetime.now()
        return result


class AggregationState(ABC):

    @abstractmethod
    def on_data(self, value: Any) -> None:
        pass

    @abstractmethod
    def get_result(self) -> Any:
        pass


class SumState(AggregationState):

    def __init__(self):
        self._sum = 0

    def on_data(self, value: Any) -> None:
        self._sum += value

    def get_result(self) -> Any:
        return self._sum


class AVGState(AggregationState):

    def __init__(self):
        self._sum = 0
        self._n_items = 0

    def on_data(self, value: Any) -> None:
        self._sum += value
        self._n_items += 1

    def get_result(self) -> Any:
        if self._n_items > 0:
            return self._sum / self._n_items
        return None


class MaxState(AggregationState):

    def __init__(self):
        self._max: Optional[int] = None

    def on_data(self, value: Any) -> None:
        if self._max is None:
            self._max = value
        else:
            self._max = max(self._max, value)

    def get_result(self) -> Any:
        return self._max


class MinState(AggregationState):

    def __init__(self):
        self._min: Optional[int] = None

    def on_data(self, value: Any) -> None:
        if self._min is None:
            self._min = value
        else:
            self._min = min(self._min, value)

    def get_result(self) -> Any:
        return self._min


class CountState(AggregationState):

    def __init__(self):
        self._count = 0

    def on_data(self, value: Any) -> None:
        if hasattr(value, "__len__"):
            self._count += len(value)
        else:
            self._count += 1

    def get_result(self) -> Any:
        return self._count


class DistinctState(AggregationState):

    def __init__(self):
        self._distinct = set()

    def on_data(self, value: Any) -> None:
        if (
            isinstance(value, list)
            or isinstance(value, set)
            or isinstance(value, tuple)
        ):
            for v in value:
                self._distinct.add(v)
            return None
        self._distinct.add(value)

    def get_result(self) -> Any:
        return list(self._distinct)


class CountDistinctState(AggregationState):

    def __init__(self):
        self._distinct = set()

    def on_data(self, value: Any) -> None:
        if (
            isinstance(value, list)
            or isinstance(value, set)
            or isinstance(value, tuple)
        ):
            for v in value:
                self._distinct.add(v)
            return None
        self._distinct.add(value)

    def get_result(self) -> Any:
        return len(self._distinct)


class ValuesCountState(AggregationState):

    def __init__(self):
        self._counts = defaultdict(int)

    def on_data(self, value: Any) -> None:
        if isinstance(value, list):
            for v in value:
                self._counts[v] += 1
            return None
        self._counts[value] += 1

    def get_result(self) -> Any:
        return dict(self._counts)


class ValuesDifferenceState(AggregationState):

    def __init__(self):
        self._min_value: Optional[Union[int, float]] = None
        self._max_value: Optional[Union[int, float]] = None

    def on_data(self, value: Any) -> None:
        if self._min_value is None:
            self._min_value = value
            return None
        if self._max_value is None:
            self._max_value = value
            return None
        self._min_value = min(self._min_value, value)
        self._max_value = max(self._max_value, value)

    def get_result(self) -> Any:
        if self._min_value is None or self._max_value is None:
            return None
        return self._max_value - self._min_value


STATE_INITIALIZERS = {
    "sum": SumState,
    "avg": AVGState,
    "max": MaxState,
    "min": MinState,
    "count": CountState,
    "distinct": DistinctState,
    "count_distinct": CountDistinctState,
    "values_counts": ValuesCountState,
    "values_difference": ValuesDifferenceState,
}


def ensure_states_initialised(
    aggregation_cache: Dict[str, AggregationState],
    data_names: Iterable[str],
    aggregation_mode: Dict[str, List[AggregationType]],
) -> Dict[str, AggregationState]:
    for data_name in data_names:
        for mode in aggregation_mode.get(data_name, []):
            state_key = generate_state_key(field_name=data_name, aggregation_mode=mode)
            if state_key in aggregation_cache:
                continue
            state = STATE_INITIALIZERS[mode]()
            aggregation_cache[f"{data_name}_{mode}"] = state
    return aggregation_cache


def generate_state_key(field_name: str, aggregation_mode: str) -> str:
    return f"{field_name}_{aggregation_mode}"
