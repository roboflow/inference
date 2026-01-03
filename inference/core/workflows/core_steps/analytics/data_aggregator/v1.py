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
Collect and process data from workflow steps over configurable time-based or run-based intervals to generate statistical summaries and analytics reports, supporting multiple aggregation operations (sum, average, max, min, count, distinct values, value counts) with optional UQL-based data transformations for comprehensive data stream analytics.

## How This Block Works

This block collects and aggregates data from workflow steps over specified intervals to produce statistical summaries. Unlike most blocks that output data for every input, this block maintains internal state and outputs aggregated results only when the configured interval is reached. The block:

1. Receives data inputs from other workflow steps (via `data` field mapping variable names to workflow step outputs)
2. Optionally applies UQL (Query Language) operations to transform the data before aggregation (e.g., extract class names from detections, calculate sequence lengths, filter or transform values) using `data_operations` for each input variable
3. Accumulates data into internal aggregation states based on the specified `aggregation_mode` for each variable
4. Tracks time elapsed or number of runs based on `interval_unit` (seconds, minutes, hours, or runs)
5. Most of the time, returns empty outputs (terminating downstream processing) while collecting data internally
6. When the interval threshold is reached (based on time elapsed or run count), computes and outputs aggregated statistics
7. Flushes internal state after outputting aggregated results and starts collecting data for the next interval
8. Produces output fields dynamically named as `{variable_name}_{aggregation_mode}` (e.g., `predictions_avg`, `classes_distinct`, `count_values_counts`)

The block supports multiple aggregation modes for numeric data (`sum`, `avg`, `max`, `min`, `values_difference`), counting operations (`count`, `count_distinct`), and value analysis (`distinct`, `values_counts`). For list-like data, operations automatically process each element (e.g., `count` adds list length, `distinct` adds each element to the distinct set). The interval can be time-based (useful for video streams where wall-clock time matters) or run-based (useful for video file processing where frame count matters more than elapsed time).

## Common Use Cases

- **Video Stream Analytics**: Aggregate detection results over time intervals from live video streams (e.g., calculate average object counts per minute, track distinct classes seen per hour, compute min/max detection counts over 30-second windows), enabling real-time analytics and monitoring for continuous video processing workflows
- **Batch Video Processing**: Aggregate statistics across video frames using run-based intervals (e.g., calculate average detections per 100 frames, count distinct objects across 500-frame windows, sum total detections per batch), enabling meaningful analytics for pre-recorded video files where frame count matters more than elapsed time
- **Time-Series Metrics Collection**: Collect and summarize workflow metrics over time (e.g., aggregate detection counts, calculate average confidence scores, track distinct class occurrences, compute value distributions), enabling statistical analysis and reporting for production workflows
- **Model Performance Analysis**: Analyze model predictions across multiple inputs (e.g., calculate average prediction counts, track distinct predicted classes, compute min/max confidence scores, count occurrences of each class), enabling comprehensive model performance evaluation and insights
- **Data Stream Summarization**: Summarize high-frequency data streams into periodic reports (e.g., aggregate every 60 seconds of detections into summary statistics, compute hourly averages, generate per-run summaries), enabling efficient data reduction and analysis for high-volume workflows
- **Multi-Model Comparison**: Aggregate results from multiple models for comparison (e.g., compare average detection counts across models, track distinct classes per model, compute aggregate statistics for model ensembles), enabling comparative analytics across different inference pipelines

## Connecting to Other Blocks

This block receives data from workflow steps and outputs aggregated statistics periodically:

- **After detection or analysis blocks** (e.g., Object Detection, Instance Segmentation, Classification) to aggregate prediction results over time or across frames, enabling statistical analysis of model outputs and detection patterns
- **After data processing blocks** (e.g., Expression, Property Definition, Detections Filter) that produce numeric or list outputs to aggregate computed values, metrics, or transformed data over intervals
- **Before sink blocks** (e.g., CSV Formatter, Local File Sink, Webhook Sink) to save periodic aggregated reports, enabling efficient storage and export of summarized analytics data instead of individual data points
- **In video processing workflows** to generate time-based or frame-based analytics reports, enabling comprehensive video analysis with periodic statistical summaries rather than per-frame outputs
- **Before visualization or reporting blocks** that need aggregated data to create dashboards, charts, or summaries from time-series data, enabling visualization of trends and statistics
- **In analytics pipelines** where high-frequency data needs to be reduced to periodic summaries, enabling efficient downstream processing and storage of statistical insights rather than raw high-volume data streams
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
        description="Dictionary mapping variable names to data sources from workflow steps. Each key becomes a variable name for aggregation, and each value is a selector referencing workflow step outputs (e.g., predictions, metrics, computed values). These variables are used in aggregation_mode to specify which aggregations to compute. Example: {'predictions': '$steps.model.predictions', 'count': '$steps.counter.total'}.",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
    )
    data_operations: Dict[str, List[AllOperationsType]] = Field(
        description="Optional dictionary mapping variable names (from data) to UQL (Query Language) operation chains that transform data before aggregation. Operations are applied in sequence to extract, filter, or transform values (e.g., extract class names from detections using DetectionsPropertyExtract, calculate sequence length using SequenceLength, filter values, perform calculations). Keys must match variable names in data. Leave empty or omit variables that don't need transformation. Example: {'predictions': [{'type': 'DetectionsPropertyExtract', 'property_name': 'class_name'}]}.",
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
        description="Dictionary mapping variable names (from data) to lists of aggregation operations to compute. Each aggregation produces an output field named '{variable_name}_{aggregation_mode}'. Supported operations: 'sum' (sum of numeric values), 'avg' (average of numeric values), 'max'/'min' (maximum/minimum numeric values), 'count' (count values, adds list length for lists), 'distinct' (list of unique values), 'count_distinct' (number of unique values), 'values_counts' (dictionary of value occurrence counts), 'values_difference' (difference between max and min numeric values). For lists, operations process each element. Multiple aggregations per variable are supported. Example: {'predictions': ['distinct', 'count_distinct', 'avg']}.",
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
        description="Unit for measuring the aggregation interval: 'seconds', 'minutes', 'hours' (time-based, uses wall-clock time elapsed since last output - useful for video streams), or 'runs' (run-based, counts number of workflow executions - useful for video file processing where frame count matters more than time). Time-based intervals track elapsed time between aggregated outputs. Run-based intervals count the number of times the block receives data. The block outputs aggregated results and flushes state when the interval threshold is reached.",
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
        description="Length of the aggregation interval in the units specified by interval_unit. Must be greater than 0. The block accumulates data internally and outputs aggregated results when this interval threshold is reached. For time-based units (seconds, minutes, hours), this is the duration elapsed since the last output. For 'runs', this is the number of workflow executions (e.g., frames processed) since the last output. After outputting results, the block resets its internal state and starts a new aggregation window. Most of the time, the block returns empty outputs while collecting data.",
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
