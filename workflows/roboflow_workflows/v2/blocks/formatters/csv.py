"""CSV formatter block: one CSV document over a step's invocations.

Port of V1 ``roboflow_core/csv_formatter@v1``::

    row k   = columns_data at invocation k ── columns_operations ──▶ cells
              + timestamp (UTC ISO 8601 from the wall clock)
    batched = some column varies: one call, outputs [None, ..., None, csv]
    single  = every column is constant: one call, output csv

A varying column contributes its value at each invocation; a constant column
(a literal or a workflow parameter) repeats in every row. As in V1, the
document lands on the last invocation and earlier invocations output ``None``,
so a downstream sink runs once. Gated or filtered invocations add no row; a
selected value that is present but ``None`` keeps its row with an empty cell.

Cells keep V1's text: ``None`` is empty, other values use ``str``, and a
string starting with ``= + - @``, a tab or a carriage return gets a leading
``'`` against spreadsheet formula injection. Minimal quoting, ``\\n`` line
ends. Unlike V1, columns keep their ``columns_data`` order (V1's batched order
followed set iteration) and integer columns with empty cells stay integers
(V1's pandas wrote ``1.0``).
"""

import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
)
from roboflow_workflows.v2.blocks.formatters.values import WallClock
from roboflow_workflows.v2.blocks.query import (
    CompiledOperations,
    Operation,
    compile_operations,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

BLOCK_TYPE = "v2/csv_formatter@v1"
TIMESTAMP_COLUMN = "timestamp"
CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

ColumnLiteral = Union[StrictStr, StrictInt, StrictFloat, StrictBool]


class CsvFormatterBlock(Block):
    """Write one CSV row per invocation, plus a timestamp column.

    Example: over three images, ``columns_data={"image": "$inputs.name",
    "count": "$steps.count.output"}`` outputs ``None``, ``None`` and::

        image,count,timestamp
        a.jpg,2,2026-07-01T10:00:00.000001+00:00
        ...

    With no column varying per invocation, the step runs once and makes one
    CSV row, shown on every result row. Add a varying column for one CSV row
    per invocation.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "CSV Formatter",
        "port_of": "roboflow_core/csv_formatter@v1",
        "section": "advanced",
    }
    outputs = {
        "csv_content": Output(
            STRING_KIND,
            description=(
                "The CSV document on the last invocation that ran; None on "
                "every other row, including gated ones. With only constant "
                "columns, one CSV row is shown on every result row."
            ),
        ),
    }

    class Params(BlockParams):
        columns_data: Dict[
            str,
            ColumnLiteral | Ref(batch="if_varying", allow_none=True),  # noqa: F821
        ] = Field(
            description=(
                "Column values by column name, in output order: selectors or "
                "literals. `timestamp` is reserved."
            ),
            examples=[{"image": "$inputs.name", "count": "$steps.count.output"}],
        )
        columns_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description=(
                "UQL operations applied to a column value before writing, by "
                "column name. Every key must be a columns_data key; the "
                "operations name no runtime values."
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

        @field_validator("columns_data", "columns_operations")
        @classmethod
        def _timestamp_is_reserved(cls, value: Dict[str, Any]) -> Dict[str, Any]:
            if TIMESTAMP_COLUMN in value:
                raise ValueError(
                    f"`{TIMESTAMP_COLUMN}` is a reserved column name; the block "
                    f"adds it to every row"
                )

            return value

        # Not a model validator: the engine reruns those per invocation.
        # This literal-only field is validated once per definition, and
        # selectors replace columns_data values, never keys.
        @field_validator("columns_operations")
        @classmethod
        def _operations_compile(
            cls, value: Dict[str, List[Operation]], info: ValidationInfo
        ) -> Dict[str, List[Operation]]:
            if "columns_data" in info.data:
                compile_column_operations(
                    value, column_names=tuple(info.data["columns_data"])
                )

            return value

    def __init__(self, *, wall_clock: WallClock = datetime.now) -> None:
        self._wall_clock = wall_clock
        self._compiled: Optional[Tuple[Any, Dict[str, CompiledOperations]]] = None

    def run(
        self,
        *,
        columns_data: Dict[str, Any],
        columns_operations: Dict[str, List[Any]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Build the CSV document over this call's invocations.

        Args:
            columns_data: Column values; varying columns arrive as ``Batch``.
            columns_operations: Validated operations by column name.

        Returns:
            ``csv_content`` per invocation: a list when a column varies (the
            document last, ``None`` before), else one mapping.

        Raises:
            QueryRuntimeError: When an operation cannot process a value.
        """
        operations = self._operations(columns_data, columns_operations)
        rows = [
            self._row(values, operations=operations)
            for values in unfold_columns(columns_data)
        ]
        document = to_csv(rows, columns=[*columns_data, TIMESTAMP_COLUMN])
        if not any(isinstance(value, Batch) for value in columns_data.values()):
            return {"csv_content": document}

        results = [{"csv_content": None} for _ in rows[:-1]]
        results.append({"csv_content": document})

        return results

    def _operations(
        self, columns_data: Dict[str, Any], columns_operations: Dict[str, List[Any]]
    ) -> Dict[str, CompiledOperations]:
        key = (tuple(columns_data), columns_operations)
        cached = self._compiled
        if cached is None or cached[0] != key:
            compiled = compile_column_operations(
                columns_operations, column_names=tuple(columns_data)
            )
            cached = (key, compiled)
            self._compiled = cached

        return cached[1]

    def _row(
        self, values: Dict[str, Any], *, operations: Dict[str, CompiledOperations]
    ) -> Dict[str, Any]:
        row = dict(values)
        for name, chain in operations.items():
            row[name] = chain.apply(row[name], parameters={})
        row[TIMESTAMP_COLUMN] = self._wall_clock(timezone.utc).isoformat()

        return row


def compile_column_operations(
    columns_operations: Dict[str, List[Any]], *, column_names: Tuple[str, ...]
) -> Dict[str, CompiledOperations]:
    """Compile per-column operation chains that name no runtime values.

    Args:
        columns_operations: Operations by column name.
        column_names: Keys of ``columns_data``.

    Returns:
        Compiled chains by column name.

    Raises:
        ValueError: When a key is not a ``columns_data`` key.
        QueryDefinitionError: On an invalid chain.
    """
    compiled = {}
    for name, operations in columns_operations.items():
        if name not in column_names:
            raise ValueError(
                f"{BLOCK_TYPE}: columns_operations names `{name}`, which is not "
                f"a columns_data column (columns: {list(column_names)})"
            )
        compiled[name] = compile_operations(
            operations,
            path=f"$.columns_operations.{name}",
            declared_names=frozenset(),
        )

    return compiled


def unfold_columns(columns_data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield one row of column values per invocation.

    Args:
        columns_data: Column values; varying columns as equally long ``Batch``
            objects, constant columns as plain values.

    Yields:
        Column values of one invocation, in ``columns_data`` order.
    """
    varying = {
        name: value for name, value in columns_data.items() if isinstance(value, Batch)
    }
    if not varying:
        yield dict(columns_data)
        return

    count = len(next(iter(varying.values())))
    for position in range(count):
        yield {
            name: varying[name][position] if name in varying else value
            for name, value in columns_data.items()
        }


def to_csv(rows: List[Dict[str, Any]], *, columns: List[str]) -> str:
    """Write rows as a CSV document with a header.

    Args:
        rows: Cell values by column name.
        columns: Column order of the header and every row.

    Returns:
        The CSV text with ``\\n`` line ends.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([csv_cell(row[name]) for name in columns])
    document = buffer.getvalue()

    return document


def csv_cell(value: Any) -> str:
    """Return V1's text for one cell.

    Args:
        value: Cell value.

    Returns:
        ``""`` for ``None``, else ``str(value)``; strings starting with a
        formula character get a leading ``'``.
    """
    if value is None:
        return ""

    if isinstance(value, str) and value.startswith(CSV_INJECTION_PREFIXES):
        return "'" + value

    text = str(value)

    return text
