"""``v2/csv_formatter@v1`` on plain values and engine-shaped batches."""

import csv
import io
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.formatters.csv import CsvFormatterBlock
from roboflow_workflows.v2.data.batch import Batch

START = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
LENGTH = [{"type": "SequenceLength"}]


class StepClock:
    """Return START, START + 1 s, ...; record the requested timezones."""

    def __init__(self) -> None:
        self.zones = []

    def __call__(self, tz):
        self.zones.append(tz)
        moment = START + timedelta(seconds=len(self.zones) - 1)

        return moment.astimezone(tz)


def _run(columns_data: dict, columns_operations=None, *, clock=None):
    selectors = {name: f"$inputs.{name}" for name in columns_data}
    params = CsvFormatterBlock.Params(
        columns_data=selectors, columns_operations=columns_operations or {}
    )
    block = CsvFormatterBlock(wall_clock=clock or StepClock())
    result = block.run(
        columns_data=columns_data, columns_operations=params.columns_operations
    )

    return result


def _parse(document: str) -> list:
    return list(csv.reader(io.StringIO(document)))


def test_constant_columns_give_one_document_in_declaration_order() -> None:
    clock = StepClock()

    result = _run({"site": "gate", "count": 3}, clock=clock)

    assert result == {
        "csv_content": "site,count,timestamp\ngate,3,2026-07-01T10:00:00+00:00\n"
    }
    assert clock.zones == [timezone.utc]


def test_a_varying_column_gives_one_row_per_invocation_on_the_last() -> None:
    result = _run(
        {"name": Batch.of(["a", "b", "c"]), "site": "gate", "n": Batch.of([1, 2, 3])}
    )

    assert [entry["csv_content"] for entry in result[:2]] == [None, None]
    assert _parse(result[2]["csv_content"]) == [
        ["name", "site", "n", "timestamp"],
        ["a", "gate", "1", "2026-07-01T10:00:00+00:00"],
        ["b", "gate", "2", "2026-07-01T10:00:01+00:00"],
        ["c", "gate", "3", "2026-07-01T10:00:02+00:00"],
    ]


def test_a_single_invocation_batch_still_returns_a_list() -> None:
    result = _run({"name": Batch.of(["only"])})

    assert len(result) == 1
    assert result[0]["csv_content"].startswith("name,timestamp\nonly,")


def test_commas_quotes_and_newlines_round_trip() -> None:
    value = 'left, "quoted"\nnext line'

    document = _run({"text": value})["csv_content"]

    assert document.startswith('text,timestamp\n"left, ""quoted""\nnext line",')
    assert _parse(document)[1][0] == value


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r"])
def test_formula_prefixes_are_neutralized_in_strings(prefix: str) -> None:
    document = _run({"text": f"{prefix}SUM(A1)"})["csv_content"]

    assert _parse(document)[1][0] == f"'{prefix}SUM(A1)"


def test_numbers_none_and_containers_keep_v1_text() -> None:
    result = _run(
        {
            "n": Batch.of([-5, 7]),
            "missing": Batch.of([None, 1]),
            "flag": True,
            "score": 0.1,
            "labels": Batch.of([["car", "dog"], []]),
        }
    )

    rows = _parse(result[-1]["csv_content"])

    assert [row[:5] for row in rows[1:]] == [
        ["-5", "", "True", "0.1", "['car', 'dog']"],
        ["7", "1", "True", "0.1", "[]"],
    ]


def test_column_operations_run_per_row() -> None:
    result = _run(
        {"labels": Batch.of([["car", "dog"], []])},
        columns_operations={"labels": LENGTH},
    )

    assert [row[0] for row in _parse(result[-1]["csv_content"])] == ["labels", "2", "0"]


@pytest.mark.parametrize("field", ["columns_data", "columns_operations"])
def test_timestamp_is_a_reserved_column(field: str) -> None:
    params = {"columns_data": {"a": 1}, field: {"timestamp": LENGTH}}
    if field == "columns_data":
        params[field] = {"timestamp": 1}

    with pytest.raises(ValidationError, match="reserved column name"):
        CsvFormatterBlock.Params(**params)


def test_operations_for_an_unknown_column_fail_validation() -> None:
    with pytest.raises(ValidationError, match="columns_operations names `count`"):
        CsvFormatterBlock.Params(
            columns_data={"a": 1}, columns_operations={"count": LENGTH}
        )


@pytest.mark.parametrize(
    "header,value,encoded_header,encoded_value",
    [
        ("plain", "plain", "plain", "plain"),
        ("head\rpart", "cell\rpart", '"head\rpart"', '"cell\rpart"'),
        ("head\npart", "cell\npart", '"head\npart"', '"cell\npart"'),
        ("head\r\npart", "cell\r\npart", '"head\r\npart"', '"cell\r\npart"'),
        ("head,part", 'cell,"part"', '"head,part"', '"cell,""part"""'),
        ("head\r", "cell\r", '"head\r"', '"cell\r"'),
        ("\rhead", "\rSUM(A1)", '"\rhead"', '"\'\rSUM(A1)"'),
        ("=header", "=SUM(A1)", "=header", "'=SUM(A1)"),
    ],
    ids=[
        "plain",
        "cr",
        "lf",
        "crlf",
        "comma-quotes",
        "trailing-cr",
        "leading-cr",
        "formula",
    ],
)
def test_record_delimiters_preserve_header_and_cell_text(
    header, value, encoded_header, encoded_value
) -> None:
    document = _run({header: value})["csv_content"]

    assert document == (
        f"{encoded_header},timestamp\n" f"{encoded_value},2026-07-01T10:00:00+00:00\n"
    )
    expected_value = "'" + value if value.startswith(("=", "\r")) else value
    assert list(csv.reader(io.StringIO(document, newline=""))) == [
        [header, "timestamp"],
        [expected_value, "2026-07-01T10:00:00+00:00"],
    ]
