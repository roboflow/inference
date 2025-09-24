import pandas as pd
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
)
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    DetectionsPropertyExtract,
    SequenceLength,
    StringToUpperCase,
)
from inference.core.workflows.core_steps.formatters.csv.v1 import (
    BlockManifest,
    prepare_csv_content,
    unfold_parameters,
)
from inference.core.workflows.execution_engine.entities.base import Batch


def test_manifest_parsing() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/csv_formatter@v1",
        "name": "csv_formatter",
        "columns_data": {
            "predicted_classes": "$steps.model.predictions",
            "number_of_bounding_boxes": "$steps.model.predictions",
            "additional_column": "$inputs.additional_column_value",
        },
        "columns_operations": {
            "predicted_classes": [
                {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
            ],
            "number_of_bounding_boxes": [{"type": "SequenceLength"}],
        },
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    assert result == BlockManifest(
        type="roboflow_core/csv_formatter@v1",
        name="csv_formatter",
        columns_data={
            "predicted_classes": "$steps.model.predictions",
            "number_of_bounding_boxes": "$steps.model.predictions",
            "additional_column": "$inputs.additional_column_value",
        },
        columns_operations={
            "predicted_classes": [
                DetectionsPropertyExtract(
                    type="DetectionsPropertyExtract",
                    property_name=DetectionsProperty.CLASS_NAME,
                )
            ],
            "number_of_bounding_boxes": [SequenceLength(type="SequenceLength")],
        },
    )


def test_manifest_parsing_when_timestamp_column_requested() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/csv_formatter@v1",
        "name": "csv_formatter",
        "columns_data": {
            "predicted_classes": "$steps.model.predictions",
            "number_of_bounding_boxes": "$steps.model.predictions",
            "additional_column": "$inputs.additional_column_value",
            "timestamp": "$inputs.some",
        },
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_unfold_parameters_when_no_batch_oriented_parameters_given() -> None:
    # given
    batch_columns_data = {"some": "value", "other": 3}

    # when
    result = list(unfold_parameters(batch_columns_data=batch_columns_data))

    # then
    assert result == [{"some": "value", "other": 3}]


def test_unfold_parameters_when_batch_oriented_parameters_given() -> None:
    # given
    batch_columns_data = {
        "some": "value",
        "other": 3,
        "batch_1": Batch(indices=[(0,), (1,), (2,)], content=[1, 2, 3]),
        "batch_2": Batch(indices=[(0,)], content=[True]),
        "batch_3": Batch(indices=[(0,), (1,), (2,)], content=["a", "b", "c"]),
    }

    # when
    result = list(unfold_parameters(batch_columns_data=batch_columns_data))

    # then
    assert result == [
        {"some": "value", "other": 3, "batch_1": 1, "batch_2": True, "batch_3": "a"},
        {"some": "value", "other": 3, "batch_1": 2, "batch_2": True, "batch_3": "b"},
        {"some": "value", "other": 3, "batch_1": 3, "batch_2": True, "batch_3": "c"},
    ]


def test_prepare_csv_rows() -> None:
    # given
    batch_columns_data = {
        "some": "value",
        "other": 3,
        "batch_1": Batch(indices=[(0,), (1,), (2,)], content=[1, 2, 3]),
        "batch_2": Batch(indices=[(0,)], content=[True]),
        "batch_3": Batch(indices=[(0,), (1,), (2,)], content=["a", "b", "c"]),
    }
    columns_operations = {"batch_3": [StringToUpperCase(type="StringToUpperCase")]}

    # when
    result = prepare_csv_content(
        batch_columns_data=batch_columns_data,
        columns_operations=columns_operations,
    )
    result_df = pd.DataFrame(result)

    # then
    assert set(result_df.columns) == {
        "some",
        "other",
        "batch_1",
        "batch_2",
        "batch_3",
        "timestamp",
    }, "Expected all explicitly defined columns + timestamp one"
    assert result_df["some"].tolist() == [
        "value",
        "value",
        "value",
    ], "Non batch data expected to be broadcast"
    assert result_df["other"].tolist() == [
        3,
        3,
        3,
    ], "Non batch data expected to be broadcast"
    assert result_df["batch_1"].tolist() == [1, 2, 3]
    assert result_df["batch_2"].tolist() == [
        True,
        True,
        True,
    ], "Single element batch expected to be broadcast"
    assert result_df["batch_3"].tolist() == [
        "A",
        "B",
        "C",
    ], "Expected operation to be applied for each letter in each row"
