import json
import os.path
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest
from pandas.errors import EmptyDataError

from inference_cli.lib.workflows.common import (
    IMAGES_EXTENSIONS,
    aggregate_batch_processing_results,
    denote_image_processed,
    dump_objects_to_json,
    get_all_images_in_directory,
    open_progress_log,
    report_failed_files,
)
from inference_cli.lib.workflows.entities import OutputFileType


@pytest.mark.parametrize("value", [3, 3.5, "some", True])
def test_dump_objects_to_json_when_primitive_type_given(value: Any) -> None:
    # when
    result = dump_objects_to_json(value=value)

    # then
    assert result == value


def test_dump_objects_to_json_when_list_given() -> None:
    # when
    result = dump_objects_to_json(value=[1, 2, 3])

    # then
    assert json.loads(result) == [1, 2, 3]


def test_dump_objects_to_json_when_set_given() -> None:
    # when
    result = dump_objects_to_json(value={1, 2, 3})

    # then
    assert set(json.loads(result)) == {1, 2, 3}


def test_dump_objects_to_json_when_dict_given() -> None:
    # when
    result = dump_objects_to_json(value={"some": "value", "other": [1, 2, 3]})

    # then
    assert json.loads(result) == {"some": "value", "other": [1, 2, 3]}


def test_aggregate_batch_processing_results_when_json_output_is_expected_and_results_present(
    empty_directory: str,
) -> None:
    # given
    _prepare_dummy_results(root_dir=empty_directory, sub_dir_name="some.jpg")
    _prepare_dummy_results(root_dir=empty_directory, sub_dir_name="other.jpg")

    # when
    file_descriptor, _ = open_progress_log(output_directory=empty_directory)
    denote_image_processed(log_file=file_descriptor, image_path="/my/path/some.jpg")
    denote_image_processed(log_file=file_descriptor, image_path="/my/path/other.jpg")
    file_descriptor.close()
    aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.JSONL,
    )

    # then
    expected_output_path = os.path.join(empty_directory, "aggregated_results.jsonl")
    decoded_results = []
    with open(expected_output_path, "r") as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            decoded_results.append(json.loads(line))
    assert decoded_results == [
        {"some": "value", "other": 3.0, "list": [1, 2, 3], "object": {"nested": "value"}}
    ] * 2


def test_aggregate_batch_processing_results_when_json_output_is_expected_and_results_not_present(
    empty_directory: str,
) -> None:
    # when
    aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.JSONL,
    )

    # then
    expected_output_path = os.path.join(empty_directory, "aggregated_results.jsonl")
    decoded_results = []
    with open(expected_output_path, "r") as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            decoded_results.append(json.loads(line))
    assert decoded_results == []


def test_aggregate_batch_processing_results_when_csv_output_is_expected_and_results_present(
    empty_directory: str,
) -> None:
    # given
    _prepare_dummy_results(root_dir=empty_directory, sub_dir_name="some.jpg")
    _prepare_dummy_results(root_dir=empty_directory, sub_dir_name="other.jpg")

    # when
    file_descriptor, _ = open_progress_log(output_directory=empty_directory)
    denote_image_processed(log_file=file_descriptor, image_path="/my/path/some.jpg")
    denote_image_processed(log_file=file_descriptor, image_path="/my/path/other.jpg")
    file_descriptor.close()
    aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.CSV,
    )

    # then
    expected_output_path = os.path.join(empty_directory, "aggregated_results.csv")
    df = pd.read_csv(expected_output_path)
    assert len(df) == 2, "Expected 2 records"
    assert df.iloc[0].some == "value"
    assert df.iloc[0].other == 3.0
    assert json.loads(df.iloc[0].list) == [1, 2, 3]
    assert json.loads(df.iloc[0].object) == {"nested": "value"}
    assert df.iloc[1].some == "value"
    assert df.iloc[1].other == 3.0
    assert json.loads(df.iloc[1].list) == [1, 2, 3]
    assert json.loads(df.iloc[1].object) == {"nested": "value"}


def test_aggregate_batch_processing_results_when_csv_output_is_expected_and_results_present_but_with_inconsistent_schema(
    empty_directory: str,
) -> None:
    # given
    _prepare_dummy_results(root_dir=empty_directory, sub_dir_name="some.jpg")
    _prepare_dummy_results(root_dir=empty_directory, sub_dir_name="other.jpg", extra_data={"extra": "column"})

    # when
    file_descriptor, _ = open_progress_log(output_directory=empty_directory)
    denote_image_processed(log_file=file_descriptor, image_path="/my/path/some.jpg")
    denote_image_processed(log_file=file_descriptor, image_path="/my/path/other.jpg")
    file_descriptor.close()
    aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.CSV,
    )

    # then
    expected_output_path = os.path.join(empty_directory, "aggregated_results.csv")
    df = pd.read_csv(expected_output_path)
    assert len(df) == 2, "Expected 2 records"
    assert df.iloc[0].some == "value"
    assert df.iloc[0].other == 3.0
    assert json.loads(df.iloc[0].list) == [1, 2, 3]
    assert json.loads(df.iloc[0].object) == {"nested": "value"}
    assert df.iloc[1].some == "value"
    assert df.iloc[1].other == 3.0
    assert json.loads(df.iloc[1].list) == [1, 2, 3]
    assert json.loads(df.iloc[1].object) == {"nested": "value"}
    assert df.iloc[1].extra == "column" or df.iloc[0].extra == "column" and df.iloc[1].extra != df.iloc[0].extra, \
        "Expected one record to have value and other to have none in extra column"


def test_aggregate_batch_processing_results_when_csv_output_is_expected_and_results_not_present(
    empty_directory: str,
) -> None:
    # when
    aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.CSV,
    )

    # then
    expected_output_path = os.path.join(empty_directory, "aggregated_results.csv")
    with pytest.raises(EmptyDataError):
        _ = pd.read_csv(expected_output_path)


def _prepare_dummy_results(
    root_dir: str,
    sub_dir_name: str,
    extra_data: Optional[dict] = None,
) -> None:
    if extra_data is None:
        extra_data = {}
    sub_dir_path = os.path.join(root_dir, sub_dir_name)
    os.makedirs(sub_dir_path, exist_ok=True)
    results = {"some": "value", "other": 3.0, "list": [1, 2, 3], "object": {"nested": "value"}}
    results.update(extra_data)
    results_path = os.path.join(sub_dir_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)


def test_report_failed_files_when_no_errors_detected(empty_directory: str) -> None:
    # when
    report_failed_files(failed_files=[], output_directory=empty_directory)

    # then
    assert len(os.listdir(empty_directory)) == 0


def test_report_failed_files_when_errors_detected(empty_directory: str) -> None:
    # when
    report_failed_files(
        failed_files=[
            ("some.jpg", "some"),
            ("other.jpg", "other")
        ],
        output_directory=empty_directory,
    )

    # then
    dir_content = os.listdir(empty_directory)
    assert len(dir_content) == 1
    file_path = os.path.join(empty_directory, dir_content[0])
    with open(file_path, "r") as f:
        decoded_file = [
            json.loads(line) for line in f.readlines()
            if len(line.strip()) > 0
        ]
    assert decoded_file == [
        {"file_path": "some.jpg", "cause": "some"},
        {"file_path": "other.jpg", "cause": "other"},
    ]


def test_get_all_images_in_directory(empty_directory: str) -> None:
    # given
    for extension in IMAGES_EXTENSIONS:
        _create_empty_file(directory=empty_directory, file_name=f"image.{extension}")
    expected_files = len(IMAGES_EXTENSIONS) // 2  # dividing by two, as we have each extension lower- and upper- case

    # when
    result = get_all_images_in_directory(input_directory=empty_directory)

    # then
    assert len(result) == expected_files


def _create_empty_file(directory: str, file_name: str) -> None:
    file_path = os.path.join(directory, file_name)
    path = Path(file_path)
    path.touch(exist_ok=True)
