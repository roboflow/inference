import base64
import json
import os.path
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import pytest
from pandas.errors import EmptyDataError

from inference_cli.lib.workflows.common import (
    IMAGES_EXTENSIONS,
    aggregate_batch_processing_results,
    decode_base64_image,
    deduct_images,
    denote_image_processed,
    dump_image_processing_results,
    dump_images_outputs,
    dump_objects_to_json,
    extract_images_from_result,
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
    print(decoded_results)
    assert decoded_results == [
        {
            "some": "value",
            "image": "other.jpg",
            "other": 3.0,
            "list_field": [1, 2, 3],
            "object_field": {"nested": "value"},
        },
        {
            "some": "value",
            "image": "some.jpg",
            "other": 3.0,
            "list_field": [1, 2, 3],
            "object_field": {"nested": "value"},
        },
    ]


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
    assert json.loads(df.iloc[0].list_field) == [1, 2, 3]
    assert json.loads(df.iloc[0].object_field) == {"nested": "value"}
    assert df.iloc[1].some == "value"
    assert df.iloc[1].other == 3.0
    assert json.loads(df.iloc[1].list_field) == [1, 2, 3]
    assert json.loads(df.iloc[1].object_field) == {"nested": "value"}


def test_aggregate_batch_processing_results_when_csv_output_is_expected_and_results_present_but_with_inconsistent_schema(
    empty_directory: str,
) -> None:
    # given
    _prepare_dummy_results(root_dir=empty_directory, sub_dir_name="some.jpg")
    _prepare_dummy_results(
        root_dir=empty_directory,
        sub_dir_name="other.jpg",
        extra_data={"extra": "column"},
    )

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
    assert json.loads(df.iloc[0].list_field) == [1, 2, 3]
    assert json.loads(df.iloc[0].object_field) == {"nested": "value"}
    assert df.iloc[1].some == "value"
    assert df.iloc[1].other == 3.0
    assert json.loads(df.iloc[1].list_field) == [1, 2, 3]
    assert json.loads(df.iloc[1].object_field) == {"nested": "value"}
    assert (
        df.iloc[1].extra == "column"
        or df.iloc[0].extra == "column"
        and df.iloc[1].extra != df.iloc[0].extra
    ), "Expected one record to have value and other to have none in extra column"


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
    results = {
        "some": "value",
        "other": 3.0,
        "list_field": [1, 2, 3],
        "object_field": {"nested": "value"},
    }
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
        failed_files=[("some.jpg", "some"), ("other.jpg", "other")],
        output_directory=empty_directory,
    )

    # then
    dir_content = os.listdir(empty_directory)
    assert len(dir_content) == 1
    file_path = os.path.join(empty_directory, dir_content[0])
    with open(file_path, "r") as f:
        decoded_file = [
            json.loads(line) for line in f.readlines() if len(line.strip()) > 0
        ]
    assert decoded_file == [
        {"file_path": "some.jpg", "cause": "some"},
        {"file_path": "other.jpg", "cause": "other"},
    ]


def test_get_all_images_in_directory(empty_directory: str) -> None:
    # given
    for extension in IMAGES_EXTENSIONS:
        _create_empty_file(directory=empty_directory, file_name=f"image.{extension}")
    _create_empty_file(directory=empty_directory, file_name=f".tmp")
    _create_empty_file(directory=empty_directory, file_name=f".bin")
    expected_files = len(os.listdir(empty_directory)) - 2

    # when
    result = get_all_images_in_directory(input_directory=empty_directory)

    # then
    assert len(result) == expected_files


def _create_empty_file(directory: str, file_name: str) -> None:
    file_path = os.path.join(directory, file_name)
    path = Path(file_path)
    path.touch(exist_ok=True)


def test_decode_base64_image_when_base64_header_present() -> None:
    # given
    image = np.zeros((192, 168, 3), dtype=np.uint8)
    encoded = _encode_image_to_base64(image=image)
    encoded = f"data:image/jpeg;base64,{encoded}"

    # when
    result = decode_base64_image(payload=encoded)

    # then
    assert np.allclose(result, image)


def test_decode_base64_image_when_base64_header_not_present() -> None:
    # given
    image = np.zeros((192, 168, 3), dtype=np.uint8)
    encoded = _encode_image_to_base64(image=image)

    # when
    result = decode_base64_image(payload=encoded)

    # then
    assert np.allclose(result, image)


def test_extract_images_from_result() -> None:
    # given
    result = {
        "some": "value",
        "other": {
            "type": "base64",
            "value": _encode_image_to_base64(np.zeros((192, 168, 3), dtype=np.uint8)),
        },
        "dict": {
            "a": 1,
            "b": 2,
            "c": [np.zeros((192, 192, 3), dtype=np.uint8), 1, "some"],
        },
        "list": [["a", "b"], ["c", np.zeros((168, 168, 3), dtype=np.uint8)]],
    }

    # when
    result = extract_images_from_result(result=result)

    # then
    assert len(result) == 3, "Expected three images returned"
    key_to_image = {key: image for key, image in result}
    assert key_to_image["other"].shape == (192, 168, 3)
    assert key_to_image["dict/c/0"].shape == (192, 192, 3)
    assert key_to_image["list/1/1"].shape == (168, 168, 3)


def test_deduct_images() -> None:
    # given
    result = {
        "some": "value",
        "other": {
            "type": "base64",
            "value": _encode_image_to_base64(np.zeros((192, 168, 3), dtype=np.uint8)),
        },
        "dict": {
            "a": 1,
            "b": 2,
            "c": [np.zeros((192, 192, 3), dtype=np.uint8), 1, "some"],
        },
        "list": [["a", "b"], ["c", np.zeros((168, 168, 3), dtype=np.uint8)]],
    }

    # when
    result = deduct_images(result=result)

    # then
    assert result == {
        "some": "value",
        "other": "<deducted_image>",
        "dict": {"a": 1, "b": 2, "c": ["<deducted_image>", 1, "some"]},
        "list": [["a", "b"], ["c", "<deducted_image>"]],
    }


def test_dump_images_outputs(empty_directory: str) -> None:
    # given
    images_in_result = [
        ("visualization", np.zeros((168, 168, 3), dtype=np.uint8)),
        ("some/crops/1", np.zeros((192, 192, 3), dtype=np.uint8)),
    ]

    # when
    dump_images_outputs(
        image_results_dir=empty_directory,
        images_in_result=images_in_result,
    )

    # then
    visualization_image = cv2.imread(os.path.join(empty_directory, "visualization.jpg"))
    assert visualization_image.shape == (168, 168, 3)
    crop_image = cv2.imread(os.path.join(empty_directory, "some/crops/1.jpg"))
    assert crop_image.shape == (192, 192, 3)


def test_dump_image_processing_results_when_images_are_to_be_saved(
    empty_directory: str,
) -> None:
    # given
    result = {
        "some": "value",
        "other": {
            "type": "base64",
            "value": _encode_image_to_base64(np.zeros((192, 168, 3), dtype=np.uint8)),
        },
        "dict": {
            "a": 1,
            "b": 2,
            "c": [np.zeros((192, 192, 3), dtype=np.uint8), 1, "some"],
        },
        "list": [["a", "b"], ["c", np.zeros((168, 168, 3), dtype=np.uint8)]],
    }

    # when
    dump_image_processing_results(
        result=result,
        image_path="/some/directory/my_image.jpeg",
        output_directory=empty_directory,
        save_image_outputs=True,
    )

    # then
    assert os.path.isdir(os.path.join(empty_directory, "my_image.jpeg"))
    structured_results_path = os.path.join(
        empty_directory, "my_image.jpeg", "results.json"
    )
    with open(structured_results_path) as f:
        structured_results = json.load(f)
    assert structured_results == {
        "some": "value",
        "other": "<deducted_image>",
        "dict": {"a": 1, "b": 2, "c": ["<deducted_image>", 1, "some"]},
        "list": [["a", "b"], ["c", "<deducted_image>"]],
    }
    other_image = cv2.imread(
        os.path.join(empty_directory, "my_image.jpeg", "other.jpg")
    )
    assert other_image.shape == (192, 168, 3)
    dict_nested_image = cv2.imread(
        os.path.join(empty_directory, "my_image.jpeg", "dict", "c", "0.jpg")
    )
    assert dict_nested_image.shape == (192, 192, 3)
    list_nested_image = cv2.imread(
        os.path.join(empty_directory, "my_image.jpeg", "list", "1", "1.jpg")
    )
    assert list_nested_image.shape == (168, 168, 3)


def test_dump_image_processing_results_when_images_not_to_be_saved(
    empty_directory: str,
) -> None:
    # given
    result = {
        "some": "value",
        "other": {
            "type": "base64",
            "value": _encode_image_to_base64(np.zeros((192, 168, 3), dtype=np.uint8)),
        },
        "dict": {
            "a": 1,
            "b": 2,
            "c": [np.zeros((192, 192, 3), dtype=np.uint8), 1, "some"],
        },
        "list": [["a", "b"], ["c", np.zeros((168, 168, 3), dtype=np.uint8)]],
    }

    # when
    dump_image_processing_results(
        result=result,
        image_path="/some/directory/my_image.jpeg",
        output_directory=empty_directory,
        save_image_outputs=False,
    )

    # then
    assert os.path.isdir(os.path.join(empty_directory, "my_image.jpeg"))
    structured_results_path = os.path.join(
        empty_directory, "my_image.jpeg", "results.json"
    )
    with open(structured_results_path) as f:
        structured_results = json.load(f)
    assert structured_results == {
        "some": "value",
        "other": "<deducted_image>",
        "dict": {"a": 1, "b": 2, "c": ["<deducted_image>", 1, "some"]},
        "list": [["a", "b"], ["c", "<deducted_image>"]],
    }
    assert not os.path.exists(
        os.path.join(empty_directory, "my_image.jpeg", "other.jpg")
    )
    assert not os.path.exists(
        os.path.join(empty_directory, "my_image.jpeg", "dict", "c", "0.jpg")
    )
    assert not os.path.exists(
        os.path.join(empty_directory, "my_image.jpeg", "list", "1", "1.jpg")
    )


def _encode_image_to_base64(image: np.ndarray) -> str:
    _, img_encoded = cv2.imencode(".jpg", image)
    image_bytes = np.array(img_encoded).tobytes()
    return base64.b64encode(image_bytes).decode("utf-8")
