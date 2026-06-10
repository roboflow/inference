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
    WorkflowsImagesProcessingIndex,
    aggregate_batch_processing_results,
    decode_base64_image,
    deduct_images,
    denote_image_processed,
    dump_image_processing_results,
    dump_images_outputs,
    dump_objects_to_json,
    extract_images_from_result,
    open_progress_log,
    report_failed_files,
)
from inference_cli.lib.workflows.entities import ImageResultsIndexEntry, OutputFileType


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
    aggregation_path = aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.JSONL,
    )

    # then
    decoded_results = []
    with open(aggregation_path, "r") as f:
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
    aggregation_path = aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.JSONL,
    )

    # then
    decoded_results = []
    with open(aggregation_path, "r") as f:
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
    aggregation_path = aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.CSV,
    )

    # then
    df = pd.read_csv(aggregation_path)
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
    aggregation_path = aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.CSV,
    )

    # then
    df = pd.read_csv(aggregation_path)
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
    aggregation_path = aggregate_batch_processing_results(
        output_directory=empty_directory,
        aggregation_format=OutputFileType.CSV,
    )

    # then
    with pytest.raises(EmptyDataError):
        _ = pd.read_csv(aggregation_path)


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
    report_file = report_failed_files(failed_files=[], output_directory=empty_directory)

    # then
    assert report_file is None
    assert len(os.listdir(empty_directory)) == 0


def test_report_failed_files_when_errors_detected(empty_directory: str) -> None:
    # when
    report_file = report_failed_files(
        failed_files=[("some.jpg", "some"), ("other.jpg", "other")],
        output_directory=empty_directory,
    )

    # then
    with open(report_file, "r") as f:
        decoded_file = [
            json.loads(line) for line in f.readlines() if len(line.strip()) > 0
        ]
    assert decoded_file == [
        {"file_path": "some.jpg", "cause": "some"},
        {"file_path": "other.jpg", "cause": "other"},
    ]


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
    results = dump_images_outputs(
        image_results_dir=empty_directory,
        images_in_result=images_in_result,
    )

    # then
    visualization_image_path = os.path.join(empty_directory, "visualization.jpg")
    visualization_image = cv2.imread(visualization_image_path)
    crop_path = os.path.join(empty_directory, "some/crops/1.jpg")
    crop_image = cv2.imread(crop_path)

    assert results == {"visualization": [visualization_image_path], "some": [crop_path]}
    assert visualization_image.shape == (168, 168, 3)
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
        "list": [
            ["a", "b"],
            ["c", np.zeros((168, 168, 3), dtype=np.uint8)],
            [np.zeros((168, 168, 3), dtype=np.uint8)],
        ],
    }

    # when
    result = dump_image_processing_results(
        result=result,
        image_path="/some/directory/my_image.jpeg",
        output_directory=empty_directory,
        save_image_outputs=True,
    )

    # then
    assert result == ImageResultsIndexEntry(
        metadata_output_path=os.path.join(
            empty_directory, "my_image.jpeg", "results.json"
        ),
        image_outputs={
            "other": [os.path.join(empty_directory, "my_image.jpeg", "other.jpg")],
            "dict": [
                os.path.join(empty_directory, "my_image.jpeg", "dict", "c", "0.jpg")
            ],
            "list": [
                os.path.join(empty_directory, "my_image.jpeg", "list", "1", "1.jpg"),
                os.path.join(empty_directory, "my_image.jpeg", "list", "2", "0.jpg"),
            ],
        },
    )
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
        "list": [["a", "b"], ["c", "<deducted_image>"], ["<deducted_image>"]],
    }
    other_image = cv2.imread(
        os.path.join(empty_directory, "my_image.jpeg", "other.jpg")
    )
    assert other_image.shape == (192, 168, 3)
    dict_nested_image = cv2.imread(
        os.path.join(empty_directory, "my_image.jpeg", "dict", "c", "0.jpg")
    )
    assert dict_nested_image.shape == (192, 192, 3)
    list_nested_image_1 = cv2.imread(
        os.path.join(empty_directory, "my_image.jpeg", "list", "1", "1.jpg")
    )
    assert list_nested_image_1.shape == (168, 168, 3)
    list_nested_image_2 = cv2.imread(
        os.path.join(empty_directory, "my_image.jpeg", "list", "2", "0.jpg")
    )
    assert list_nested_image_2.shape == (168, 168, 3)


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


def test_workflows_images_processor_index() -> None:
    # given
    index = WorkflowsImagesProcessingIndex.init()

    # when
    entry_1 = ImageResultsIndexEntry(
        metadata_output_path="/some/image_1.jpg/results.json",
        image_outputs={
            "visualization": ["/some/image_1.jpg/visualization.jpg"],
            "crops": ["/some/image_1.jpg/crops/0.jpg", "/some/image_1.jpg/crops/1.jpg"],
        },
    )
    index.collect_entry(image_path="/inputs/image_1.jpg", entry=entry_1)
    entry_2 = ImageResultsIndexEntry(
        metadata_output_path="/some/image_2.jpg/results.json",
        image_outputs={
            "visualization": ["/some/image_2.jpg/visualization.jpg"],
            "crops": ["/some/image_2.jpg/crops/0.jpg"],
        },
    )
    index.collect_entry(image_path="/inputs/image_2.jpg", entry=entry_2)
    entry_3 = ImageResultsIndexEntry(
        metadata_output_path="/some/image_3.jpg/results.json",
        image_outputs={
            "visualization": ["/some/image_3.jpg/visualization.jpg"],
        },
    )
    index.collect_entry(image_path="/inputs/image_3.jpg", entry=entry_3)

    # then
    metadata = index.export_metadata()
    assert sorted(metadata, key=lambda e: e[0]) == [
        ("/inputs/image_1.jpg", "/some/image_1.jpg/results.json"),
        ("/inputs/image_2.jpg", "/some/image_2.jpg/results.json"),
        ("/inputs/image_3.jpg", "/some/image_3.jpg/results.json"),
    ], "Expected metadata to be indexed correctly"
    assert index.registered_output_images == {
        "visualization",
        "crops",
    }, "Expected to report all images outputs, including ones that were not registered for all images"
    exported_images = index.export_images()
    assert sorted(exported_images["visualization"], key=lambda e: e[0]) == [
        ("/inputs/image_1.jpg", ["/some/image_1.jpg/visualization.jpg"]),
        ("/inputs/image_2.jpg", ["/some/image_2.jpg/visualization.jpg"]),
        ("/inputs/image_3.jpg", ["/some/image_3.jpg/visualization.jpg"]),
    ], "Expected all visualization field outputs to be indexed"
    assert sorted(exported_images["crops"], key=lambda e: e[0]) == [
        (
            "/inputs/image_1.jpg",
            ["/some/image_1.jpg/crops/0.jpg", "/some/image_1.jpg/crops/1.jpg"],
        ),
        ("/inputs/image_2.jpg", ["/some/image_2.jpg/crops/0.jpg"]),
    ], "Expected all crops outputs to be indexed"
