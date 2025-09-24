import json
import os
import subprocess
from copy import deepcopy

import cv2
import pandas as pd
import pytest
import supervision as sv

from tests.inference_cli.integration_tests.conftest import (
    INFERENCE_CLI_TESTS_API_KEY,
    RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED,
    RUN_TESTS_WITH_INFERENCE_PACKAGE,
)


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.timeout(180)
def test_processing_image_with_hosted_api(
    image_to_be_processed: str,
    empty_directory: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-image "
        f"--image_path {image_to_be_processed} "
        f"--output_dir {empty_directory} "
        f"--processing_target api "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640 "
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    # when
    result = subprocess.run(command, env=new_process_env)

    # then
    assert result.returncode == 0
    result_dir = os.path.join(empty_directory, os.path.basename(image_to_be_processed))
    results_json = os.path.join(result_dir, "results.json")
    result_image = os.path.join(result_dir, "bounding_box_visualization.jpg")
    with open(results_json) as f:
        decoded_json = json.load(f)
    assert set(decoded_json.keys()) == {
        "model_predictions",
        "bounding_box_visualization",
    }
    assert decoded_json["bounding_box_visualization"] == "<deducted_image>"
    assert cv2.imread(result_image) is not None


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.timeout(180)
def test_processing_images_directory_with_hosted_api(
    dataset_directory: str,
    empty_directory: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-images-directory "
        f"--input_directory {dataset_directory} "
        f"--output_dir {empty_directory} "
        f"--processing_target api "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640 "
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    # when
    result = subprocess.run(command, env=new_process_env)

    # then
    assert result.returncode == 0
    assert (
        len(os.listdir(empty_directory)) == 5
    ), "Expected 3 images dirs, log file and aggregated results"
    for i in range(3):
        image_results_dir = os.path.join(empty_directory, f"{i}.jpg")
        image_results_dir_content = set(os.listdir(image_results_dir))
        assert image_results_dir_content == {
            "results.json",
            "bounding_box_visualization.jpg",
        }
    result_csv = pd.read_csv(os.path.join(empty_directory, "aggregated_results.csv"))
    assert len(result_csv) == 3
    assert (
        len(result_csv.columns) == 3
    ), "3 columns expected - predictions and deducted visualization"


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.skipif(
    not RUN_TESTS_WITH_INFERENCE_PACKAGE,
    reason="`RUN_TESTS_WITH_INFERENCE_PACKAGE` set to False",
)
@pytest.mark.timeout(180)
def test_processing_image_with_inference_package(
    image_to_be_processed: str,
    empty_directory: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-image "
        f"--image_path {image_to_be_processed} "
        f"--output_dir {empty_directory} "
        f"--processing_target inference_package "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640"
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    # when
    result = subprocess.run(command, env=new_process_env)

    # then
    assert result.returncode == 0
    result_dir = os.path.join(empty_directory, os.path.basename(image_to_be_processed))
    results_json = os.path.join(result_dir, "results.json")
    result_image = os.path.join(result_dir, "bounding_box_visualization.jpg")
    with open(results_json) as f:
        decoded_json = json.load(f)
    assert set(decoded_json.keys()) == {
        "model_predictions",
        "bounding_box_visualization",
    }
    assert decoded_json["bounding_box_visualization"] == "<deducted_image>"
    assert cv2.imread(result_image) is not None


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.skipif(
    not RUN_TESTS_WITH_INFERENCE_PACKAGE,
    reason="`RUN_TESTS_WITH_INFERENCE_PACKAGE` set to False",
)
@pytest.mark.timeout(180)
def test_processing_image_with_inference_package_when_output_images_should_not_be_preserved(
    image_to_be_processed: str,
    empty_directory: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-image "
        f"--image_path {image_to_be_processed} "
        f"--output_dir {empty_directory} "
        f"--processing_target inference_package "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640 "
        f"--no_save_image_outputs"
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    # when
    result = subprocess.run(command, env=new_process_env)

    # then
    assert result.returncode == 0
    result_dir = os.path.join(empty_directory, os.path.basename(image_to_be_processed))
    results_json = os.path.join(result_dir, "results.json")
    assert not os.path.exists(
        os.path.join(result_dir, "bounding_box_visualization.jpg")
    )
    with open(results_json) as f:
        decoded_json = json.load(f)
    assert set(decoded_json.keys()) == {
        "model_predictions",
        "bounding_box_visualization",
    }
    assert decoded_json["bounding_box_visualization"] == "<deducted_image>"


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.skipif(
    not RUN_TESTS_WITH_INFERENCE_PACKAGE,
    reason="`RUN_TESTS_WITH_INFERENCE_PACKAGE` set to False",
)
@pytest.mark.timeout(180)
def test_processing_images_directory_with_inference_package(
    dataset_directory: str,
    empty_directory: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-images-directory "
        f"--input_directory {dataset_directory} "
        f"--output_dir {empty_directory} "
        f"--processing_target inference_package "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640 "
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    # when
    result = subprocess.run(command, env=new_process_env)

    # then
    assert result.returncode == 0
    assert (
        len(os.listdir(empty_directory)) == 5
    ), "Expected 3 images dirs, log file and aggregated results"
    for i in range(3):
        image_results_dir = os.path.join(empty_directory, f"{i}.jpg")
        image_results_dir_content = set(os.listdir(image_results_dir))
        assert image_results_dir_content == {
            "results.json",
            "bounding_box_visualization.jpg",
        }
    result_csv = pd.read_csv(os.path.join(empty_directory, "aggregated_results.csv"))
    assert len(result_csv) == 3
    assert (
        len(result_csv.columns) == 3
    ), "3 columns expected - predictions and deducted visualization"


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.skipif(
    not RUN_TESTS_WITH_INFERENCE_PACKAGE,
    reason="`RUN_TESTS_WITH_INFERENCE_PACKAGE` set to False",
)
@pytest.mark.timeout(180)
def test_processing_video_with_inference_package_with_modulated_fps(
    video_to_be_processed: str,
    empty_directory: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-video "
        f"--video_path {video_to_be_processed} "
        f"--output_dir {empty_directory} "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640 "
        f"--max_fps 1.0"
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    result = subprocess.run(command, env=new_process_env, stdout=subprocess.PIPE)

    # then
    assert result.returncode == 0
    result_csv = pd.read_csv(
        os.path.join(empty_directory, "workflow_results_source_0.csv")
    )
    assert len(result_csv) == 14
    assert len(result_csv.columns) == 2
    output_video_info = sv.VideoInfo.from_video_path(
        video_path=os.path.join(
            empty_directory, "source_0_output_bounding_box_visualization_preview.mp4"
        )
    )
    assert output_video_info.total_frames == 14


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.skipif(
    not RUN_TESTS_WITH_INFERENCE_PACKAGE,
    reason="`RUN_TESTS_WITH_INFERENCE_PACKAGE` set to False",
)
@pytest.mark.timeout(180)
def test_processing_video_with_inference_package_with_modulated_fps_when_video_should_not_be_preserved(
    video_to_be_processed: str,
    empty_directory: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-video "
        f"--video_path {video_to_be_processed} "
        f"--output_dir {empty_directory} "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640 "
        f"--max_fps 1.0 "
        f"--no_save_out_video"
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    result = subprocess.run(command, env=new_process_env, stdout=subprocess.PIPE)

    # then
    assert result.returncode == 0
    result_csv = pd.read_csv(
        os.path.join(empty_directory, "workflow_results_source_0.csv")
    )
    assert len(result_csv) == 14
    assert len(result_csv.columns) == 2
    assert not os.path.exists(
        os.path.join(
            empty_directory, "source_0_output_bounding_box_visualization_preview.mp4"
        )
    )


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.skipif(
    not RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED,
    reason="`RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED` set to False",
)
@pytest.mark.timeout(180)
def test_processing_image_with_inference_package_when_inference_not_installed(
    empty_directory: str,
    image_to_be_processed: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-image "
        f"--image_path {image_to_be_processed} "
        f"--output_dir {empty_directory} "
        f"--processing_target inference_package "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640"
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    # when
    result = subprocess.run(command, env=new_process_env, stdout=subprocess.PIPE)

    # then
    assert result.returncode != 0
    assert (
        "You need to install `inference` package to use this feature"
        in result.stdout.decode("utf-8")
    )


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.skipif(
    not RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED,
    reason="`RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED` set to False",
)
@pytest.mark.timeout(180)
def test_processing_images_directory_with_inference_package_when_inference_not_installed(
    empty_directory: str,
    dataset_directory: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-images-directory "
        f"--input_directory {dataset_directory} "
        f"--output_dir {empty_directory} "
        f"--processing_target inference_package "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640 "
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    # when
    result = subprocess.run(command, env=new_process_env, stdout=subprocess.PIPE)

    # then
    assert result.returncode != 0
    assert (
        "You need to install `inference` package to use this feature"
        in result.stdout.decode("utf-8")
    )


@pytest.mark.skipif(
    INFERENCE_CLI_TESTS_API_KEY is None,
    reason="`INFERENCE_CLI_TESTS_API_KEY` not provided.",
)
@pytest.mark.skipif(
    not RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED,
    reason="`RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED` set to False",
)
@pytest.mark.timeout(180)
def test_processing_video_with_inference_package_when_inference_not_installed(
    empty_directory: str,
    video_to_be_processed: str,
) -> None:
    # given
    command = (
        f"python -m inference_cli.main workflows process-video "
        f"--video_path {video_to_be_processed} "
        f"--output_dir {empty_directory} "
        f"--workspace_name paul-guerrie-tang1 "
        f"--workflow_id prod-test-workflow "
        f"--api-key {INFERENCE_CLI_TESTS_API_KEY} "
        f"--model_id yolov8n-640"
    ).split()
    new_process_env = deepcopy(os.environ)
    new_process_env["ALLOW_INTERACTIVE_INFERENCE_INSTALLATION"] = "False"

    result = subprocess.run(command, env=new_process_env, stdout=subprocess.PIPE)

    # then
    assert result.returncode != 0
    assert (
        "You need to install `inference` package to use this feature"
        in result.stdout.decode("utf-8")
    )
