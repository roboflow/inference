import numpy as np
import pytest

from inference import get_model
from inference_cli.lib.benchmark.dataset import load_dataset_images


@pytest.fixture
def dataset_reference() -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    dataset_images = load_dataset_images(
        dataset_reference="coco",
    )
    return dataset_images, {i.shape[:2] for i in dataset_images}


# args of inference benchmark python-package-speed -m yolov8n-seg-640 -bi 10000 command
args = {
    "dataset_reference": "coco",
    "warm_up_inferences": 10,
    "benchmark_inferences": 10000,
    "batch_size": 1,
    "api_key": None,
    "model_configuration": None,
    "output_location": None,
}


def test_benchmark_equivalent_rfdetr(benchmark, dataset_reference):
    images, image_sizes = dataset_reference

    model = get_model(model_id="rfdetr-base", api_key=None)

    benchmark(model.infer, images)


def test_benchmark_equivalent_yolov8n_seg(benchmark, dataset_reference):
    images, image_sizes = dataset_reference

    model = get_model(model_id="yolov8n-seg-640", api_key=None)

    benchmark(model.infer, images)


def test_benchmark_equivalent_yolov8n(benchmark, dataset_reference):
    images, image_sizes = dataset_reference

    model = get_model(model_id="yolov8n-640", api_key=None)

    benchmark(model.infer, images)
