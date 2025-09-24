import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
from supervision.utils.file import read_yaml_file
from tqdm import tqdm

from inference import get_model
from inference.core.models.base import Model
from inference.core.registries.roboflow import get_model_type
from inference_cli.lib.benchmark.results_gathering import ResultsCollector


def run_python_package_speed_benchmark(
    model_id: str,
    images: List[np.ndarray],
    results_collector: ResultsCollector,
    warm_up_inferences: int = 10,
    benchmark_inferences: int = 1000,
    batch_size: int = 1,
    api_key: Optional[str] = None,
    model_configuration: Optional[str] = None,
) -> None:
    inference_configuration = {}
    if model_configuration is not None:
        inference_configuration = read_yaml_file(file_path=model_configuration)
    print(
        f"Inference will be executed with the following parameters: {inference_configuration}"
    )
    model_type = get_model_type(model_id, api_key=api_key)
    model = get_model(model_id=model_id, api_key=api_key)
    model_batch_size = getattr(model, "batch_size", None)
    input_height = getattr(model, "img_size_h", None)
    input_width = getattr(model, "img_size_w", None)
    print(
        f"Model details | task_type={model_type[0]} | model_type={model_type[1]} | "
        f"batch_size={model_batch_size} | input_height={input_height} | input_width={input_width}"
    )
    run_model_warm_up(
        model=model,
        inference_configuration=inference_configuration,
        image=images[0],
        warm_up_inferences=warm_up_inferences,
    )
    run_benchmark(
        model=model,
        inference_configuration=inference_configuration,
        images=images,
        results_collector=results_collector,
        benchmark_inferences=benchmark_inferences,
        batch_size=batch_size,
    )


def run_model_warm_up(
    model: Model,
    inference_configuration: Dict[str, Any],
    image: np.ndarray,
    warm_up_inferences: int,
) -> None:
    for _ in tqdm(
        range(warm_up_inferences), desc="Warming up model...", total=warm_up_inferences
    ):
        _ = model.infer(image, **inference_configuration)


def run_benchmark(
    model: Model,
    inference_configuration: Dict[str, Any],
    images: List[np.ndarray],
    results_collector: ResultsCollector,
    benchmark_inferences: int,
    batch_size: int,
) -> None:
    while len(images) < batch_size:
        images = images + images
    results_collector.start_benchmark()
    try:
        for _ in range(benchmark_inferences):
            random.shuffle(images)
            payload = images[:batch_size]
            start = time.time()
            _ = model.infer(payload, **inference_configuration)
            duration = time.time() - start
            results_collector.register_inference_duration(
                batch_size=batch_size, duration=duration
            )
    finally:
        results_collector.stop_benchmark()
