import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from supervision.utils.file import read_yaml_file
from tqdm import tqdm

from inference_cli.lib.benchmark.results_gathering import ResultsCollector
from inference_models import AutoModel
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.auto_loaders.entities import AnyModel


def run_inference_experimental_benchmark(
    model_id: str,
    images: List[np.ndarray],
    results_collector: ResultsCollector,
    warm_up_inferences: int = 10,
    benchmark_inferences: int = 1000,
    batch_size: int = 1,
    api_key: Optional[str] = None,
    model_configuration: Optional[str] = None,
    model_package_id: Optional[str] = None,
    turn_images_to_tensors: bool = True,
    allow_untrusted_packages: bool = True,
) -> None:
    inference_configuration = {}
    if model_configuration is not None:
        inference_configuration = read_yaml_file(file_path=model_configuration)
    print(
        f"Inference will be executed with the following parameters: {inference_configuration}"
    )
    AutoModel.describe_model(model_id=model_id, api_key=api_key)
    if model_package_id:
        AutoModel.describe_model_package(
            model_id=model_id, package_id=model_package_id, api_key=api_key
        )
    if turn_images_to_tensors:
        images = [
            torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1]))
            .permute(2, 0, 1)
            .to(DEFAULT_DEVICE)
            for image in images
        ]
    model = AutoModel.from_pretrained(
        model_id,
        api_key=api_key,
        model_package_id=model_package_id,
        allow_untrusted_packages=allow_untrusted_packages,
        device=DEFAULT_DEVICE,
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
    model: AnyModel,
    inference_configuration: Dict[str, Any],
    image: Union[np.ndarray, torch.Tensor],
    warm_up_inferences: int,
) -> None:
    for _ in tqdm(
        range(warm_up_inferences), desc="Warming up model...", total=warm_up_inferences
    ):
        _ = model(image, **inference_configuration)


def run_benchmark(
    model: AnyModel,
    inference_configuration: Dict[str, Any],
    images: List[Union[np.ndarray, torch.Tensor]],
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
            _ = model(payload, **inference_configuration)
            duration = time.time() - start
            results_collector.register_inference_duration(
                batch_size=batch_size, duration=duration
            )
    finally:
        results_collector.stop_benchmark()
