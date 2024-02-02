import math
import os.path
import random
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from threading import Thread
from typing import Callable, List, Optional

import numpy as np
from tqdm import tqdm

from inference_cli.lib.benchmark.dataset import load_dataset_images
from inference_cli.lib.benchmark.results_gathering import (
    InferenceStatistics,
    ResultsCollector,
)
from inference_cli.lib.utils import dump_json, initialise_client
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.entities import HTTPClientMode


def run_api_speed_benchmark(
    model_id: str,
    dataset_reference: str,
    host: str,
    warm_up_requests: int = 10,
    benchmark_requests: int = 1000,
    request_batch_size: int = 1,
    number_of_clients: int = 1,
    requests_per_second: Optional[int] = None,
    api_key: Optional[str] = None,
    model_configuration: Optional[str] = None,
    output_location: Optional[str] = None,
) -> None:
    dataset_images = load_dataset_images(
        dataset_reference=dataset_reference,
    )
    client = initialise_client(
        host=host,
        api_key=api_key,
        model_configuration=model_configuration,
        disable_active_learning=True,
        max_concurrent_requests=1,
        max_batch_size=request_batch_size,
    )
    client.select_model(model_id=model_id)
    run_api_warm_up(
        client=client, image=dataset_images[0], warm_up_requests=warm_up_requests
    )
    image_sizes = {i.shape[:2] for i in dataset_images}
    print(f"Detected images dimensions: {image_sizes}")
    if client.client_mode is HTTPClientMode.V1:
        model_details = client.get_model_description(model_id=model_id)
        print(
            f"Model details | task_type={model_details.task_type} | batch_size={model_details.batch_size} | "
            f"input_height={model_details.input_height} | input_width={model_details.input_width}"
        )
    benchmark_results = coordinate_api_speed_benchmark(
        client=client,
        images=dataset_images,
        benchmark_requests=benchmark_requests,
        request_batch_size=request_batch_size,
        number_of_clients=number_of_clients,
        requests_per_second=requests_per_second,
    )
    if output_location is not None:
        benchmark_parameters = {
            "model_id": model_id,
            "dataset_reference": dataset_reference,
            "host": host,
            "benchmark_inferences": benchmark_requests,
            "batch_size": request_batch_size,
            "number_of_clients": number_of_clients,
            "requests_per_second": requests_per_second,
            "model_configuration": model_configuration,
        }
        dump_benchmark_results(
            output_directory=output_location,
            benchmark_parameters=benchmark_parameters,
            benchmark_results=benchmark_results,
        )


def run_python_package_speed_benchmark(
    model_id: str,
    dataset_reference: str,
    warm_up_inferences: int = 10,
    benchmark_inferences: int = 1000,
    batch_size: int = 1,
    api_key: Optional[str] = None,
    model_configuration: Optional[str] = None,
    output_location: Optional[str] = None,
) -> None:
    # importing here not to affect other entrypoints by missing `inference` core library
    from inference_cli.lib.benchmark.python_package_speed import (
        run_python_package_speed_benchmark,
    )

    dataset_images = load_dataset_images(
        dataset_reference=dataset_reference,
    )
    image_sizes = {i.shape[:2] for i in dataset_images}
    print(f"Detected images dimensions: {image_sizes}")
    results_collector = ResultsCollector()
    statistics_display_thread = Thread(
        target=display_benchmark_statistics, args=(results_collector,)
    )
    statistics_display_thread.start()
    run_python_package_speed_benchmark(
        model_id=model_id,
        images=dataset_images,
        results_collector=results_collector,
        warm_up_inferences=warm_up_inferences,
        benchmark_inferences=benchmark_inferences,
        batch_size=batch_size,
        api_key=api_key,
        model_configuration=model_configuration,
    )
    benchmark_results = results_collector.get_statistics()
    statistics_display_thread.join()
    if output_location is not None:
        benchmark_parameters = {
            "model_id": model_id,
            "dataset_reference": dataset_reference,
            "benchmark_inferences": benchmark_inferences,
            "batch_size": batch_size,
            "model_configuration": model_configuration,
        }
        dump_benchmark_results(
            output_directory=output_location,
            benchmark_parameters=benchmark_parameters,
            benchmark_results=benchmark_results,
        )


def run_api_warm_up(
    client: InferenceHTTPClient,
    image: np.ndarray,
    warm_up_requests: int,
) -> None:
    for _ in tqdm(
        range(warm_up_requests), desc="Warming up API...", total=warm_up_requests
    ):
        _ = client.infer(inference_input=image)


def coordinate_api_speed_benchmark(
    client: InferenceHTTPClient,
    images: List[np.ndarray],
    benchmark_requests: int,
    request_batch_size: int,
    number_of_clients: int,
    requests_per_second: Optional[int],
) -> InferenceStatistics:
    results_collector = ResultsCollector()
    statistics_display_thread = Thread(
        target=display_benchmark_statistics, args=(results_collector,)
    )
    statistics_display_thread.start()
    execute_api_speed_benchmark(
        results_collector=results_collector,
        client=client,
        images=images,
        benchmark_requests=benchmark_requests,
        request_batch_size=request_batch_size,
        number_of_clients=number_of_clients,
        requests_per_second=requests_per_second,
    )
    statistics = results_collector.get_statistics()
    statistics_display_thread.join()
    return statistics


def display_benchmark_statistics(
    results_collector: ResultsCollector,
    sleep_time: float = 5.0,
    window: int = 100,
) -> None:
    while not results_collector.has_benchmark_finished():
        statistics = results_collector.get_statistics(window=window)
        if statistics is None:
            time.sleep(sleep_time)
            continue
        print(statistics.to_string())
        time.sleep(sleep_time)


def execute_api_speed_benchmark(
    results_collector: ResultsCollector,
    client: InferenceHTTPClient,
    images: List[np.ndarray],
    benchmark_requests: int,
    request_batch_size: int,
    number_of_clients: int,
    requests_per_second: Optional[int],
) -> None:
    while len(images) < request_batch_size:
        images = images + images
    api_request_executor = partial(
        execute_api_request,
        results_collector=results_collector,
        client=client,
        images=images,
        request_batch_size=request_batch_size,
    )
    if requests_per_second is not None:
        results_collector.start_benchmark()
        execute_given_rps_sequentially(
            executor=api_request_executor,
            benchmark_requests=benchmark_requests,
            requests_per_second=requests_per_second,
        )
        results_collector.stop_benchmark()
        return None
    client_threads = []
    results_collector.start_benchmark()
    for _ in range(number_of_clients):
        client_thread = Thread(
            target=execute_requests_sequentially,
            args=(api_request_executor, benchmark_requests),
        )
        client_thread.start()
        client_threads.append(client_thread)
    for thread in client_threads:
        thread.join()
    results_collector.stop_benchmark()
    return None


def execute_requests_sequentially(
    executor: Callable[[], None], benchmark_requests: int
) -> None:
    for _ in range(benchmark_requests):
        executor()


def execute_given_rps_sequentially(
    executor: Callable[[], None],
    benchmark_requests: int,
    requests_per_second: int,
) -> None:
    rounds = math.ceil(benchmark_requests / requests_per_second)
    for _ in range(rounds):
        start = time.time()
        for _ in range(requests_per_second):
            executor_thread = Thread(target=executor, daemon=True)
            executor_thread.start()
        duration = time.time() - start
        remaining = max(0.0, 1.0 - duration)
        time.sleep(remaining)


def execute_api_request(
    results_collector: ResultsCollector,
    client: InferenceHTTPClient,
    images: List[np.ndarray],
    request_batch_size: int,
) -> None:
    try:
        random.shuffle(images)
        payload = images[:request_batch_size]
        start = time.time()
        _ = client.infer(payload)
        duration = time.time() - start
        results_collector.register_inference_duration(
            batch_size=request_batch_size, duration=duration
        )
    except Exception:
        results_collector.register_error(batch_size=request_batch_size)


def dump_benchmark_results(
    output_directory: str,
    benchmark_parameters: dict,
    benchmark_results: InferenceStatistics,
) -> None:
    target_path = os.path.join(output_directory, f"{datetime.now().isoformat()}.json")
    print(f"Saving statistics under: {target_path}")
    results = {
        "benchmark_parameters": benchmark_parameters,
        "benchmark_results": asdict(benchmark_results),
    }
    dump_json(path=target_path, content=results)
