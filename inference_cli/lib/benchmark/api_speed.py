import math
import random
import time
from functools import partial
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests
from tqdm import tqdm

from inference_cli.lib.benchmark.results_gathering import (
    InferenceStatistics,
    ResultsCollector,
)
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.entities import HTTPClientMode


def run_api_warm_up(
    client: InferenceHTTPClient,
    image: np.ndarray,
    warm_up_requests: int,
) -> None:
    for _ in tqdm(
        range(warm_up_requests), desc="Warming up API...", total=warm_up_requests
    ):
        try:
            _ = client.infer(inference_input=image)
        except Exception:
            # ignoring errors, without slight API instability may terminate benchmark
            pass


def coordinate_infer_api_speed_benchmark(
    client: InferenceHTTPClient,
    images: List[np.ndarray],
    model_id: str,
    warm_up_requests: int,
    benchmark_requests: int,
    request_batch_size: int,
    number_of_clients: int,
    requests_per_second: Optional[int],
) -> InferenceStatistics:
    run_api_warm_up(client=client, image=images[0], warm_up_requests=warm_up_requests)
    image_sizes = {i.shape[:2] for i in images}
    print(f"Detected images dimensions: {image_sizes}")
    if client.client_mode is HTTPClientMode.V1:
        model_details = client.get_model_description(model_id=model_id)
        print(
            f"Model details | task_type={model_details.task_type} | batch_size={model_details.batch_size} | "
            f"input_height={model_details.input_height} | input_width={model_details.input_width}"
        )
    results_collector = ResultsCollector()
    statistics_display_thread = Thread(
        target=display_benchmark_statistics, args=(results_collector,)
    )
    statistics_display_thread.start()
    execute_infer_api_speed_benchmark(
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


def coordinate_workflow_api_speed_benchmark(
    client: InferenceHTTPClient,
    images: List[np.ndarray],
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    workflow_specification: Optional[str],
    workflow_parameters: Optional[Dict[str, Any]],
    benchmark_requests: int,
    request_batch_size: int,
    number_of_clients: int,
    requests_per_second: Optional[int],
) -> InferenceStatistics:
    image_sizes = {i.shape[:2] for i in images}
    print(f"Detected images dimensions: {image_sizes}")
    results_collector = ResultsCollector()
    statistics_display_thread = Thread(
        target=display_benchmark_statistics, args=(results_collector,)
    )
    statistics_display_thread.start()
    execute_workflow_api_speed_benchmark(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        workflow_specification=workflow_specification,
        workflow_parameters=workflow_parameters,
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


def execute_infer_api_speed_benchmark(
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
        execute_infer_api_request,
        results_collector=results_collector,
        client=client,
        images=images,
        request_batch_size=request_batch_size,
        delay=requests_per_second is not None,
    )
    if requests_per_second is not None:
        if number_of_clients is not None:
            print(
                "Parameter specifying `number_of_clients` is ignored when number of "
                "RPS to maintain is specified."
            )
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


def execute_workflow_api_speed_benchmark(
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    workflow_specification: Optional[str],
    workflow_parameters: Optional[Dict[str, Any]],
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
        execute_workflow_api_request,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        workflow_specification=workflow_specification,
        workflow_parameters=workflow_parameters,
        results_collector=results_collector,
        client=client,
        images=images,
        request_batch_size=request_batch_size,
        delay=requests_per_second is not None,
    )
    if requests_per_second is not None:
        if number_of_clients is not None:
            print(
                "Parameter specifying `number_of_clients` is ignored when number of "
                "RPS to maintain is specified."
            )
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
    threads = []
    for _ in range(rounds):
        start = time.time()
        for _ in range(requests_per_second):
            executor_thread = Thread(target=executor, daemon=True)
            executor_thread.start()
            threads.append(executor_thread)
        still_alive_threads = []
        for thread in threads:
            thread.join(timeout=0.0)
            if thread.is_alive():
                still_alive_threads.append(thread)
        threads = still_alive_threads
        duration = time.time() - start
        remaining = max(0.0, 1.0 - duration)
        time.sleep(remaining)
    for thread in threads:
        thread.join()


def execute_infer_api_request(
    results_collector: ResultsCollector,
    client: InferenceHTTPClient,
    images: List[np.ndarray],
    request_batch_size: int,
    delay: bool = False,
) -> None:
    if delay:
        time.sleep(random.random())
    random.shuffle(images)
    payload = images[:request_batch_size]
    start = time.time()
    try:
        _ = client.infer(payload)
        duration = time.time() - start
        results_collector.register_inference_duration(
            batch_size=request_batch_size, duration=duration
        )
    except Exception as exc:
        duration = time.time() - start
        results_collector.register_inference_duration(
            batch_size=request_batch_size, duration=duration
        )
        status_code = exc.__class__.__name__
        if isinstance(exc, requests.exceptions.HTTPError):
            status_code = str(exc.response.status_code)

        results_collector.register_error(
            batch_size=request_batch_size, status_code=status_code
        )


def execute_workflow_api_request(
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    workflow_specification: Optional[str],
    workflow_parameters: Optional[Dict[str, Any]],
    results_collector: ResultsCollector,
    client: InferenceHTTPClient,
    images: List[np.ndarray],
    request_batch_size: int,
    delay: bool = False,
) -> None:
    if delay:
        time.sleep(random.random())
    random.shuffle(images)
    images = {f"image": images[:request_batch_size]}
    start = time.time()
    try:
        kwargs = {
            "images": images,
        }
        if workflow_parameters:
            kwargs["parameters"] = workflow_parameters
        if workspace_name and workflow_id:
            kwargs["workspace_name"] = workspace_name
            kwargs["workflow_id"] = workflow_id
        else:
            kwargs["specification"] = workflow_specification
        _ = client.run_workflow(**kwargs)
        duration = time.time() - start
        results_collector.register_inference_duration(
            batch_size=request_batch_size, duration=duration
        )
    except Exception as exc:
        duration = time.time() - start
        results_collector.register_inference_duration(
            batch_size=request_batch_size, duration=duration
        )
        status_code = exc.__class__.__name__
        if isinstance(exc, requests.exceptions.HTTPError):
            status_code = str(exc.response.status_code)

        results_collector.register_error(
            batch_size=request_batch_size, status_code=status_code
        )


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
