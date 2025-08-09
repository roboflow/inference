import json
import os.path
from dataclasses import asdict
from datetime import datetime
from threading import Thread
from typing import Any, Dict, Optional

from inference.core import logger
from inference_cli.lib.benchmark.api_speed import (
    coordinate_infer_api_speed_benchmark,
    coordinate_workflow_api_speed_benchmark,
    display_benchmark_statistics,
)
from inference_cli.lib.benchmark.dataset import load_dataset_images
from inference_cli.lib.benchmark.platform import retrieve_platform_specifics
from inference_cli.lib.benchmark.results_gathering import (
    InferenceStatistics,
    ResultsCollector,
)
from inference_cli.lib.utils import (
    dump_json,
    ensure_inference_is_installed,
    initialise_client,
)


def run_infer_api_speed_benchmark(
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
    enforce_legacy_endpoints: bool = False,
    max_error_rate: Optional[float] = None,
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
    if enforce_legacy_endpoints:
        client.select_api_v0()
    benchmark_results = coordinate_infer_api_speed_benchmark(
        client=client,
        images=dataset_images,
        model_id=model_id,
        warm_up_requests=warm_up_requests,
        benchmark_requests=benchmark_requests,
        request_batch_size=request_batch_size,
        number_of_clients=number_of_clients,
        requests_per_second=requests_per_second,
    )
    if output_location is None:
        ensure_error_rate_is_below_threshold(
            error_rate=benchmark_results.error_rate,
            threshold=max_error_rate,
        )
        return None
    benchmark_parameters = {
        "datetime": datetime.now().isoformat(),
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
        output_location=output_location,
        benchmark_parameters=benchmark_parameters,
        benchmark_results=benchmark_results,
    )
    ensure_error_rate_is_below_threshold(
        error_rate=benchmark_results.error_rate,
        threshold=max_error_rate,
    )


def run_workflow_api_speed_benchmark(
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    workflow_specification: Optional[str],
    workflow_parameters: Optional[Dict[str, Any]],
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
    max_error_rate: Optional[float] = None,
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
    benchmark_results = coordinate_workflow_api_speed_benchmark(
        client=client,
        images=dataset_images,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        workflow_specification=workflow_specification,
        workflow_parameters=workflow_parameters,
        benchmark_requests=benchmark_requests,
        request_batch_size=request_batch_size,
        number_of_clients=number_of_clients,
        requests_per_second=requests_per_second,
    )
    if output_location is None:
        ensure_error_rate_is_below_threshold(
            error_rate=benchmark_results.error_rate,
            threshold=max_error_rate,
        )
        return None
    benchmark_parameters = {
        "datetime": datetime.now().isoformat(),
        "dataset_reference": dataset_reference,
        "host": host,
        "benchmark_inferences": benchmark_requests,
        "batch_size": request_batch_size,
        "number_of_clients": number_of_clients,
        "requests_per_second": requests_per_second,
        "model_configuration": model_configuration,
    }
    if workflow_id and workspace_name:
        benchmark_parameters["workflow_id"] = workflow_id
        benchmark_parameters["workspace_name"] = workspace_name
    else:
        benchmark_parameters["workflow_id"] = "locally defined"
    dump_benchmark_results(
        output_location=output_location,
        benchmark_parameters=benchmark_parameters,
        benchmark_results=benchmark_results,
    )
    ensure_error_rate_is_below_threshold(
        error_rate=benchmark_results.error_rate,
        threshold=max_error_rate,
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
    ensure_inference_is_installed()

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
    statistics_display_thread.daemon = (
        True  # Make thread daemon so it doesn't block exit
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
    if output_location is None:
        return None
    benchmark_parameters = {
        "datetime": datetime.now().isoformat(),
        "model_id": model_id,
        "dataset_reference": dataset_reference,
        "benchmark_inferences": benchmark_inferences,
        "batch_size": batch_size,
        "model_configuration": model_configuration,
    }
    dump_benchmark_results(
        output_location=output_location,
        benchmark_parameters=benchmark_parameters,
        benchmark_results=benchmark_results,
    )


def dump_benchmark_results(
    output_location: str,
    benchmark_parameters: dict,
    benchmark_results: InferenceStatistics,
) -> None:
    platform_specifics = retrieve_platform_specifics()
    if os.path.isdir(output_location):
        target_path = os.path.join(
            output_location, f"{datetime.now().isoformat()}.json"
        )
    else:
        target_path = output_location
    print(f"Saving statistics under: {target_path}")
    results = {
        "benchmark_parameters": benchmark_parameters,
        "benchmark_results": asdict(benchmark_results),
        "platform": platform_specifics,
    }
    dump_json(path=target_path, content=results)


def run_pipeline_speed_benchmark(
    video_reference: str,
    model_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workspace_name: Optional[str] = None,
    workflow_specification: Optional[str] = None,
    workflow_parameters: Optional[str] = None,
    duration_seconds: int = 60,
    max_fps: Optional[float] = None,
    api_key: Optional[str] = None,
    model_configuration: Optional[str] = None,
    output_location: Optional[str] = None,
    num_pipelines: int = 1,
) -> None:
    ensure_inference_is_installed()

    # importing here not to affect other entrypoints by missing `inference` core library
    # Add a hard timeout for the entire benchmark
    import signal

    from inference_cli.lib.benchmark.pipeline_speed import run_pipeline_speed_benchmark

    def timeout_handler(signum, frame):
        logger.warning("Benchmark exceeded maximum allowed time, forcing termination")
        import sys

        sys.exit(1)

    # Set alarm for duration + 30 seconds grace period
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration_seconds + 30)

    # Convert video_reference to int if it's a numeric string (for webcam)
    processed_video_reference = video_reference
    try:
        # Try to convert to int - handles positive, negative, and zero
        processed_video_reference = int(video_reference)
    except ValueError:
        # Not a number, keep as string (for file paths or URLs)
        pass

    # Create video sources for all pipelines
    video_sources = [processed_video_reference] * num_pipelines

    # Parse JSON parameters if provided
    parsed_workflow_specification = None
    parsed_workflow_parameters = None

    if workflow_specification:
        try:
            parsed_workflow_specification = json.loads(workflow_specification)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in workflow_specification: {e}")

    if workflow_parameters:
        try:
            parsed_workflow_parameters = json.loads(workflow_parameters)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in workflow_parameters: {e}")

    results_collector = ResultsCollector()
    statistics_display_thread = Thread(
        target=display_benchmark_statistics, args=(results_collector,)
    )
    statistics_display_thread.daemon = (
        True  # Make thread daemon so it doesn't block exit
    )
    statistics_display_thread.start()

    run_pipeline_speed_benchmark(
        video_sources=video_sources,
        results_collector=results_collector,
        model_id=model_id,
        workflow_id=workflow_id,
        workspace_name=workspace_name,
        workflow_specification=parsed_workflow_specification,
        workflow_parameters=parsed_workflow_parameters,
        duration_seconds=duration_seconds,
        max_fps=max_fps,
        api_key=api_key,
        model_configuration=model_configuration,
    )

    benchmark_results = results_collector.get_statistics()

    # Force stop the results collector to ensure display thread exits
    results_collector.stop_benchmark()

    # Wait for statistics thread with timeout
    # The display thread now checks every 100ms for completion, so it should exit quickly
    statistics_display_thread.join(timeout=5.0)
    if statistics_display_thread.is_alive():
        logger.warning("Statistics display thread did not terminate properly")
        # Force terminate by making it a daemon thread (already is) and continuing

    # Benchmark completion is already reported by run_pipeline_speed_benchmark
    print("\nBenchmark completed successfully!")

    if output_location is None:
        return None

    benchmark_parameters = {
        "datetime": datetime.now().isoformat(),
        "video_reference": video_reference,
        "duration_seconds": duration_seconds,
        "num_pipelines": num_pipelines,
        "max_fps": max_fps,
        "model_configuration": model_configuration,
    }

    if model_id:
        benchmark_parameters["model_id"] = model_id
    else:
        if workflow_id and workspace_name:
            benchmark_parameters["workflow_id"] = workflow_id
            benchmark_parameters["workspace_name"] = workspace_name
        else:
            benchmark_parameters["workflow_id"] = "locally defined"
        if workflow_parameters:
            benchmark_parameters["workflow_parameters"] = workflow_parameters

    dump_benchmark_results(
        output_location=output_location,
        benchmark_parameters=benchmark_parameters,
        benchmark_results=benchmark_results,
    )

    # Cancel the alarm if we finished normally
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


def ensure_error_rate_is_below_threshold(
    error_rate: float, threshold: Optional[float]
) -> None:
    if threshold is None:
        return None
    if error_rate <= threshold:
        return None
    raise RuntimeError(
        f"Benchmark error rate: {error_rate}% is higher than threshold ({threshold}%)"
    )
