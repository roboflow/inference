import os.path
from dataclasses import asdict
from datetime import datetime
from threading import Thread
from typing import Any, Dict, Optional

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
from inference_cli.lib.utils import dump_json, initialise_client


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
