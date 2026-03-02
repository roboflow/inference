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
    prompt: Optional[str] = None,
    stream: bool = False,
) -> None:
    inference_configuration = {}
    if model_configuration is not None:
        inference_configuration = read_yaml_file(file_path=model_configuration)
    print(
        f"Inference will be executed with the following parameters: {inference_configuration}"
    )
    try:
        model_type = get_model_type(model_id, api_key=api_key)
        print(
            f"Model details | task_type={model_type[0]} | model_type={model_type[1]}"
        )
    except Exception as e:
        print(f"Could not fetch roboflow model type: {e}")
        
    model = get_model(model_id=model_id, api_key=api_key)
    model_batch_size = getattr(model, "batch_size", None)
    input_height = getattr(model, "img_size_h", None)
    input_width = getattr(model, "img_size_w", None)
    print(
        f"Model details | batch_size={model_batch_size} | input_height={input_height} | input_width={input_width}"
    )
    run_model_warm_up(
        model=model,
        inference_configuration=inference_configuration,
        image=images[0],
        warm_up_inferences=warm_up_inferences,
        prompt=prompt,
        stream=stream,
    )
    run_benchmark(
        model=model,
        inference_configuration=inference_configuration,
        images=images,
        results_collector=results_collector,
        benchmark_inferences=benchmark_inferences,
        batch_size=batch_size,
        prompt=prompt,
        stream=stream,
    )


def run_model_warm_up(
    model: Model,
    inference_configuration: Dict[str, Any],
    image: np.ndarray,
    warm_up_inferences: int,
    prompt: Optional[str] = None,
    stream: bool = False,
) -> None:
    if prompt is not None or stream:
        print("Skipping warm-up for VLM...")
        return
    
    kwargs = copy.deepcopy(inference_configuration)
    if prompt is not None:
        kwargs["prompt"] = prompt
    if stream:
        kwargs["stream"] = stream
    for _ in tqdm(
        range(warm_up_inferences), desc="Warming up model...", total=warm_up_inferences
    ):
        result = model.infer(image, **kwargs)
        if stream:
            for _ in result:
                pass



def run_benchmark(
    model: Model,
    inference_configuration: Dict[str, Any],
    images: List[np.ndarray],
    results_collector: ResultsCollector,
    benchmark_inferences: int,
    batch_size: int,
    prompt: Optional[str] = None,
    stream: bool = False,
) -> None:
    while len(images) < batch_size:
        images = images + images
    results_collector.start_benchmark()
    kwargs = copy.deepcopy(inference_configuration)
    if prompt is not None:
        kwargs["prompt"] = prompt
    if stream:
        kwargs["stream"] = stream

    try:
        for _ in range(benchmark_inferences):
            random.shuffle(images)
            payload = images[:batch_size]
            start = time.time()
            if stream and prompt:
                first_token = True
                ttft_duration = 0.0
                tokens_generated = 0
                
                if stream:
                    print(f"\n[Prompt]: {prompt}\n[Output]: ", end="", flush=True)

                generator = model.infer(payload, **kwargs)
                full_output = []
                for token in generator:
                    if first_token:
                        ttft_duration = time.time() - start
                        first_token = False
                    
                    tokens_generated += 1
                    if stream:
                        content = token
                        # some generators yield strings, some yield objects with `text` or `content` attribute
                        if hasattr(token, "text"):
                            content = token.text
                        elif hasattr(token, "content"):
                            content = token.content
                        elif isinstance(token, dict) and "text" in token:
                            content = token["text"]
                        
                        full_output.append(content)
                        print(content, end="", flush=True)
                
                with open("benchmark_vlm_output.txt", "a") as f:
                    f.write(f"--- Prompt: {prompt} ---\n{''.join(full_output)}\n\n")

                duration = time.time() - start
                results_collector.register_vlm_generation(
                    batch_size=batch_size, ttft=ttft_duration, tokens_generated=tokens_generated
                )
                if stream:
                    print("\n" + "-"*40)
            else:
                _ = model.infer(payload, **kwargs)
            duration = time.time() - start
            results_collector.register_inference_duration(
                batch_size=batch_size, duration=duration
            )
    finally:
        results_collector.stop_benchmark()
