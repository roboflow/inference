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


def run_inference_models_benchmark(
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
        AutoModel.describe_model(model_id=model_id, api_key=api_key)
        if model_package_id:
            AutoModel.describe_model_package(
                model_id=model_id, package_id=model_package_id, api_key=api_key
            )
    except Exception as e:
        print(f"Could not fetch model profile from Roboflow API: {e}")
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
    model: AnyModel,
    inference_configuration: Dict[str, Any],
    image: Union[np.ndarray, torch.Tensor],
    warm_up_inferences: int,
    prompt: Optional[str] = None,
    stream: bool = False,
) -> None:
    if prompt is not None or stream:
        print("Skipping warm-up for VLM...")
        return
        
    import copy
    kwargs = copy.deepcopy(inference_configuration)
    if prompt is not None:
        kwargs["prompt"] = prompt
    if stream:
        kwargs["stream"] = stream
    for _ in tqdm(
        range(warm_up_inferences), desc="Warming up model...", total=warm_up_inferences
    ):
        result = model(image, **kwargs)
        if stream and hasattr(result, "__iter__") and not isinstance(result, dict) and not isinstance(result, list):
            for _ in result:
                pass



def run_benchmark(
    model: AnyModel,
    inference_configuration: Dict[str, Any],
    images: List[Union[np.ndarray, torch.Tensor]],
    results_collector: ResultsCollector,
    benchmark_inferences: int,
    batch_size: int,
    prompt: Optional[str] = None,
    stream: bool = False,
) -> None:
    import copy
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
                # VLMs do not support raw Numpy array ingest directly through standard pipeline decorators
                import base64
                from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
                
                vlm_payload = []
                for img in payload:
                    # Depending on initialization in this specific script images could be torch tensors
                    import numpy as np
                    if hasattr(img, "cpu"):
                        img = img.cpu().numpy()
                        if img.ndim == 3 and img.shape[0] in [1, 3]:
                            img = np.transpose(img, (1, 2, 0)) # CHW to HWC
                    
                    jpeg_bytes = encode_image_to_jpeg_bytes(img)
                    base64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
                    vlm_payload.append({"type": "base64", "value": base64_str})
                
                payload = vlm_payload[0] if len(vlm_payload) == 1 else vlm_payload

                first_token = True
                ttft_duration = 0.0
                tokens_generated = 0
                
                if stream:
                    print(f"\n[Prompt]: {prompt}\n[Output]: ", end="", flush=True)

                generator = model(payload, **kwargs)
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
                _ = model(payload, **kwargs)
            duration = time.time() - start
            results_collector.register_inference_duration(
                batch_size=batch_size, duration=duration
            )
    finally:
        results_collector.stop_benchmark()
