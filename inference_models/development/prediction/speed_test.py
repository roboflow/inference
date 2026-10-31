import argparse
import os.path
import time
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from inference_models import AutoModel
from inference_models.models.auto_loaders.entities import AnyModel

from .dataset import download_dataset
from .serialization import dump_json


def main(
    model_id: str,
    output_dir: str,
    model_package_id: Optional[str] = None,
) -> None:
    print(f"Testing model speed: {model_id}")
    results_path = os.path.join(output_dir, "benchmark.json")
    if os.path.exists(results_path):
        print("Skipping the tests, results exist.")
        return None
    dataset = download_dataset()
    images = [e[1] for e in dataset]
    model = AutoModel.from_pretrained(
        model_id_or_path=model_id, model_package_id=model_package_id
    )
    for _ in tqdm(range(20), desc="Model warm-up...", total=20):
        _ = model(images[0])
    batch_sizes = [1, 4, 8, 16]
    results = []
    for batch_size in batch_sizes:
        result = run_test_with_batch_size(
            model_id=model_id,
            images=images,
            model=model,
            batch_size=batch_size,
        )
        results.append(result)
    dump_json(path=results_path, content=results)


def run_test_with_batch_size(
    model_id: str,
    images: List[np.ndarray],
    model: AnyModel,
    batch_size: int,
) -> dict:
    while len(images) < batch_size:
        images = images + images
    images = images[:batch_size]
    if batch_size == 1:
        images = images[0]
    iterations = round(1000 / batch_size)
    start = time.monotonic()
    for _ in tqdm(
        range(iterations), desc=f"Benchmarking bs={batch_size}...", total=iterations
    ):
        _ = model(images)
    end = time.monotonic()
    total_time = end - start
    throughput = round((iterations * batch_size) / total_time, 2)
    avg_img_latency = round((total_time / (iterations * batch_size)) * 1000, 2)
    avg_batch_latency = round((total_time / iterations) * 1000, 2)
    return {
        "model_id": model_id,
        "batch_size": batch_size,
        "throughput": throughput,
        "avg_img_latency": avg_img_latency,
        "avg_batch_latency": avg_batch_latency,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_package_id", type=str, required=False, default=None)
    args = parser.parse_args()
    main(
        model_id=args.model_id,
        output_dir=args.output_dir,
        model_package_id=args.model_package_id,
    )
