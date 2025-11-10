"""
Benchmark tool for comparing old (base64 JSON) vs new (multipart) API performance.

This tool measures:
- Request parsing time
- End-to-end inference time
- Memory usage
- Throughput

Usage:
    python -m tests.benchmark.api_performance \\
        --host http://localhost:9001 \\
        --api-key <your-api-key> \\
        --model-id <model-id> \\
        --image-path <path-to-image>
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

try:
    import requests
    from PIL import Image
    import psutil
except ImportError:
    print("Missing dependencies. Install with: pip install requests pillow psutil")
    raise


class APIBenchmark:
    """Benchmark tool for comparing API versions."""

    def __init__(
        self,
        host: str,
        api_key: str,
        model_id: str,
    ):
        """
        Initialize the benchmark tool.

        Args:
            host: Base URL of the inference server (e.g., http://localhost:9001)
            api_key: Roboflow API key
            model_id: Model ID to test (e.g., 'project-id/version-id')
        """
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id

    def load_image(self, image_path: str) -> Tuple[bytes, int]:
        """
        Load an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (image_bytes, file_size_bytes)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            image_bytes = f.read()

        return image_bytes, len(image_bytes)

    def image_to_base64(self, image_bytes: bytes) -> str:
        """
        Convert image bytes to base64 string.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Base64 encoded string
        """
        return base64.b64encode(image_bytes).decode("utf-8")

    def benchmark_old_api(
        self,
        image_bytes: bytes,
        config: Dict,
        num_runs: int = 10,
    ) -> Dict:
        """
        Benchmark the old API (base64 JSON).

        Args:
            image_bytes: Raw image bytes
            config: Inference configuration
            num_runs: Number of times to run the test

        Returns:
            Dictionary with timing statistics
        """
        # Build endpoint URL
        url = f"{self.host}/infer/object_detection"

        # Convert image to base64
        print("Converting image to base64...")
        t_start = time.perf_counter()
        base64_image = self.image_to_base64(image_bytes)
        t_end = time.perf_counter()
        base64_encoding_time = t_end - t_start

        # Build request payload
        payload = {
            "model_id": self.model_id,
            "api_key": self.api_key,
            "image": {
                "type": "base64",
                "value": base64_image,
            },
            **config,
        }

        payload_size = len(json.dumps(payload))

        print(f"Running old API benchmark ({num_runs} iterations)...")
        times = []
        process = psutil.Process()

        for i in range(num_runs):
            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Make request
            t_start = time.perf_counter()
            response = requests.post(url, json=payload)
            t_end = time.perf_counter()

            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                raise Exception(f"Old API request failed: {response.status_code}")

            elapsed = t_end - t_start
            times.append(elapsed)
            mem_delta = mem_after - mem_before

            print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s (mem delta: {mem_delta:.1f}MB)")

        return {
            "times": times,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "base64_encoding_time": base64_encoding_time,
            "payload_size_bytes": payload_size,
        }

    def benchmark_new_api(
        self,
        image_bytes: bytes,
        config: Dict,
        num_runs: int = 10,
    ) -> Dict:
        """
        Benchmark the new API (multipart).

        Args:
            image_bytes: Raw image bytes
            config: Inference configuration
            num_runs: Number of times to run the test

        Returns:
            Dictionary with timing statistics
        """
        # Build endpoint URL
        url = f"{self.host}/v1/object-detection/{self.model_id}"

        # Build multipart request
        files = {
            "image": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg"),
            "config": (None, json.dumps(config), "application/json"),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Estimate payload size (approximate)
        payload_size = len(image_bytes) + len(json.dumps(config)) + 500  # +500 for multipart overhead

        print(f"Running new API benchmark ({num_runs} iterations)...")
        times = []
        process = psutil.Process()

        for i in range(num_runs):
            # Recreate files for each request (file pointer gets consumed)
            files = {
                "image": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg"),
                "config": (None, json.dumps(config), "application/json"),
            }

            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Make request
            t_start = time.perf_counter()
            response = requests.post(url, files=files, headers=headers)
            t_end = time.perf_counter()

            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                raise Exception(f"New API request failed: {response.status_code}")

            elapsed = t_end - t_start
            times.append(elapsed)
            mem_delta = mem_after - mem_before

            print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s (mem delta: {mem_delta:.1f}MB)")

        return {
            "times": times,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "base64_encoding_time": 0,  # No base64 encoding in new API
            "payload_size_bytes": payload_size,
        }

    def compare_apis(
        self,
        image_path: str,
        config: Dict = None,
        num_runs: int = 10,
    ):
        """
        Run full comparison benchmark between old and new APIs.

        Args:
            image_path: Path to test image
            config: Inference configuration (optional)
            num_runs: Number of runs per API
        """
        if config is None:
            config = {
                "confidence": 0.5,
                "iou_threshold": 0.4,
                "max_detections": 300,
            }

        # Load image
        print(f"\nLoading image: {image_path}")
        image_bytes, image_size = self.load_image(image_path)
        print(f"Image size: {image_size / 1024:.1f} KB ({image_size / 1024 / 1024:.2f} MB)")

        # Benchmark old API
        print("\n" + "=" * 60)
        print("BENCHMARKING OLD API (base64 JSON)")
        print("=" * 60)
        old_results = self.benchmark_old_api(image_bytes, config, num_runs)

        # Benchmark new API
        print("\n" + "=" * 60)
        print("BENCHMARKING NEW API (multipart)")
        print("=" * 60)
        new_results = self.benchmark_new_api(image_bytes, config, num_runs)

        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"\nImage size: {image_size / 1024:.1f} KB")
        print(f"Model: {self.model_id}")
        print(f"Runs: {num_runs}")

        print("\n--- OLD API (base64 JSON) ---")
        print(f"  Base64 encoding time: {old_results['base64_encoding_time']:.4f}s")
        print(f"  Payload size: {old_results['payload_size_bytes'] / 1024:.1f} KB")
        print(f"  Mean time: {old_results['mean']:.3f}s")
        print(f"  Median time: {old_results['median']:.3f}s")
        print(f"  Min time: {old_results['min']:.3f}s")
        print(f"  Max time: {old_results['max']:.3f}s")
        print(f"  Std dev: {old_results['stdev']:.3f}s")

        print("\n--- NEW API (multipart) ---")
        print(f"  Base64 encoding time: {new_results['base64_encoding_time']:.4f}s")
        print(f"  Payload size: {new_results['payload_size_bytes'] / 1024:.1f} KB")
        print(f"  Mean time: {new_results['mean']:.3f}s")
        print(f"  Median time: {new_results['median']:.3f}s")
        print(f"  Min time: {new_results['min']:.3f}s")
        print(f"  Max time: {new_results['max']:.3f}s")
        print(f"  Std dev: {new_results['stdev']:.3f}s")

        # Calculate improvements
        time_improvement = ((old_results['mean'] - new_results['mean']) / old_results['mean']) * 100
        size_improvement = ((old_results['payload_size_bytes'] - new_results['payload_size_bytes']) / old_results['payload_size_bytes']) * 100

        print("\n--- IMPROVEMENTS ---")
        print(f"  Time improvement: {time_improvement:+.1f}%")
        print(f"  Payload size reduction: {size_improvement:+.1f}%")
        print(f"  Speedup factor: {old_results['mean'] / new_results['mean']:.2f}x")

        return {
            "old_api": old_results,
            "new_api": new_results,
            "improvements": {
                "time_improvement_percent": time_improvement,
                "size_improvement_percent": size_improvement,
                "speedup_factor": old_results['mean'] / new_results['mean'],
            },
        }


def main():
    """Main entry point for the benchmark tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark old vs new API performance"
    )
    parser.add_argument(
        "--host",
        default="http://localhost:9001",
        help="Inference server host (default: http://localhost:9001)",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Roboflow API key",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID (e.g., 'project-id/version-id')",
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to test image",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per API (default: 10)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )

    args = parser.parse_args()

    config = {
        "confidence": args.confidence,
        "iou_threshold": 0.4,
        "max_detections": 300,
    }

    benchmark = APIBenchmark(
        host=args.host,
        api_key=args.api_key,
        model_id=args.model_id,
    )

    try:
        results = benchmark.compare_apis(
            image_path=args.image_path,
            config=config,
            num_runs=args.runs,
        )
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
