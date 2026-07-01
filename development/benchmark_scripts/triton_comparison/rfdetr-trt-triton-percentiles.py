# pyright: reportMissingImports=false
"""Benchmark RF-DETR TRT latency through Triton HTTP.

This script is intended to run inside a Triton-specific image. Triton normally
accepts a pre-processed tensor and runs the TRT engine, so this benchmark replays
RF-DETR preprocessing from the model package on the client side and reports that
preparation latency separately from the Triton HTTP request latency.

Example:

    python development/benchmark_scripts/triton_comparison/rfdetr-trt-triton-percentiles.py \\
        --source /input/images \\
        --model-package-dir /models/rfdetr-small-trt-package \\
        --triton-url localhost:8000 \\
        --triton-model-name rfdetr_small \\
        --result-out /output/triton-benchmark.json
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

import click
import cv2
import numpy as np


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass
class _TimedCall:
    total_latency_ms: float
    request_latency_ms: float
    client_prepare_ms: float


@dataclass
class _BenchmarkResult:
    profile: str
    frames: int
    elapsed: float
    aggregate_fps: float
    total_latency_ms: dict[str, Optional[float]]
    request_latency_ms: dict[str, Optional[float]]
    client_prepare_ms: dict[str, Optional[float]]
    total_latency_reads_ms: list[float]
    request_latency_reads_ms: list[float]
    client_prepare_reads_ms: list[float]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        result = {
            "profile": self.profile,
            "frames": self.frames,
            "elapsed": self.elapsed,
            "aggregate_fps": self.aggregate_fps,
            "fps": self.aggregate_fps,
            "total_latency_ms": self.total_latency_ms,
            "request_latency_ms": self.request_latency_ms,
            "client_prepare_ms": self.client_prepare_ms,
            "total_latency_reads_ms": self.total_latency_reads_ms,
            "request_latency_reads_ms": self.request_latency_reads_ms,
            "client_prepare_reads_ms": self.client_prepare_reads_ms,
            "metadata": self.metadata,
        }

        return result


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None

    ordered_values = sorted(values)
    if len(ordered_values) == 1:
        return ordered_values[0]

    position = (len(ordered_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    fraction = position - lower_index
    interpolated_value = (
        ordered_values[lower_index] * (1.0 - fraction)
        + ordered_values[upper_index] * fraction
    )

    return interpolated_value


def _percentiles(values: list[float]) -> dict[str, Optional[float]]:
    percentile_values = {
        "p_50": _percentile(values, 0.50),
        "p_95": _percentile(values, 0.95),
        "p_99": _percentile(values, 0.99),
    }

    return percentile_values


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"

    formatted_value = f"{value:.2f}"

    return formatted_value


def _format_percentiles(percentiles: dict[str, Optional[float]]) -> str:
    rendered_percentiles = (
        f"p50={_format_optional_float(percentiles['p_50'])} "
        f"p95={_format_optional_float(percentiles['p_95'])} "
        f"p99={_format_optional_float(percentiles['p_99'])}"
    )

    return rendered_percentiles


def _read_image_rgb(path: Path, resize_wh: tuple[int, int]) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")

    if (image_bgr.shape[1], image_bgr.shape[0]) != resize_wh:
        image_bgr = cv2.resize(image_bgr, resize_wh, interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def _iter_image_paths(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def _load_frames(
    *,
    source: str,
    resize_wh: tuple[int, int],
    frame_limit: Optional[int],
    stride: int,
    repeat: int,
) -> list[np.ndarray]:
    source_path = Path(source)
    frames: list[np.ndarray] = []

    if source_path.is_dir():
        for image_path in _iter_image_paths(source_path):
            frames.append(_read_image_rgb(path=image_path, resize_wh=resize_wh))
            if frame_limit is not None and len(frames) >= frame_limit:
                break
    elif source_path.suffix.lower() in _IMAGE_SUFFIXES:
        frames.append(_read_image_rgb(path=source_path, resize_wh=resize_wh))
    else:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        frame_index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            if frame_index % stride == 0:
                if (frame_bgr.shape[1], frame_bgr.shape[0]) != resize_wh:
                    frame_bgr = cv2.resize(
                        frame_bgr,
                        resize_wh,
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if frame_limit is not None and len(frames) >= frame_limit:
                    break

            frame_index += 1
        capture.release()

    if not frames:
        raise ValueError(f"No frames loaded from source: {source}")

    if repeat > 1:
        frames = frames * repeat

    return frames


class _RFDetrPackagePreprocessor:
    def __init__(self, model_package_dir: str) -> None:
        import torch
        from inference_models.models.common.model_packages import (
            get_model_package_contents,
        )
        from inference_models.models.common.roboflow.model_packages import (
            ResizeMode,
            parse_inference_config,
        )
        from inference_models.models.rfdetr.pre_processing import (
            pre_process_network_input,
        )

        package_contents = get_model_package_contents(
            model_package_dir=model_package_dir,
            elements=["inference_config.json"],
        )
        self._inference_config = parse_inference_config(
            config_path=package_contents["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
            },
        )
        self._torch = torch
        self._pre_process_network_input = pre_process_network_input

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        batch, _ = self._pre_process_network_input(
            images=image_rgb,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._torch.device("cpu"),
            input_color_format="rgb",
        )
        preprocessed_tensor = np.ascontiguousarray(
            batch.cpu().numpy().astype(np.float32)
        )

        return preprocessed_tensor


def _metadata_items(metadata: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(metadata, dict):
        items = list(metadata.get(key, []))

        return items

    items = list(getattr(metadata, key, []))

    return items


def _metadata_name(item: Any) -> str:
    if isinstance(item, dict):
        name = item["name"]

        return name

    name = item.name

    return name


class _TritonRunner:
    def __init__(
        self,
        *,
        url: str,
        model_name: str,
        model_version: str,
        input_name: Optional[str],
        output_names: Optional[list[str]],
        model_package_dir: str,
        timeout: float,
    ) -> None:
        try:
            import tritonclient.http as httpclient
            from tritonclient.utils import np_to_triton_dtype
        except ImportError as error:
            raise ImportError(
                "Triton benchmark requires tritonclient[http]. Install it in the "
                "benchmark environment before running this script."
            ) from error

        self._httpclient = httpclient
        self._np_to_triton_dtype = np_to_triton_dtype
        self._client = httpclient.InferenceServerClient(
            url=url,
            connection_timeout=timeout,
            network_timeout=timeout,
        )
        self._model_name = model_name
        self._model_version = model_version
        self._timeout = timeout
        self._preprocessor = _RFDetrPackagePreprocessor(model_package_dir)
        metadata = self._client.get_model_metadata(
            model_name=model_name,
            model_version=model_version,
        )
        inputs = _metadata_items(metadata, "inputs")
        outputs = _metadata_items(metadata, "outputs")
        if input_name is None:
            if len(inputs) != 1:
                raise ValueError(
                    "Triton input name is ambiguous. Pass --triton-input-name."
                )
            input_name = _metadata_name(inputs[0])
        if output_names is None:
            output_names = [_metadata_name(output) for output in outputs]

        self._input_name = input_name
        self._output_names = output_names
        self._metadata = metadata

    def __call__(self, image_rgb: np.ndarray) -> _TimedCall:
        start = perf_counter()
        tensor = self._preprocessor(image_rgb)
        prepared = perf_counter()
        infer_input = self._httpclient.InferInput(
            self._input_name,
            tensor.shape,
            self._np_to_triton_dtype(tensor.dtype),
        )
        infer_input.set_data_from_numpy(tensor)
        outputs = [
            self._httpclient.InferRequestedOutput(output_name)
            for output_name in self._output_names
        ]
        self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=[infer_input],
            outputs=outputs,
            request_compression_algorithm=None,
            response_compression_algorithm=None,
            timeout=self._timeout,
        )
        finished = perf_counter()
        timed_call = _TimedCall(
            total_latency_ms=(finished - start) * 1000.0,
            request_latency_ms=(finished - prepared) * 1000.0,
            client_prepare_ms=(prepared - start) * 1000.0,
        )

        return timed_call

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = {
            "model_name": self._model_name,
            "model_version": self._model_version,
            "input_name": self._input_name,
            "output_names": self._output_names,
            "model_metadata": self._metadata,
        }

        return metadata


def _run_profile(
    *,
    profile: str,
    runner: Callable[[np.ndarray], _TimedCall],
    metadata: dict[str, Any],
    frames: list[np.ndarray],
    warmup: int,
    progress_every: int,
) -> _BenchmarkResult:
    if warmup:
        print(f"[benchmark] profile={profile} warmup={warmup}", flush=True)

    for index in range(warmup):
        runner(frames[index % len(frames)])

    print(f"[benchmark] profile={profile} frames={len(frames)}", flush=True)
    total_latencies: list[float] = []
    request_latencies: list[float] = []
    prepare_latencies: list[float] = []
    started = perf_counter()
    for index, frame in enumerate(frames, start=1):
        timed_call = runner(frame)
        total_latencies.append(timed_call.total_latency_ms)
        request_latencies.append(timed_call.request_latency_ms)
        prepare_latencies.append(timed_call.client_prepare_ms)
        if progress_every > 0 and index % progress_every == 0:
            elapsed = perf_counter() - started
            fps = index / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] profile={profile} frames={index} fps={fps:.2f}",
                flush=True,
            )

    elapsed = perf_counter() - started
    aggregate_fps = len(frames) / elapsed if elapsed > 0 else 0.0
    result = _BenchmarkResult(
        profile=profile,
        frames=len(frames),
        elapsed=elapsed,
        aggregate_fps=aggregate_fps,
        total_latency_ms=_percentiles(total_latencies),
        request_latency_ms=_percentiles(request_latencies),
        client_prepare_ms=_percentiles(prepare_latencies),
        total_latency_reads_ms=total_latencies,
        request_latency_reads_ms=request_latencies,
        client_prepare_reads_ms=prepare_latencies,
        metadata=metadata,
    )
    _print_profile_result(result)

    return result


def _print_profile_result(result: _BenchmarkResult) -> None:
    print(
        f"[benchmark] profile={result.profile} frames={result.frames} "
        f"elapsed={result.elapsed:.2f}s aggregate_fps={result.aggregate_fps:.2f}",
        flush=True,
    )
    print(
        f"[benchmark] total_latency_ms "
        f"{_format_percentiles(result.total_latency_ms)}",
        flush=True,
    )
    print(
        f"[benchmark] request_latency_ms "
        f"{_format_percentiles(result.request_latency_ms)}",
        flush=True,
    )
    print(
        f"[benchmark] client_prepare_ms "
        f"{_format_percentiles(result.client_prepare_ms)}",
        flush=True,
    )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


@click.command()
@click.option(
    "--source",
    required=True,
    help="Image file, image directory, or video file to benchmark.",
)
@click.option(
    "--resize-width",
    type=click.IntRange(
        min=1,
    ),
    default=512,
    show_default=True,
)
@click.option(
    "--resize-height",
    type=click.IntRange(
        min=1,
    ),
    default=512,
    show_default=True,
)
@click.option(
    "--frame-limit",
    type=click.IntRange(
        min=1,
    ),
    default=None,
    help="Maximum number of source frames to benchmark before repeat expansion.",
)
@click.option(
    "--stride",
    type=click.IntRange(
        min=1,
    ),
    default=1,
    show_default=True,
)
@click.option(
    "--repeat",
    type=click.IntRange(
        min=1,
    ),
    default=1,
    show_default=True,
)
@click.option(
    "--warmup",
    type=click.IntRange(
        min=0,
    ),
    default=10,
    show_default=True,
)
@click.option(
    "--progress-every",
    type=click.IntRange(
        min=0,
    ),
    default=50,
    show_default=True,
)
@click.option(
    "--triton-url",
    required=True,
    help="Triton HTTP URL without scheme, for example localhost:8000.",
)
@click.option(
    "--triton-model-name",
    required=True,
)
@click.option(
    "--triton-model-version",
    default="",
    help="Optional Triton model version. Empty string uses Triton's default.",
)
@click.option(
    "--triton-input-name",
    default=None,
    help="Triton input tensor name. Required only when metadata is ambiguous.",
)
@click.option(
    "--triton-output-name",
    multiple=True,
    default=(),
    help="Triton output name to request. Repeat to request multiple outputs.",
)
@click.option(
    "--triton-timeout",
    type=float,
    default=60.0,
    show_default=True,
    help="Triton request timeout in seconds.",
)
@click.option(
    "--model-package-dir",
    required=True,
    help="RF-DETR TRT package directory used to replay Triton preprocessing.",
)
@click.option(
    "--result-out",
    default=None,
    help="Optional JSON path for benchmark results.",
)
def _main(
    source: str,
    resize_width: int,
    resize_height: int,
    frame_limit: Optional[int],
    stride: int,
    repeat: int,
    warmup: int,
    progress_every: int,
    triton_url: str,
    triton_model_name: str,
    triton_model_version: str,
    triton_input_name: Optional[str],
    triton_output_name: tuple[str, ...],
    triton_timeout: float,
    model_package_dir: str,
    result_out: Optional[str],
) -> None:
    triton_output_names = list(triton_output_name) or None

    frames = _load_frames(
        source=source,
        resize_wh=(resize_width, resize_height),
        frame_limit=frame_limit,
        stride=stride,
        repeat=repeat,
    )
    print(
        f"[benchmark] loaded frames={len(frames)} "
        f"resize={resize_width}x{resize_height}",
        flush=True,
    )

    runner = _TritonRunner(
        url=triton_url,
        model_name=triton_model_name,
        model_version=triton_model_version,
        input_name=triton_input_name,
        output_names=triton_output_names,
        model_package_dir=model_package_dir,
        timeout=triton_timeout,
    )
    result = _run_profile(
        profile="triton",
        runner=runner,
        metadata=runner.metadata,
        frames=frames,
        warmup=warmup,
        progress_every=progress_every,
    )

    if result_out is not None:
        payload = {
            "mode": "run",
            "server": "triton",
            "source": source,
            "resize": {
                "width": resize_width,
                "height": resize_height,
            },
            "result": result.as_dict(),
        }
        _write_json(result_out, payload)
        print(f"[benchmark] wrote result: {result_out}", flush=True)


if __name__ == "__main__":
    _main()
