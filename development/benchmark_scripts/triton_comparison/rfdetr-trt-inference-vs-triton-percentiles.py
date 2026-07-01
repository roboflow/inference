# pyright: reportMissingImports=false
"""Compare RF-DETR TRT latency through Inference Server and Triton HTTP.

The two servers usually expose different contracts:

* Inference Server accepts image payloads and runs pre-processing, TRT inference,
  post-processing, and response serialization.
* Triton normally accepts a pre-processed tensor and runs the TRT engine.

To make that difference visible, this script reports total latency plus the
client-side preparation and HTTP request portions separately. For Triton, the
pre-processing is replayed from the RF-DETR model package using inference-models.

Example:

    python development/stream_interface/rfdetr-trt-inference-vs-triton-percentiles.py \\
        --source /input/images \\
        --model-id rfdetr-small \\
        --task object_detection \\
        --model-package-dir development/rfdetr-small-trt-build/trt_package \\
        --inference-server-url http://localhost:9001 \\
        --triton-url localhost:8000 \\
        --triton-model-name rfdetr_small \\
        --result-out /output/benchmark.json
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

import click
import cv2
import numpy as np
import requests


_TASK_TO_INFERENCE_ENDPOINT = {
    "object_detection": "/infer/object_detection",
    "instance_segmentation": "/infer/instance_segmentation",
}
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


def _encode_image_base64(
    image_rgb: np.ndarray,
    *,
    image_format: str,
    jpeg_quality: int,
) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    extension = ".png" if image_format == "png" else ".jpg"
    params = []
    if image_format == "jpeg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, encoded = cv2.imencode(extension, image_bgr, params)
    if not ok:
        raise ValueError("Could not encode image for inference server payload.")
    encoded_image = base64.b64encode(encoded.tobytes()).decode("ascii")

    return encoded_image


def _load_json_object(value: Optional[str]) -> dict[str, Any]:
    if value is None:
        return {}
    path = Path(value)
    if path.exists():
        loaded_from_file = json.loads(path.read_text())

        return loaded_from_file

    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("Expected JSON object for extra payload.")

    return loaded


class _InferenceServerRunner:
    def __init__(
        self,
        *,
        base_url: str,
        task: str,
        model_id: str,
        api_key: Optional[str],
        confidence: float,
        timeout: float,
        image_format: str,
        jpeg_quality: int,
        extra_payload: dict[str, Any],
    ) -> None:
        self._url = base_url.rstrip("/") + _TASK_TO_INFERENCE_ENDPOINT[task]
        self._model_id = model_id
        self._api_key = api_key
        self._confidence = confidence
        self._timeout = timeout
        self._image_format = image_format
        self._jpeg_quality = jpeg_quality
        self._extra_payload = extra_payload
        self._session = requests.Session()
        self._task = task

    def __call__(self, image_rgb: np.ndarray) -> _TimedCall:
        start = perf_counter()
        image_base64 = _encode_image_base64(
            image_rgb,
            image_format=self._image_format,
            jpeg_quality=self._jpeg_quality,
        )
        payload: dict[str, Any] = {
            "model_id": self._model_id,
            "image": {"type": "base64", "value": image_base64},
            "confidence": self._confidence,
            "disable_active_learning": True,
            "visualize_predictions": False,
        }
        if self._api_key is not None:
            payload["api_key"] = self._api_key
        if self._task == "instance_segmentation":
            payload.setdefault("response_mask_format", "rle")
            payload.setdefault("enforce_dense_masks_in_inference_models", False)
        payload.update(self._extra_payload)
        prepared = perf_counter()
        response = self._session.post(self._url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        # Force response parsing so serialization/deserialization cost is included.
        response.json()
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
            "url": self._url,
            "model_id": self._model_id,
            "task": self._task,
            "image_format": self._image_format,
        }

        return metadata


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
            headers=None,
            query_params=None,
            request_id=None,
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
            priority=0,
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


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> float:
    if numerator is None or denominator is None or denominator <= 0:
        return 0.0

    ratio = numerator / denominator

    return ratio


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
    "--task",
    type=click.Choice(
        sorted(_TASK_TO_INFERENCE_ENDPOINT),
    ),
    default="object_detection",
    show_default=True,
)
@click.option(
    "--model-id",
    default="rfdetr-small",
    show_default=True,
)
@click.option(
    "--confidence",
    type=float,
    default=0.4,
    show_default=True,
)
@click.option(
    "--api-key",
    default=None,
    help="Optional Roboflow API key passed to Inference Server requests.",
)
@click.option(
    "--inference-server-url",
    default="http://localhost:9001",
    show_default=True,
)
@click.option(
    "--inference-timeout",
    type=float,
    default=60.0,
    show_default=True,
    help="Inference Server request timeout in seconds.",
)
@click.option(
    "--inference-extra-json",
    default=None,
    help="JSON object or path to a JSON object merged into Inference Server payloads.",
)
@click.option(
    "--image-format",
    type=click.Choice(
        ("jpeg", "png"),
    ),
    default="jpeg",
    show_default=True,
    help="Image encoding used for Inference Server base64 payloads.",
)
@click.option(
    "--jpeg-quality",
    type=click.IntRange(
        min=1,
        max=100,
    ),
    default=95,
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
    task: str,
    model_id: str,
    confidence: float,
    api_key: Optional[str],
    inference_server_url: str,
    inference_timeout: float,
    inference_extra_json: Optional[str],
    image_format: str,
    jpeg_quality: int,
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

    inference_runner = _InferenceServerRunner(
        base_url=inference_server_url,
        task=task,
        model_id=model_id,
        api_key=api_key,
        confidence=confidence,
        timeout=inference_timeout,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        extra_payload=_load_json_object(inference_extra_json),
    )
    triton_runner = _TritonRunner(
        url=triton_url,
        model_name=triton_model_name,
        model_version=triton_model_version,
        input_name=triton_input_name,
        output_names=triton_output_names,
        model_package_dir=model_package_dir,
        timeout=triton_timeout,
    )

    inference_result = _run_profile(
        profile="inference_server",
        runner=inference_runner,
        metadata=inference_runner.metadata,
        frames=frames,
        warmup=warmup,
        progress_every=progress_every,
    )
    triton_result = _run_profile(
        profile="triton",
        runner=triton_runner,
        metadata=triton_runner.metadata,
        frames=frames,
        warmup=warmup,
        progress_every=progress_every,
    )

    inference_p50 = inference_result.total_latency_ms["p_50"]
    triton_p50 = triton_result.total_latency_ms["p_50"]
    triton_request_p50 = triton_result.request_latency_ms["p_50"]
    print("\n---- compare ----", flush=True)
    print(
        f"  inference_server aggregate_fps={inference_result.aggregate_fps:.2f} "
        f"total_latency_ms {_format_percentiles(inference_result.total_latency_ms)}",
        flush=True,
    )
    print(
        f"  triton           aggregate_fps={triton_result.aggregate_fps:.2f} "
        f"total_latency_ms {_format_percentiles(triton_result.total_latency_ms)}",
        flush=True,
    )
    print(
        f"  p50_total_latency_ratio inference_server/triton="
        f"{_ratio(inference_p50, triton_p50):.2f}x",
        flush=True,
    )
    print(
        f"  p50_total_latency_ratio inference_server/triton_request_only="
        f"{_ratio(inference_p50, triton_request_p50):.2f}x",
        flush=True,
    )

    if result_out is not None:
        payload = {
            "mode": "compare",
            "source": source,
            "resize": {
                "width": resize_width,
                "height": resize_height,
            },
            "inference_server": inference_result.as_dict(),
            "triton": triton_result.as_dict(),
            "p50_total_latency_ratio_inference_server_to_triton": _ratio(
                inference_p50,
                triton_p50,
            ),
            "p50_total_latency_ratio_inference_server_to_triton_request_only": _ratio(
                inference_p50,
                triton_request_p50,
            ),
        }
        _write_json(result_out, payload)
        print(f"[benchmark] wrote compare result: {result_out}", flush=True)


if __name__ == "__main__":
    _main()
