# pyright: reportMissingImports=false
"""Benchmark RF-DETR TRT latency through the experimental Inference Server.

This client mirrors the frame loading and percentile reporting used by the
existing RF-DETR TRT comparison scripts, but targets the new inference_server
API. It supports both the structured v2 JSON endpoint and the lower-overhead raw
image endpoint.

Examples:

    python development/benchmark_scripts/triton_comparison/rfdetr-trt-new-inference-server-percentiles.py \\
        --source /input/video.mp4 \\
        --model-id /models/rfdetr-trt-package \\
        --server-url http://localhost:8000 \\
        --endpoint-mode v2-json \\
        --api-key "$ROBOFLOW_API_KEY" \\
        --result-out /output/new-inference-server.json

    python development/benchmark_scripts/triton_comparison/rfdetr-trt-new-inference-server-percentiles.py \\
        --source /input/video.mp4 \\
        --model-id /models/rfdetr-trt-package \\
        --server-url http://localhost:8000 \\
        --endpoint-mode raw \\
        --api-key "$ROBOFLOW_API_KEY" \\
        --result-out /output/new-inference-server-raw.json
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


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
_SERVER_PROFILE_REQUEST_HEADER = "X-Inference-Profile"
_SERVER_STAGE_TIMINGS_HEADER = "X-Inference-Stage-Timings"


@dataclass
class _TimedCall:
    total_latency_ms: float
    request_latency_ms: float
    client_prepare_ms: float
    server_stage_timings_ms: dict[str, float]


@dataclass
class _BenchmarkResult:
    profile: str
    frames: int
    elapsed: float
    aggregate_fps: float
    total_latency_ms: dict[str, Optional[float]]
    request_latency_ms: dict[str, Optional[float]]
    client_prepare_ms: dict[str, Optional[float]]
    server_stage_timing_ms: dict[str, dict[str, Optional[float]]]
    total_latency_reads_ms: list[float]
    request_latency_reads_ms: list[float]
    client_prepare_reads_ms: list[float]
    server_stage_timing_reads_ms: dict[str, list[float]]
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
            "server_stage_timing_ms": self.server_stage_timing_ms,
            "total_latency_reads_ms": self.total_latency_reads_ms,
            "request_latency_reads_ms": self.request_latency_reads_ms,
            "client_prepare_reads_ms": self.client_prepare_reads_ms,
            "server_stage_timing_reads_ms": self.server_stage_timing_reads_ms,
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


def _encode_image(
    image_rgb: np.ndarray,
    *,
    image_format: str,
    jpeg_quality: int,
) -> tuple[bytes, str]:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    extension = ".png" if image_format == "png" else ".jpg"
    media_type = "image/png" if image_format == "png" else "image/jpeg"
    params = []
    if image_format == "jpeg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    ok, encoded = cv2.imencode(extension, image_bgr, params)
    if not ok:
        raise ValueError("Could not encode image for inference payload.")

    encoded_image = encoded.tobytes()

    return encoded_image, media_type


def _load_json_object(value: Optional[str]) -> dict[str, Any]:
    if value is None:
        return {}

    path = Path(value)
    if path.exists():
        loaded_from_file = json.loads(path.read_text())

        return loaded_from_file

    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("Expected JSON object.")

    return loaded


def _parse_key_value_pairs(values: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, got: {value}")

        key, parsed_value = value.split("=", 1)
        if not key:
            raise ValueError(f"Expected non-empty key in: {value}")

        parsed[key] = parsed_value

    return parsed


def _load_server_stage_timings(response: requests.Response) -> dict[str, float]:
    header_value = response.headers.get(_SERVER_STAGE_TIMINGS_HEADER)
    if not header_value:
        return {}

    try:
        decoded = json.loads(header_value)
    except json.JSONDecodeError:
        return {}

    if not isinstance(decoded, dict):
        return {}

    stage_timings = {
        str(stage): float(duration_ms)
        for stage, duration_ms in decoded.items()
        if isinstance(duration_ms, (int, float))
    }

    return stage_timings


class _ExperimentalInferenceServerRunner:
    def __init__(
        self,
        *,
        base_url: str,
        endpoint_mode: str,
        model_id: str,
        api_key: Optional[str],
        confidence: float,
        timeout: float,
        image_format: str,
        jpeg_quality: int,
        action: Optional[str],
        task: Optional[str],
        model_package_id: Optional[str],
        response_style: str,
        query_params: dict[str, str],
        input_extra: dict[str, Any],
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._endpoint_mode = endpoint_mode
        self._model_id = model_id
        self._api_key = api_key
        self._confidence = confidence
        self._timeout = timeout
        self._image_format = image_format
        self._jpeg_quality = jpeg_quality
        self._action = action
        self._task = task
        self._model_package_id = model_package_id
        self._response_style = response_style
        self._query_params = query_params
        self._input_extra = input_extra
        self._session = requests.Session()

    def __call__(self, image_rgb: np.ndarray) -> _TimedCall:
        start = perf_counter()
        encoded_image, media_type = _encode_image(
            image_rgb,
            image_format=self._image_format,
            jpeg_quality=self._jpeg_quality,
        )
        if self._endpoint_mode == "v2-json":
            request_kwargs = self._prepare_v2_json_request(
                encoded_image=encoded_image,
            )
        else:
            request_kwargs = self._prepare_raw_request(
                encoded_image=encoded_image,
                media_type=media_type,
            )

        prepared = perf_counter()
        response = self._session.post(
            timeout=self._timeout,
            **request_kwargs,
        )
        response.raise_for_status()
        response.json()
        finished = perf_counter()

        timed_call = _TimedCall(
            total_latency_ms=(finished - start) * 1000.0,
            request_latency_ms=(finished - prepared) * 1000.0,
            client_prepare_ms=(prepared - start) * 1000.0,
            server_stage_timings_ms=_load_server_stage_timings(response=response),
        )

        return timed_call

    def _prepare_v2_json_request(self, *, encoded_image: bytes) -> dict[str, Any]:
        url = self._base_url + "/v2/models/infer"
        params = {
            "model_id": self._model_id,
            "response_style": self._response_style,
            **self._query_params,
        }
        if self._action is not None:
            params["action"] = self._action
        if self._model_package_id is not None:
            params["model_package_id"] = self._model_package_id

        inputs: dict[str, Any] = {
            "image": {
                "type": "base64",
                "value": base64.b64encode(encoded_image).decode("ascii"),
            },
            "confidence": self._confidence,
            **self._input_extra,
        }
        headers = self._headers(content_type="application/json")
        request_kwargs = {
            "url": url,
            "params": params,
            "json": {"inputs": inputs},
            "headers": headers,
        }

        return request_kwargs

    def _prepare_raw_request(
        self,
        *,
        encoded_image: bytes,
        media_type: str,
    ) -> dict[str, Any]:
        url = self._base_url + "/infer"
        params = {
            "model_id": self._model_id,
            "format": "json",
            **self._query_params,
        }
        if self._task is not None:
            params["task"] = self._task
        if self._confidence is not None:
            params["confidence"] = str(self._confidence)

        headers = self._headers(content_type=media_type)
        request_kwargs = {
            "url": url,
            "params": params,
            "data": encoded_image,
            "headers": headers,
        }

        return request_kwargs

    def _headers(self, *, content_type: str) -> dict[str, str]:
        headers = {
            "Content-Type": content_type,
            _SERVER_PROFILE_REQUEST_HEADER: "1",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        return headers

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = {
            "base_url": self._base_url,
            "endpoint_mode": self._endpoint_mode,
            "model_id": self._model_id,
            "model_package_id": self._model_package_id,
            "action": self._action,
            "task": self._task,
            "response_style": self._response_style,
            "image_format": self._image_format,
            "query_params": self._query_params,
            "input_extra": self._input_extra,
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
    server_stage_latencies: dict[str, list[float]] = {}
    started = perf_counter()
    for index, frame in enumerate(frames, start=1):
        timed_call = runner(frame)
        total_latencies.append(timed_call.total_latency_ms)
        request_latencies.append(timed_call.request_latency_ms)
        prepare_latencies.append(timed_call.client_prepare_ms)
        for stage, duration_ms in timed_call.server_stage_timings_ms.items():
            server_stage_latencies.setdefault(stage, []).append(duration_ms)
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
        server_stage_timing_ms={
            stage: _percentiles(stage_latencies)
            for stage, stage_latencies in sorted(server_stage_latencies.items())
        },
        total_latency_reads_ms=total_latencies,
        request_latency_reads_ms=request_latencies,
        client_prepare_reads_ms=prepare_latencies,
        server_stage_timing_reads_ms=server_stage_latencies,
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
    for stage, percentiles in result.server_stage_timing_ms.items():
        print(
            f"[benchmark] server_stage={stage} "
            f"{_format_percentiles(percentiles)}",
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
    "--server-url",
    default="http://localhost:8000",
    show_default=True,
)
@click.option(
    "--endpoint-mode",
    type=click.Choice(
        ("v2-json", "raw"),
    ),
    default="v2-json",
    show_default=True,
)
@click.option(
    "--model-id",
    required=True,
)
@click.option(
    "--model-package-id",
    default=None,
    help="Optional model_package_id query parameter for the v2 endpoint.",
)
@click.option(
    "--action",
    default=None,
    help="Optional v2 action query parameter. Empty lets the server use the model default.",
)
@click.option(
    "--task",
    default=None,
    help="Optional task query parameter for the raw /infer endpoint.",
)
@click.option(
    "--response-style",
    type=click.Choice(
        ("compact", "rich"),
    ),
    default="compact",
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
    help="Optional API key sent as Authorization: Bearer <api-key>.",
)
@click.option(
    "--timeout",
    type=float,
    default=60.0,
    show_default=True,
)
@click.option(
    "--query-param",
    multiple=True,
    default=(),
    help="Extra query parameter as KEY=VALUE. Can be repeated.",
)
@click.option(
    "--input-extra-json",
    default=None,
    help="JSON object or path merged into v2-json inputs.",
)
@click.option(
    "--image-format",
    type=click.Choice(
        ("jpeg", "png"),
    ),
    default="jpeg",
    show_default=True,
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
    server_url: str,
    endpoint_mode: str,
    model_id: str,
    model_package_id: Optional[str],
    action: Optional[str],
    task: Optional[str],
    response_style: str,
    confidence: float,
    api_key: Optional[str],
    timeout: float,
    query_param: tuple[str, ...],
    input_extra_json: Optional[str],
    image_format: str,
    jpeg_quality: int,
    result_out: Optional[str],
) -> None:
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

    runner = _ExperimentalInferenceServerRunner(
        base_url=server_url,
        endpoint_mode=endpoint_mode,
        model_id=model_id,
        api_key=api_key,
        confidence=confidence,
        timeout=timeout,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        action=action,
        task=task,
        model_package_id=model_package_id,
        response_style=response_style,
        query_params=_parse_key_value_pairs(query_param),
        input_extra=_load_json_object(input_extra_json),
    )
    result = _run_profile(
        profile=f"experimental_inference_server_{endpoint_mode}",
        runner=runner,
        metadata=runner.metadata,
        frames=frames,
        warmup=warmup,
        progress_every=progress_every,
    )

    if result_out is not None:
        payload = {
            "mode": "run",
            "server": "experimental_inference_server",
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
