"""Profile one Cosmos 3 Edge action recognition call stage by stage.

Developer-only. Loads an action recognition fine-tune the way the server and
the workflow block load it, runs ``infer`` over one window of frames, and
splits the wall time into processor, vision tower, LLM prefill and decode.
The split says which stage a sliding-window optimisation would save: a
per-frame vision feature cache only removes the vision tower share, while
decode is paid in full on every call.

Needs a CUDA GPU, ``inference_models`` with ``transformers>=5.15`` and the
model weights reachable (Roboflow model id plus API key, or a local package
directory).

Example:

    ROBOFLOW_API_KEY=... python development/benchmark_scripts/profile_cosmos3_action_recognition_call.py \\
        --model-id my-workspace/my-action-model/1 \\
        --video path/to/clip.mp4 \\
        --window-seconds 16,8,4 \\
        --sample-fps 4,2
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import cv2
import numpy as np
import torch

from inference_models import AutoModel
from inference_models.models.base.action_recognition import (
    effective_max_frame_side,
)
from inference_models.models.cosmos3.cosmos3_action_recognition import (
    Cosmos3EdgeActionRecognition,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner


@dataclass
class CallProfile:
    """Timings of one ``infer`` call and of its stages, in seconds.

    ``prefill_total_s`` is one forward pass over the prompt, vision tower
    included. ``prefill_llm_s`` subtracts the vision tower from it.
    ``decode_s`` subtracts the prefill from ``generate``, so it is the share
    the autoregressive loop costs. Stages that could not be measured are
    ``None`` and ``stage_errors`` says why.
    """

    window_seconds: float
    sample_fps: float
    frames: int
    frame_height: int
    frame_width: int
    prompt_tokens: Optional[int]
    visual_tokens: Optional[int]
    generated_tokens: Optional[int]
    processor_s: Optional[float]
    vision_tower_s: Optional[float]
    prefill_total_s: Optional[float]
    prefill_llm_s: Optional[float]
    generate_s: Optional[float]
    decode_s: Optional[float]
    decode_tokens_per_s: Optional[float]
    end_to_end_s: float
    peak_memory_mb: Optional[float]
    segments: List[Dict[str, Any]]
    stage_errors: Dict[str, str]


def _parse_float_list(value: str) -> List[float]:
    parsed_values = [float(item) for item in value.split(",") if item.strip()]
    if not parsed_values:
        raise click.BadParameter("Expected a comma-separated list of numbers.")

    return parsed_values


def _load_model(
    model_id: str, *, api_key: Optional[str]
) -> Cosmos3EdgeActionRecognition:
    # Mirrors inference.core.models.inference_models_adapters.load_action_recognition_model
    # without importing the server package.
    loaded_model = AutoModel.from_pretrained(model_id_or_path=model_id, api_key=api_key)
    if isinstance(loaded_model, Cosmos3EdgeActionRecognition):
        return loaded_model
    if isinstance(loaded_model, Cosmos3EdgeReasoner):
        wrapped_model = Cosmos3EdgeActionRecognition.from_reasoner(
            reasoner=loaded_model
        )
        return wrapped_model

    raise click.ClickException(
        f"Model {model_id} loaded as {type(loaded_model).__name__}, which does "
        f"not serve action recognition."
    )


def _resize_longest_side(frame: np.ndarray, *, max_side: Optional[int]) -> np.ndarray:
    if not max_side or max_side <= 0:
        return frame

    height, width = frame.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return frame

    resized_frame = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )

    return resized_frame


def _read_video_frames(
    video_path: Path,
    *,
    count: int,
    sample_fps: float,
    max_side: Optional[int],
) -> List[np.ndarray]:
    """Pick ``count`` RGB frames from the clip at ``sample_fps``, looping a short clip."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open video {video_path}.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    decoded_frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_frames.append(frame)
    capture.release()
    if not decoded_frames:
        raise click.ClickException(f"Video {video_path} has no decodable frames.")

    step = max(1.0, source_fps / sample_fps)
    frames = []
    cursor = 0.0
    while len(frames) < count:
        frame = decoded_frames[int(cursor) % len(decoded_frames)]
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        frames.append(_resize_longest_side(rgb_frame, max_side=max_side))
        cursor += step

    return frames


def _synthetic_frames(
    count: int,
    *,
    height: int,
    width: int,
    seed: int,
) -> List[np.ndarray]:
    """Build frames with a moving block over noise, so content changes over time."""
    generator = np.random.default_rng(seed)
    background = generator.integers(0, 255, size=(height, width, 3), dtype=np.uint8)

    frames = []
    block = max(8, min(height, width) // 6)
    for index in range(count):
        frame = background.copy()
        x = int((index / max(1, count - 1)) * (width - block))
        y = (height - block) // 2
        frame[y : y + block, x : x + block] = (255, 64, 64)
        frames.append(frame)

    return frames


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _count_visual_tokens(hf_model: Any, inputs: Dict[str, Any]) -> Optional[int]:
    config = getattr(hf_model, "config", None)
    video_token_id = getattr(config, "video_token_id", None)
    input_ids = inputs.get("input_ids")
    if video_token_id is not None and input_ids is not None:
        visual_tokens = int((input_ids == video_token_id).sum().item())
        return visual_tokens

    grid = inputs.get("video_grid_thw")
    vision_config = getattr(config, "vision_config", None)
    merge_size = getattr(vision_config, "spatial_merge_size", None)
    if grid is None or merge_size is None:
        return None

    visual_tokens = int(grid.prod(-1).sum().item()) // (merge_size**2)

    return visual_tokens


def _time_vision_tower(hf_model: Any, inputs: Dict[str, Any]) -> float:
    pixel_values = inputs.get("pixel_values_videos")
    grid = inputs.get("video_grid_thw")
    if pixel_values is None or grid is None:
        raise RuntimeError(
            "processor output carries no pixel_values_videos/video_grid_thw"
        )

    get_video_features = getattr(hf_model, "get_video_features", None)
    if get_video_features is None:
        raise RuntimeError("model exposes no get_video_features")

    _synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        get_video_features(pixel_values_videos=pixel_values, video_grid_thw=grid)
    _synchronize()
    elapsed = time.perf_counter() - started

    return elapsed


def _time_prefill(hf_model: Any, inputs: Dict[str, Any]) -> float:
    _synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        hf_model(**inputs, use_cache=True)
    _synchronize()
    elapsed = time.perf_counter() - started

    return elapsed


def _profile_call(
    model: Cosmos3EdgeActionRecognition,
    frames: List[np.ndarray],
    *,
    window_seconds: float,
    sample_fps: float,
    class_filter: Optional[List[str]],
) -> CallProfile:
    """Run ``infer`` once with the reasoner's stages timed from the outside.

    The processor and ``generate`` are timed by wrapping the reasoner's own
    methods for the duration of the call, so the prompt, the decoding
    constraint and the frame handling are exactly what the block pays for.
    The vision tower and the prefill are then re-run on the captured inputs.
    """
    reasoner = model._reasoner
    hf_model = reasoner._model
    captured: Dict[str, Any] = {}
    stage_errors: Dict[str, str] = {}

    original_pre_process = reasoner.pre_process_generation
    original_generate = reasoner.generate

    def timed_pre_process(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        inputs = original_pre_process(*args, **kwargs)
        _synchronize()
        captured["processor_s"] = time.perf_counter() - started
        captured["inputs"] = inputs

        return inputs

    def timed_generate(inputs: Dict[str, Any], **kwargs: Any) -> torch.Tensor:
        _synchronize()
        started = time.perf_counter()
        generated_ids = original_generate(inputs, **kwargs)
        _synchronize()
        captured["generate_s"] = time.perf_counter() - started
        captured["generated_tokens"] = int(generated_ids.shape[-1])

        return generated_ids

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    reasoner.pre_process_generation = timed_pre_process
    reasoner.generate = timed_generate
    try:
        _synchronize()
        started = time.perf_counter()
        segments = model.infer(frames=frames, class_names=class_filter, fps=sample_fps)
        _synchronize()
        end_to_end_s = time.perf_counter() - started
    finally:
        del reasoner.pre_process_generation
        del reasoner.generate
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else None
    )

    inputs = captured.get("inputs")
    prompt_tokens = None
    visual_tokens = None
    vision_tower_s = None
    prefill_total_s = None
    if inputs is None:
        stage_errors["processor"] = "pre_process_generation was not called"
    else:
        input_ids = inputs.get("input_ids")
        prompt_tokens = int(input_ids.shape[-1]) if input_ids is not None else None
        visual_tokens = _count_visual_tokens(hf_model, inputs)
        try:
            vision_tower_s = _time_vision_tower(hf_model, inputs)
        except Exception as error:
            stage_errors["vision_tower"] = f"{type(error).__name__}: {error}"
        try:
            prefill_total_s = _time_prefill(hf_model, inputs)
        except Exception as error:
            stage_errors["prefill"] = f"{type(error).__name__}: {error}"

    generate_s = captured.get("generate_s")
    generated_tokens = captured.get("generated_tokens")
    prefill_llm_s = None
    if prefill_total_s is not None and vision_tower_s is not None:
        prefill_llm_s = max(0.0, prefill_total_s - vision_tower_s)
    decode_s = None
    decode_tokens_per_s = None
    if generate_s is not None and prefill_total_s is not None:
        decode_s = max(0.0, generate_s - prefill_total_s)
        if generated_tokens and decode_s > 0:
            decode_tokens_per_s = generated_tokens / decode_s

    frame_height, frame_width = frames[0].shape[:2]
    profile = CallProfile(
        window_seconds=window_seconds,
        sample_fps=sample_fps,
        frames=len(frames),
        frame_height=frame_height,
        frame_width=frame_width,
        prompt_tokens=prompt_tokens,
        visual_tokens=visual_tokens,
        generated_tokens=generated_tokens,
        processor_s=captured.get("processor_s"),
        vision_tower_s=vision_tower_s,
        prefill_total_s=prefill_total_s,
        prefill_llm_s=prefill_llm_s,
        generate_s=generate_s,
        decode_s=decode_s,
        decode_tokens_per_s=decode_tokens_per_s,
        end_to_end_s=end_to_end_s,
        peak_memory_mb=peak_memory_mb,
        segments=[
            {
                "start_frame_idx": segment.start_frame_idx,
                "end_frame_idx": segment.end_frame_idx,
                "class_name": segment.class_name,
            }
            for segment in segments
        ],
        stage_errors=stage_errors,
    )

    return profile


def _median_profile(profiles: List[CallProfile]) -> CallProfile:
    """Median of every numeric field; segments and errors come from the last run."""
    numeric_fields = [
        "prompt_tokens",
        "visual_tokens",
        "generated_tokens",
        "processor_s",
        "vision_tower_s",
        "prefill_total_s",
        "prefill_llm_s",
        "generate_s",
        "decode_s",
        "decode_tokens_per_s",
        "end_to_end_s",
        "peak_memory_mb",
    ]
    merged = asdict(profiles[-1])
    for field_name in numeric_fields:
        values = [getattr(profile, field_name) for profile in profiles]
        present_values = [value for value in values if value is not None]
        merged[field_name] = (
            statistics.median(present_values) if present_values else None
        )

    median_profile = CallProfile(**merged)

    return median_profile


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "   n/a"

    return f"{value:6.3f}"


def _format_count(value: Optional[float]) -> str:
    if value is None:
        return "  n/a"

    return f"{int(round(value)):5d}"


def _print_table(profiles: List[CallProfile]) -> None:
    header = (
        "window  fps  frames   size      prompt  visual   proc  vision  prefill"
        "  decode  gen_tok  tok/s     e2e   peak_MB"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for profile in profiles:
        tokens_per_s = (
            f"{profile.decode_tokens_per_s:6.1f}"
            if profile.decode_tokens_per_s is not None
            else "   n/a"
        )
        peak = (
            f"{profile.peak_memory_mb:8.0f}" if profile.peak_memory_mb else "     n/a"
        )
        click.echo(
            f"{profile.window_seconds:6.1f} {profile.sample_fps:4.1f} {profile.frames:7d}"
            f" {profile.frame_width:4d}x{profile.frame_height:<4d}"
            f" {_format_count(profile.prompt_tokens)}  {_format_count(profile.visual_tokens)}"
            f" {_format_seconds(profile.processor_s)} {_format_seconds(profile.vision_tower_s)}"
            f"  {_format_seconds(profile.prefill_llm_s)}  {_format_seconds(profile.decode_s)}"
            f"  {_format_count(profile.generated_tokens)}  {tokens_per_s}"
            f"  {_format_seconds(profile.end_to_end_s)} {peak}"
        )
    click.echo("")
    click.echo(
        "proc = processor (CPU), vision = vision tower, prefill = LLM prefill without "
        "the vision tower, decode = generate minus prefill. All seconds, medians."
    )
    for profile in profiles:
        if profile.stage_errors:
            click.echo(
                f"window {profile.window_seconds}s fps {profile.sample_fps}: "
                f"stages not measured: {profile.stage_errors}"
            )


@click.command()
@click.option(
    "--model-id",
    type=str,
    required=True,
    help="Roboflow model id of a Cosmos 3 Edge action recognition fine-tune, or a local package directory.",
)
@click.option(
    "--api-key",
    type=str,
    envvar="ROBOFLOW_API_KEY",
    default=None,
    help="Roboflow API key. Read from ROBOFLOW_API_KEY when omitted.",
)
@click.option(
    "--video",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    default=None,
    help="Clip to sample frames from. Synthetic frames are used when omitted.",
)
@click.option(
    "--window-seconds",
    type=str,
    default=None,
    help="Comma-separated window lengths to profile. Defaults to the model's trained window.",
)
@click.option(
    "--sample-fps",
    type=str,
    default=None,
    help="Comma-separated sample rates to profile. Defaults to the model's trained rate.",
)
@click.option(
    "--frame-side",
    type=click.IntRange(
        min=32,
    ),
    default=640,
    show_default=True,
    help="Longest side of the source frames before the model's own cap applies.",
)
@click.option(
    "--class-filter",
    type=str,
    default=None,
    help="Comma-separated subset of the model's classes to ask for.",
)
@click.option(
    "--warmup",
    type=click.IntRange(
        min=0,
    ),
    default=1,
    show_default=True,
    help="Untimed calls per configuration before measuring.",
)
@click.option(
    "--repeats",
    type=click.IntRange(
        min=1,
    ),
    default=3,
    show_default=True,
    help="Timed calls per configuration; the table reports medians.",
)
@click.option(
    "--output",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    default=None,
    help="Write every timed call as JSON to this file.",
)
def main(
    model_id: str,
    api_key: Optional[str],
    video: Optional[Path],
    window_seconds: Optional[str],
    sample_fps: Optional[str],
    frame_side: int,
    class_filter: Optional[str],
    warmup: int,
    repeats: int,
    output: Optional[Path],
) -> None:
    """Profile Cosmos 3 Edge action recognition calls over windows of frames."""
    if not torch.cuda.is_available():
        click.echo(
            "No CUDA device; timings will not represent the served path.", err=True
        )

    model = _load_model(model_id, api_key=api_key)
    sampling = model.video_sampling
    windows = (
        _parse_float_list(window_seconds)
        if window_seconds
        else [float(sampling.window_seconds)]
    )
    rates = (
        _parse_float_list(sample_fps) if sample_fps else [float(sampling.sample_fps)]
    )
    classes = (
        [name.strip() for name in class_filter.split(",") if name.strip()]
        if class_filter
        else None
    )
    model_cap = effective_max_frame_side(sampling)

    hf_model = model._reasoner._model
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    click.echo(f"model: {model_id}")
    click.echo(f"device: {device_name}, dtype: {model._reasoner._torch_dtype}")
    click.echo(
        f"attention: {getattr(getattr(hf_model, 'config', None), '_attn_implementation', 'unknown')}"
    )
    click.echo(
        f"trained sampling: window {sampling.window_seconds}s, {sampling.sample_fps} fps, "
        f"min_frames {sampling.min_frames}, max_frames {sampling.max_frames}, "
        f"max_frame_side {sampling.max_frame_side}, mode {sampling.mode}"
    )
    click.echo(f"classes: {len(model.class_names or [])}, filter: {classes}")
    if torch.cuda.is_available():
        click.echo(
            f"memory after load: {torch.cuda.memory_allocated() / (1024**2):.0f} MB"
        )
    click.echo("")

    medians = []
    all_profiles = []
    for window in windows:
        for rate in rates:
            count = max(int(sampling.min_frames), round(window * rate))
            if video is not None:
                frames = _read_video_frames(
                    video,
                    count=count,
                    sample_fps=rate,
                    max_side=frame_side,
                )
            else:
                height = round(frame_side * 9 / 16)
                frames = _synthetic_frames(
                    count,
                    height=height,
                    width=frame_side,
                    seed=0,
                )
            # The block shrinks frames to the trained side before buffering,
            # so the model sees the same pixels it would in a workflow.
            frames = [
                _resize_longest_side(frame, max_side=model_cap) for frame in frames
            ]

            for _ in range(warmup):
                _profile_call(
                    model,
                    frames,
                    window_seconds=window,
                    sample_fps=rate,
                    class_filter=classes,
                )
            profiles = []
            for _ in range(repeats):
                profile = _profile_call(
                    model,
                    frames,
                    window_seconds=window,
                    sample_fps=rate,
                    class_filter=classes,
                )
                profiles.append(profile)
            all_profiles.extend(profiles)
            median = _median_profile(profiles)
            medians.append(median)
            click.echo(
                f"window {window}s @ {rate} fps: {median.end_to_end_s:.3f}s end to end, "
                f"answer: {median.segments}"
            )

    click.echo("")
    _print_table(medians)

    if output is not None:
        output.write_text(
            json.dumps([asdict(profile) for profile in all_profiles], indent=2)
        )
        click.echo(f"wrote {len(all_profiles)} profiles to {output}")


if __name__ == "__main__":
    main()
