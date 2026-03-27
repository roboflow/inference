import dataclasses
import gc
import io
import json
import os
import shutil
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import click
import numpy as np
import requests
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision.io import ImageReadMode, decode_image

# Enable running this script directly from source checkout.
REPO_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_PACKAGE_ROOT))

from inference_models import AutoModel
from inference_models.logger import LOGGER

# Load .env from current working directory ancestry when available.
load_dotenv(find_dotenv(usecwd=True), override=False)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

def run_flakiness_check(
    models_by_workspace: Mapping[str, Mapping[str, List[str]]],
    iterations: int,
    cache_dir: Path,
    api_key: str,
    float_precision: int,
    streams_output_dir: Path,
    sample_different_images_limit: int,
    num_test_images: int,
    roboflow_images_cache_dir: Path,
    use_roboflow_image_cache: bool,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "iterations": iterations,
        "cache_dir": str(cache_dir),
        "results": defaultdict(lambda: defaultdict(dict)),
        "project_image_counts": defaultdict(dict),
    }

    for workspace, models_by_project in models_by_workspace.items():
        LOGGER.info("")
        LOGGER.info("=== Workspace: %s ===", workspace)
        for project, model_ids in models_by_project.items():
            LOGGER.info("")
            LOGGER.info("=== Project: %s ===", project)
            image_paths = fetch_roboflow_test_images(
                workspace=workspace,
                project=project,
                num_images=num_test_images,
                api_key=api_key,
                cache_dir=roboflow_images_cache_dir,
                use_cache=use_roboflow_image_cache,
            )
            report["project_image_counts"][workspace][project] = len(image_paths)
            LOGGER.info(
                "Using %s test image file(s) for workspace=%s project=%s.",
                len(image_paths),
                workspace,
                project,
            )

            for model_id in model_ids:
                LOGGER.info("Running model: %s", model_id)
                accumulated_outputs: List[List[Any]] = []
                accumulated_load_streams: List[str] = []
                status = "stable"
                mismatch_iterations: List[int] = []
                iteration_diffs: Dict[str, Any] = {}
                mismatch_stream_logs: List[str] = []

                for iteration in range(1, iterations + 1):
                    LOGGER.info(
                        "  Iteration %s/%s: clearing cache and reloading",
                        iteration,
                        iterations,
                    )
                    clear_cache(cache_dir)
                    run_outputs, load_stream = run_model_once(
                        model_id=model_id,
                        image_paths=image_paths,
                        api_key=api_key,
                        float_precision=float_precision,
                    )
                    accumulated_outputs.append(run_outputs)
                    accumulated_load_streams.append(load_stream)

                    if iteration > 1 and run_outputs != accumulated_outputs[0]:
                        status = "flaky"
                        mismatch_iterations.append(iteration)
                        iteration_diffs[str(iteration)] = compute_diff_summary(
                            baseline=accumulated_outputs[0],
                            candidate=run_outputs,
                            image_paths=image_paths,
                            sample_different_images_limit=sample_different_images_limit,
                        )
                        stream_log_path = save_mismatch_streams(
                            workspace=workspace,
                            project=project,
                            model_id=model_id,
                            baseline_iteration=1,
                            candidate_iteration=iteration,
                            baseline_stream=accumulated_load_streams[0],
                            candidate_stream=load_stream,
                            streams_output_dir=streams_output_dir,
                        )
                        mismatch_stream_logs.append(str(stream_log_path))
                        LOGGER.info(
                            "  Saved mismatch load streams to: %s", stream_log_path
                        )

                LOGGER.info(
                    "  Result: %s (mismatched iterations: %s)",
                    status.upper(),
                    mismatch_iterations or "none",
                )
                if status == "stable":
                    # Drop captured load logs when model is stable.
                    accumulated_load_streams = []
                report["results"][workspace][project][model_id] = {
                    "status": status,
                    "mismatch_iterations": mismatch_iterations,
                    "iteration_diffs": iteration_diffs,
                    "mismatch_stream_logs": mismatch_stream_logs,
                    "accumulated_outputs": accumulated_outputs,
                }

    report["results"] = {
        ws: dict(projects) for ws, projects in report["results"].items()
    }
    report["project_image_counts"] = {
        ws: dict(counts) for ws, counts in report["project_image_counts"].items()
    }
    report["summary"] = summarize_report(report["results"])
    return report


def clear_cache(cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def run_model_once(
    model_id: str, image_paths: List[Path], api_key: Optional[str], float_precision: int
) -> tuple[List[Any], str]:
    model, load_stream = load_model(model_id=model_id, api_key=api_key)
    normalized_outputs: List[Any] = []
    for image_path in image_paths:
        image = load_image(image_path)
        predictions = model(image)
        normalized_outputs.append(
            normalize_output(predictions, float_precision=float_precision)
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return normalized_outputs, load_stream


def load_image(image_path: Path) -> torch.Tensor:
    try:
        return decode_image(
            image_path,
            mode=ImageReadMode.RGB,
            apply_exif_orientation=True,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to decode image with torchvision: {image_path}") from error


def load_model(model_id: str, api_key: Optional[str]) -> tuple[Any, str]:
    stream_buffer = io.StringIO()
    with redirect_stdout(stream_buffer):
        if api_key:
            model = AutoModel.from_pretrained(model_id, api_key=api_key, verbose=True)
        else:
            model = AutoModel.from_pretrained(model_id, verbose=True)
    return model, stream_buffer.getvalue()


def normalize_output(value: Any, float_precision: int) -> Any:
    if dataclasses.is_dataclass(value):
        return normalize_output(dataclasses.asdict(value), float_precision=float_precision)
    # Namedtuple support (for metadata like PreProcessingMetadata / ImageDimensions).
    if hasattr(value, "_asdict") and callable(value._asdict):
        return normalize_output(value._asdict(), float_precision=float_precision)
    if isinstance(value, Enum):
        return normalize_output(value.value, float_precision=float_precision)
    if isinstance(value, torch.Tensor):
        return normalize_output(value.detach().cpu().numpy(), float_precision=float_precision)
    if isinstance(value, np.ndarray):
        return normalize_output(value.tolist(), float_precision=float_precision)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return round(float(value), float_precision)
    if isinstance(value, float):
        return round(value, float_precision)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.generic):
        return normalize_output(value.item(), float_precision=float_precision)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): normalize_output(sub_value, float_precision=float_precision)
            for key, sub_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_output(sub_value, float_precision=float_precision) for sub_value in value]
    return value


def compute_diff_summary(
    baseline: List[Any],
    candidate: List[Any],
    image_paths: Iterable[Path],
    sample_different_images_limit: int,
) -> Dict[str, Any]:
    mismatched_images: List[str] = []
    baseline_by_path = dict(zip([str(p) for p in image_paths], baseline))
    candidate_by_path = dict(zip([str(p) for p in image_paths], candidate))

    for image_path in baseline_by_path.keys():
        if baseline_by_path[image_path] != candidate_by_path[image_path]:
            mismatched_images.append(image_path)

    return {
        "num_images_compared": len(baseline),
        "num_images_with_differences": len(mismatched_images),
        "sample_different_images": mismatched_images[:sample_different_images_limit],
    }


def save_mismatch_streams(
    workspace: str,
    project: str,
    model_id: str,
    baseline_iteration: int,
    candidate_iteration: int,
    baseline_stream: str,
    candidate_stream: str,
    streams_output_dir: Path,
) -> Path:
    streams_output_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"{slugify(workspace)}__{slugify(project)}__{slugify(model_id)}"
        f"__iter-{baseline_iteration}-vs-{candidate_iteration}.log"
    )
    output_path = streams_output_dir / file_name
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"workspace: {workspace}\n")
        f.write(f"project: {project}\n")
        f.write(f"model_id: {model_id}\n")
        f.write(f"baseline_iteration: {baseline_iteration}\n")
        f.write(f"candidate_iteration: {candidate_iteration}\n\n")
        f.write("=== BASELINE LOAD STREAM ===\n")
        f.write(baseline_stream)
        f.write("\n\n=== CANDIDATE LOAD STREAM ===\n")
        f.write(candidate_stream)
        f.write("\n")
    return output_path


def slugify(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    slug = "".join(ch if ch in allowed else "-" for ch in value)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "value"


def summarize_report(
    results: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    total_models = 0
    flaky_models = 0
    flaky_by_workspace: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for workspace, projects in results.items():
        for project, models in projects.items():
            for model_id, model_report in models.items():
                total_models += 1
                if model_report["status"] == "flaky":
                    flaky_models += 1
                    flaky_by_workspace[workspace][project].append(model_id)

    return {
        "total_models": total_models,
        "flaky_models": flaky_models,
        "stable_models": total_models - flaky_models,
        "flaky_by_workspace": {
            ws: dict(projects) for ws, projects in flaky_by_workspace.items()
        },
    }


ROBOFLOW_SEARCH_PAGE_SIZE = 25


def fetch_roboflow_test_images(
    *,
    workspace: str,
    project: str,
    num_images: int,
    api_key: str,
    cache_dir: Path,
    use_cache: bool,
) -> List[Path]:
    """Search test split images via Roboflow API and download into cache_dir."""
    cache_root = cache_dir / slugify(workspace) / slugify(project)
    cache_root.mkdir(parents=True, exist_ok=True)

    collected: List[Dict[str, Any]] = []
    offset = 0
    url = f"https://api.roboflow.com/{workspace}/{project}/search"

    while len(collected) < num_images:
        need = num_images - len(collected)
        limit = min(ROBOFLOW_SEARCH_PAGE_SIZE, need)
        payload: Dict[str, Any] = {
            "split": "test",
            "limit": limit,
            "offset": offset,
        }
        response = requests.post(
            url,
            params={"api_key": api_key},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        batch = data.get("images") or []
        if not batch:
            LOGGER.warning(
                "Roboflow search returned no more images at offset %s "
                "(have %s, need %s).",
                offset,
                len(collected),
                num_images,
            )
            break
        take = batch[:need]
        collected.extend(take)
        offset += len(batch)
        if len(batch) < limit:
            break

    collected = collected[:num_images]

    if len(collected) < num_images:
        LOGGER.warning(
            "Only %s test images available from API (requested %s).",
            len(collected),
            num_images,
        )

    if not collected:
        raise RuntimeError(
            f"No test images returned for workspace={workspace!r} project={project!r}."
        )

    paths: List[Path] = []
    for image in collected[:num_images]:
        image_id = image.get("id")
        name = image.get("name") or "image"
        original_url = image.get("original_url")
        if not original_url:
            LOGGER.warning("Skipping image without original_url: %s", image)
            continue

        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            ext = ".jpg"
        safe_name = slugify(Path(name).stem) or "image"
        file_stem = f"{image_id}_{safe_name}" if image_id is not None else safe_name
        dest = cache_root / f"{file_stem}{ext}"

        if use_cache and dest.exists():
            paths.append(dest)
            continue

        img_response = requests.get(original_url, timeout=120)
        img_response.raise_for_status()
        dest.write_bytes(img_response.content)
        paths.append(dest)

    if not paths:
        raise RuntimeError("No images could be downloaded or found in cache.")
    return paths


def load_models_config(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load JSON: workspace slug -> project slug -> list of model_id strings."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("models-config must be a JSON object.")
    if not payload:
        raise ValueError("models-config must contain at least one workspace entry.")
    normalized: Dict[str, Dict[str, List[str]]] = {}
    for workspace, projects in payload.items():
        if not isinstance(workspace, str) or not workspace.strip():
            raise ValueError(
                "Each top-level key must be a non-empty Roboflow workspace slug."
            )
        if not isinstance(projects, dict) or not projects:
            raise ValueError(
                f"Workspace '{workspace}' must map to a non-empty object of projects."
            )
        ws_key = workspace.strip()
        normalized[ws_key] = {}
        for project, model_ids in projects.items():
            if not isinstance(project, str) or not project.strip():
                raise ValueError(
                    f"Workspace '{workspace}': each project key must be a non-empty slug."
                )
            if not isinstance(model_ids, list) or not all(
                isinstance(model_id, str) and model_id for model_id in model_ids
            ):
                raise ValueError(
                    f"Workspace '{workspace}', project '{project}' must map to a "
                    "non-empty list of model_id strings."
                )
            normalized[ws_key][project.strip()] = model_ids
    return normalized


@click.command(
    help=(
        "Check prediction flakiness by repeatedly reloading models and "
        "comparing outputs across runs."
    )
)
@click.option(
    "--models-config",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    help=(
        "Path to JSON: workspace slug -> project slug -> list[model_id]. "
        "Test images are fetched per workspace/project from the API."
    ),
)
@click.option(
    "--iterations",
    type=int,
    default=3,
    show_default=True,
    help="How many times each model should be reloaded and re-run.",
)
@click.option(
    "--inference-home",
    type=click.Path(path_type=Path),
    default=lambda: Path(os.getenv("INFERENCE_HOME", "/tmp/cache")),
    show_default="INFERENCE_HOME or /tmp/cache",
    help="Model cache path to purge before each iteration.",
)
@click.option(
    "--api-key",
    type=str,
    default=lambda: os.getenv("ROBOFLOW_API_KEY"),
    show_default="ROBOFLOW_API_KEY",
    help="Optional Roboflow API key.",
)
@click.option(
    "--float-precision",
    type=int,
    default=6,
    show_default=True,
    help="Decimal places used when normalizing float outputs for comparison.",
)
@click.option(
    "--report-json",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output path for a JSON report.",
)
@click.option(
    "--streams-output-dir",
    type=click.Path(path_type=Path),
    default=Path("flakiness_load_streams"),
    show_default=True,
    help="Directory where mismatch model-load streams are saved.",
)
@click.option(
    "--sample-different-images-limit",
    type=int,
    default=10,
    show_default=True,
    help="How many different-image paths to keep in sample_different_images.",
)
@click.option(
    "--num-test-images",
    type=int,
    required=True,
    help="How many test-split images to fetch per project from the Roboflow search API.",
)
@click.option(
    "--roboflow-images-cache-dir",
    type=click.Path(path_type=Path),
    default=Path("flakiness_roboflow_images_cache"),
    show_default=True,
    help="Directory to cache downloaded Roboflow test images (reused across runs).",
)
@click.option(
    "--no-roboflow-image-cache",
    is_flag=True,
    default=False,
    help="Re-download Roboflow images instead of reusing files under --roboflow-images-cache-dir.",
)
def main(
    models_config: Path,
    iterations: int,
    inference_home: Path,
    api_key: Optional[str],
    float_precision: int,
    report_json: Optional[Path],
    streams_output_dir: Path,
    sample_different_images_limit: int,
    num_test_images: int,
    roboflow_images_cache_dir: Path,
    no_roboflow_image_cache: bool,
) -> None:
    if iterations < 2:
        raise click.BadParameter("iterations must be >= 2 to detect differences.")
    if sample_different_images_limit < 0:
        raise click.BadParameter("sample-different-images-limit must be >= 0.")
    if num_test_images < 1:
        raise click.BadParameter("--num-test-images must be >= 1.")
    if not api_key:
        raise click.BadParameter(
            "ROBOFLOW_API_KEY (or --api-key) is required for image search and model access."
        )

    models_by_workspace = load_models_config(models_config)

    report = run_flakiness_check(
        models_by_workspace=models_by_workspace,
        iterations=iterations,
        cache_dir=inference_home,
        api_key=api_key,
        float_precision=float_precision,
        streams_output_dir=streams_output_dir,
        sample_different_images_limit=sample_different_images_limit,
        num_test_images=num_test_images,
        roboflow_images_cache_dir=roboflow_images_cache_dir,
        use_roboflow_image_cache=not no_roboflow_image_cache,
    )

    LOGGER.info("")
    LOGGER.info("=== Summary ===")
    LOGGER.info("%s", json.dumps(report["summary"], indent=2))

    if report_json:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        with report_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        LOGGER.info("Report written to: %s", report_json)


if __name__ == "__main__":
    main()
