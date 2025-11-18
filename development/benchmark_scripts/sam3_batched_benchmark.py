import argparse
import csv
import gc
import io
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image

# Local imports from the private SAM3 repo copy
from sam3.model.data_misc import BatchedDatapoint
from sam3.model_builder import build_sam3_image_model
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.train.utils.train_utils import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    Datapoint as Sam3Datapoint,
    Image as Sam3ImageDP,
    FindQueryLoaded,
    InferenceMetadata,
)
from sam3.train.data.collator import collate_fn_api

# Benchmark configuration
# Examples with image and meaningful prompts
DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000077595.jpg"
DEFAULT_PROMPTS = ["cat", "laptop", "keyboard", "blanket", "eyes", "wall"]

EXAMPLES = [
    {
        "name": "example1",
        "image_url": DEFAULT_IMAGE_URL,
        # Text-only prompts; we will benchmark using the first N (1..len)
        "prompts": [
            "cat",
            "laptop",
            "keyboard",
            "blanket",
            "eyes",
            "wall",
        ],
    },
    {
        "name": "example2",
        "image_url": "http://images.cocodataset.org/val2017/000000136466.jpg",
        # Sequence mixes text and visual prompts, matching the notebook flow; we add
        # prompts cumulatively (first 1, first 2, ...)
        "prompts": [
            {"type": "text", "text": "pot"},
            {"type": "visual", "boxes": [[59, 144, 76, 163]], "labels": [True]},
            {
                "type": "visual",
                "boxes": [[59, 144, 76, 163], [87, 148, 104, 159]],
                "labels": [True, True],
            },
            {"type": "text", "text": "handle"},
            {
                "type": "visual",
                "boxes": [[40, 183, 318, 204]],
                "labels": [False],
                "text": "handle",
            },
        ],
    },
]


@dataclass
class PromptSpec:
    num_text_prompts: int
    # Future: add boxes/points variations


def _fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _create_empty_datapoint() -> Sam3Datapoint:
    return Sam3Datapoint(find_queries=[], images=[], raw_images=None)


def _set_image(datapoint: Sam3Datapoint, pil_image: Image.Image):
    w, h = pil_image.size
    datapoint.images = [Sam3ImageDP(data=pil_image, objects=[], size=(h, w))]
    datapoint.raw_images = [pil_image]


_GLOBAL_ID = 0


def _add_text_prompt(datapoint: Sam3Datapoint, text_prompt: str) -> int:
    global _GLOBAL_ID
    h, w = datapoint.images[0].size
    q = FindQueryLoaded(
        query_text=text_prompt,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        input_bbox=None,
        input_bbox_label=None,
        input_points=None,
        semantic_target=None,
        is_pixel_exhaustive=None,
        inference_metadata=InferenceMetadata(
            coco_image_id=_GLOBAL_ID,
            original_image_id=_GLOBAL_ID,
            original_category_id=1,
            original_size=(h, w),
            object_id=0,
            frame_index=0,
        ),
    )
    datapoint.find_queries.append(q)
    _GLOBAL_ID += 1
    return len(datapoint.find_queries) - 1


def _add_visual_prompt(
    datapoint: Sam3Datapoint,
    boxes: List[List[float]],
    labels: List[bool],
    text_prompt: Optional[str] = None,
) -> int:
    """Add a visual (box) prompt with optional associated text.

    boxes are in XYXY pixel coords; labels booleans for positive/negative.
    """
    global _GLOBAL_ID
    h, w = datapoint.images[0].size
    q = FindQueryLoaded(
        query_text=text_prompt if text_prompt is not None else "visual",
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        input_bbox=torch.tensor(boxes, dtype=torch.float32),
        input_bbox_label=torch.tensor(labels, dtype=torch.bool),
        input_points=None,
        semantic_target=None,
        is_pixel_exhaustive=None,
        inference_metadata=InferenceMetadata(
            coco_image_id=_GLOBAL_ID,
            original_image_id=_GLOBAL_ID,
            original_category_id=1,
            original_size=(h, w),
            object_id=0,
            frame_index=0,
        ),
    )
    datapoint.find_queries.append(q)
    _GLOBAL_ID += 1
    return len(datapoint.find_queries) - 1


def _transform_datapoint(datapoint: Sam3Datapoint, transform) -> Sam3Datapoint:
    # API transforms operate over Datapoint with images[].data
    return transform(datapoint)


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def _measure_cuda_memory() -> Tuple[int, int, int]:
    if not torch.cuda.is_available():
        return 0, 0, 0
    device = torch.device("cuda")
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)
    return alloc, reserved, peak


def benchmark(
    batch_sizes: List[int],
    warmup_iters: int,
    measure_iters: int,
    bpe_path: str,
    checkpoint_path: str,
    output_csv: str,
    output_plot: str,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and preprocessing identical to examples
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=True,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008, max_size=1008, square=True, consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    rows: List[dict] = []

    for example in EXAMPLES:
        img = _fetch_image(example["image_url"])
        prompts = example["prompts"]
        prompt_counts = list(range(1, len(prompts) + 1))
        for batch_size in batch_sizes:
            for num_prompts in prompt_counts:
                _clear_cuda()
                gc.collect()

                print(
                    f"Running {example['name']} batch={batch_size} prompts={num_prompts}...",
                    flush=True,
                )

                batch = None
                datapoints: List[Sam3Datapoint] = []
                try:
                    # Build datapoints and transform
                    for _ in range(batch_size):
                        dp = _create_empty_datapoint()
                        _set_image(dp, img)
                        for p in prompts[:num_prompts]:
                            if isinstance(p, str):
                                _add_text_prompt(dp, p)
                            else:
                                if p.get("type") == "text":
                                    _add_text_prompt(dp, p.get("text", ""))
                                elif p.get("type") == "visual":
                                    _add_visual_prompt(
                                        dp,
                                        boxes=p.get("boxes", []),
                                        labels=p.get("labels", []),
                                        text_prompt=p.get("text"),
                                    )
                                else:
                                    _add_text_prompt(dp, str(p))
                        dp = _transform_datapoint(dp, transform)
                        datapoints.append(dp)

                    # Collate to BatchedDatapoint using provided API utility
                    batch = collate_fn_api(batch=datapoints, dict_key="dummy")["dummy"]

                    # Move to device
                    batch = copy_data_to_device(
                        batch, torch.device(device), non_blocking=True
                    )

                    # Warmup
                    with torch.inference_mode():
                        for _ in range(warmup_iters):
                            _ = model(batch)
                    _sync_cuda()

                    # Measure
                    start = time.perf_counter()
                    with torch.inference_mode():
                        for _ in range(measure_iters):
                            _ = model(batch)
                    _sync_cuda()
                    duration = time.perf_counter() - start

                    alloc, reserved, peak = _measure_cuda_memory()

                    row = {
                        "example": example["name"],
                        "batch_size": batch_size,
                        "num_text_prompts": num_prompts,
                        "warmup_iters": warmup_iters,
                        "measure_iters": measure_iters,
                        "time_s": duration,
                        "time_per_iter_ms": (duration / max(measure_iters, 1)) * 1000.0,
                        "cuda_alloc_bytes": alloc,
                        "cuda_reserved_bytes": reserved,
                        "cuda_peak_bytes": peak,
                    }
                    rows.append(row)
                    print(
                        f"OK {example['name']} batch={batch_size} prompts={num_prompts} "
                        f"time/iter={row['time_per_iter_ms']:.2f} ms "
                        f"peak_mem={peak/(1024**3):.2f} GiB",
                        flush=True,
                    )

                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "cuda" in msg:
                        alloc, reserved, peak = _measure_cuda_memory()
                        row = {
                            "example": example["name"],
                            "batch_size": batch_size,
                            "num_text_prompts": num_prompts,
                            "warmup_iters": warmup_iters,
                            "measure_iters": measure_iters,
                            "time_s": float("nan"),
                            "time_per_iter_ms": float("nan"),
                            "cuda_alloc_bytes": alloc,
                            "cuda_reserved_bytes": reserved,
                            "cuda_peak_bytes": peak,
                        }
                        rows.append(row)
                        print(
                            f"OOM {example['name']} batch={batch_size} prompts={num_prompts} "
                            f"peak_mem={peak/(1024**3):.2f} GiB; skipping",
                            flush=True,
                        )
                    else:
                        raise
                finally:
                    # Cleanup tensors explicitly
                    if batch is not None:
                        del batch
                    if datapoints:
                        del datapoints
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.reset_peak_memory_stats()

    # Write CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example",
                "batch_size",
                "num_text_prompts",
                "warmup_iters",
                "measure_iters",
                "time_s",
                "time_per_iter_ms",
                "cuda_alloc_bytes",
                "cuda_reserved_bytes",
                "cuda_peak_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Plot: per-example prompts vs peak memory for each batch size
    def _plot_path(base: str, example_name: str) -> str:
        base_dir, base_file = os.path.split(base)
        name, ext = os.path.splitext(base_file)
        if ext:
            fname = f"{name}_{example_name}{ext}"
        else:
            fname = f"sam3_benchmark_{example_name}.png"
        out_dir = base_dir if base_dir else "."
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, fname)

    def _plot_variant_path(base: str, example_name: str, variant: str) -> str:
        base_dir, base_file = os.path.split(base)
        name, ext = os.path.splitext(base_file)
        if not ext:
            ext = ".png"
        fname = f"{name}_{example_name}_{variant}{ext}"
        out_dir = base_dir if base_dir else "."
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, fname)

    for example in EXAMPLES:
        plt.figure(figsize=(8, 5))
        for b in batch_sizes:
            xs = sorted(
                {
                    r["num_text_prompts"]
                    for r in rows
                    if r.get("example") == example["name"] and r["batch_size"] == b
                }
            )
            ys = []
            for x in xs:
                vals = [
                    r["cuda_peak_bytes"]
                    for r in rows
                    if r.get("example") == example["name"]
                    and r["batch_size"] == b
                    and r["num_text_prompts"] == x
                ]
                ys.append(np.mean(vals) if len(vals) > 0 else 0)
            if xs:
                plt.plot(xs, np.array(ys) / (1024**3), marker="o", label=f"batch={b}")
        plt.xlabel("# text prompts per image")
        plt.ylabel("Peak CUDA memory (GiB)")
        plt.title(f"SAM3 Peak CUDA Memory vs #Prompts ({example['name']})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(_plot_path(output_plot, example_name=example["name"]))

    # Plot: per-example prompts vs time per iteration (ms) for each batch size
    for example in EXAMPLES:
        plt.figure(figsize=(8, 5))
        for b in batch_sizes:
            xs = sorted(
                {
                    r["num_text_prompts"]
                    for r in rows
                    if r.get("example") == example["name"] and r["batch_size"] == b
                }
            )
            ys = []
            for x in xs:
                vals = [
                    r["time_per_iter_ms"]
                    for r in rows
                    if r.get("example") == example["name"]
                    and r["batch_size"] == b
                    and r["num_text_prompts"] == x
                    and np.isfinite(r["time_per_iter_ms"])
                ]
                ys.append(np.mean(vals) if len(vals) > 0 else np.nan)
            if xs:
                plt.plot(xs, ys, marker="o", label=f"batch={b}")
        plt.xlabel("# text prompts per image")
        plt.ylabel("Time per iteration (ms)")
        plt.title(f"SAM3 Time per Iteration vs #Prompts ({example['name']})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            _plot_variant_path(
                output_plot, example_name=example["name"], variant="time"
            )
        )


def plot_from_csv(csv_path: str, output_plot: str) -> None:
    # Read rows from CSV and cast types
    rows: List[dict] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Build a list of batch sizes
    try:
        batch_sizes = sorted({int(r["batch_size"]) for r in rows})
    except Exception:
        batch_sizes = sorted({r["batch_size"] for r in rows})

    def _plot_path(base: str, example_name: str) -> str:
        base_dir, base_file = os.path.split(base)
        name, ext = os.path.splitext(base_file)
        if ext:
            fname = f"{name}_{example_name}{ext}"
        else:
            fname = f"sam3_benchmark_{example_name}.png"
        out_dir = base_dir if base_dir else "."
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, fname)

    def _plot_variant_path(base: str, example_name: str, variant: str) -> str:
        base_dir, base_file = os.path.split(base)
        name, ext = os.path.splitext(base_file)
        if not ext:
            ext = ".png"
        fname = f"{name}_{example_name}_{variant}{ext}"
        out_dir = base_dir if base_dir else "."
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, fname)

    examples = sorted({r.get("example", "example") for r in rows})

    # Memory plots
    for example_name in examples:
        plt.figure(figsize=(8, 5))
        for b in batch_sizes:
            try:
                b_cast = int(b)
            except Exception:
                b_cast = b
            xs = sorted(
                {
                    (
                        int(r["num_text_prompts"])
                        if str(r["num_text_prompts"]).isdigit()
                        else r["num_text_prompts"]
                    )
                    for r in rows
                    if r.get("example", "example") == example_name
                    and (
                        int(r["batch_size"])
                        if str(r["batch_size"]).isdigit()
                        else r["batch_size"]
                    )
                    == b_cast
                }
            )
            ys = []
            for x in xs:
                vals = []
                for r in rows:
                    if r.get("example", "example") != example_name:
                        continue
                    rb = (
                        int(r["batch_size"])
                        if str(r["batch_size"]).isdigit()
                        else r["batch_size"]
                    )
                    rn = (
                        int(r["num_text_prompts"])
                        if str(r["num_text_prompts"]).isdigit()
                        else r["num_text_prompts"]
                    )
                    if rb == b_cast and rn == x:
                        try:
                            vals.append(int(r["cuda_peak_bytes"]))
                        except Exception:
                            pass
                ys.append(np.mean(vals) if len(vals) > 0 else 0)
            if xs:
                plt.plot(xs, np.array(ys) / (1024**3), marker="o", label=f"batch={b}")
        plt.xlabel("# text prompts per image")
        plt.ylabel("Peak CUDA memory (GiB)")
        plt.title(f"SAM3 Peak CUDA Memory vs #Prompts ({example_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(_plot_path(output_plot, example_name=example_name))

    # Time plots
    for example_name in examples:
        plt.figure(figsize=(8, 5))
        for b in batch_sizes:
            try:
                b_cast = int(b)
            except Exception:
                b_cast = b
            xs = sorted(
                {
                    (
                        int(r["num_text_prompts"])
                        if str(r["num_text_prompts"]).isdigit()
                        else r["num_text_prompts"]
                    )
                    for r in rows
                    if r.get("example", "example") == example_name
                    and (
                        int(r["batch_size"])
                        if str(r["batch_size"]).isdigit()
                        else r["batch_size"]
                    )
                    == b_cast
                }
            )
            ys = []
            for x in xs:
                vals = []
                for r in rows:
                    if r.get("example", "example") != example_name:
                        continue
                    rb = (
                        int(r["batch_size"])
                        if str(r["batch_size"]).isdigit()
                        else r["batch_size"]
                    )
                    rn = (
                        int(r["num_text_prompts"])
                        if str(r["num_text_prompts"]).isdigit()
                        else r["num_text_prompts"]
                    )
                    if rb == b_cast and rn == x:
                        try:
                            v = (
                                float(r["time_per_iter_ms"])
                                if r["time_per_iter_ms"]
                                not in ("nan", "NaN", "None", "")
                                else np.nan
                            )
                        except Exception:
                            v = np.nan
                        if np.isfinite(v):
                            vals.append(v)
                ys.append(np.mean(vals) if len(vals) > 0 else np.nan)
            if xs:
                plt.plot(xs, ys, marker="o", label=f"batch={b}")
        plt.xlabel("# text prompts per image")
        plt.ylabel("Time per iteration (ms)")
        plt.title(f"SAM3 Time per Iteration vs #Prompts ({example_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            _plot_variant_path(output_plot, example_name=example_name, variant="time")
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark SAM3 batched inference memory/time"
    )
    p.add_argument("--batch_sizes", type=int, nargs="*", default=[1, 2, 4])
    p.add_argument("--warmup_iters", type=int, default=2)
    p.add_argument("--measure_iters", type=int, default=5)
    p.add_argument(
        "--bpe_path",
        type=str,
        default="/tmp/cache/sam3/sam3_final/bpe_simple_vocab_16e6.txt.gz",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="/tmp/cache/sam3/sam3_final/weights.pt",
    )
    p.add_argument(
        "--output_csv", type=str, default="benchmarks/out/sam3_benchmark.csv"
    )
    p.add_argument(
        "--output_plot", type=str, default="benchmarks/out/sam3_benchmark.png"
    )
    p.add_argument(
        "--device", type=str, default=None, choices=[None, "cpu", "cuda"], nargs="?"
    )
    p.add_argument(
        "--plot_only", action="store_true", help="Skip inference; plot from CSV"
    )
    p.add_argument(
        "--csv",
        type=str,
        default="benchmarks/out/sam3_benchmark.csv",
        help="CSV to plot in --plot_only mode",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.plot_only:
        plot_from_csv(csv_path=args.csv, output_plot=args.output_plot)
        return
    benchmark(
        batch_sizes=args.batch_sizes,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        bpe_path=args.bpe_path,
        checkpoint_path=args.checkpoint_path,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        device=args.device,
    )


if __name__ == "__main__":
    main()
