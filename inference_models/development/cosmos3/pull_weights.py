"""Pull NVIDIA Cosmos 3 Edge weights and materialize inference_models package layouts.

The HF repo (nvidia/Cosmos3-Edge) is dual-format: a transformers-style reasoner at
the repo root and a diffusers-style generator (model_index.json + transformer/ +
vae/ + scheduler/). The two towers share weight files (the reasoner's
model.safetensors.index.json maps into transformer/*.safetensors and
vision_encoder/model.safetensors), so this script downloads ONE snapshot and
materializes each tower's package directory from it with hardlinks (falling back
to copies across filesystems).

Package layouts produced (what `from_pretrained` expects):

    <output-dir>/cosmos-3-edge/          -> Cosmos3EdgeReasoner.from_pretrained(...)
    <output-dir>/cosmos-3-edge-world/    -> Cosmos3EdgeWorldModel.from_pretrained(...)

The world package additionally needs a `cosmos3_generator_runtime.py` at its root
(the self-contained runtime module the loader imports). Pass --runtime-module to
inject one; the reference implementation lives next to this script
(reference_generator_runtime.py).

To mirror a package into GCS (same layout, no wrapping directory), pass
--gcs-dest; each package uploads to <gcs-dest>/<package-name>/:

    python pull_weights.py --towers reasoner,world \
        --runtime-module reference_generator_runtime.py \
        --gcs-dest gs://my-bucket/cosmos3

Use --dry-run to print the file plan without touching disk or GCS.
"""

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys

HF_REPO = "nvidia/Cosmos3-Edge"
RUNTIME_MODULE_NAME = "cosmos3_generator_runtime.py"

SNAPSHOT_IGNORE = ["assets/*", "images/*"]

# Files each tower's package needs, as fnmatch patterns over repo-relative paths.
TOWER_FILES = {
    "reasoner": [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "vision_encoder/*",
        # shared MoT weights - the reasoner index maps into these shards
        "transformer/diffusion_pytorch_model*.safetensors",
    ],
    "world": [
        "model_index.json",
        "modular_model_index.json",
        "negative_prompt.json",
        "scheduler/*",
        "transformer/*",
        "vae/*",
        "text_tokenizer/*",
    ],
}

PACKAGE_NAMES = {
    "reasoner": "cosmos-3-edge",
    "world": "cosmos-3-edge-world",
}


def main() -> int:
    args = _parse_args()
    towers = [t.strip() for t in args.towers.split(",") if t.strip()]
    unknown = set(towers).difference(TOWER_FILES)
    if unknown:
        raise SystemExit(f"Unknown tower(s): {sorted(unknown)}; pick from {sorted(TOWER_FILES)}")

    snapshot_dir = args.snapshot_dir or _download_snapshot(args)
    available = _list_snapshot_files(snapshot_dir)

    for tower in towers:
        package_dir = os.path.join(args.output_dir, PACKAGE_NAMES[tower])
        selected = _select_files(available, TOWER_FILES[tower])
        if not selected:
            raise SystemExit(f"No files matched for tower '{tower}' in {snapshot_dir}")
        print(f"[{tower}] {len(selected)} files -> {package_dir}")
        if args.dry_run:
            for rel in selected:
                print(f"  {rel}")
        else:
            _materialize(snapshot_dir, package_dir, selected)
        if tower == "world":
            _inject_runtime_module(args, package_dir)
        if args.gcs_dest:
            _upload_to_gcs(
                package_dir=package_dir,
                gcs_dest=f"{args.gcs_dest.rstrip('/')}/{PACKAGE_NAMES[tower]}",
                dry_run=args.dry_run,
            )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--revision", default=None, help="HF revision/tag to pin.")
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Reuse an existing snapshot directory instead of downloading.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/packages",
        help="Directory receiving one package subdirectory per tower.",
    )
    parser.add_argument(
        "--towers",
        default="reasoner,world",
        help="Comma-separated subset of: reasoner, world.",
    )
    parser.add_argument(
        "--runtime-module",
        default=None,
        help=f"Python file copied into the world package as {RUNTIME_MODULE_NAME}.",
    )
    parser.add_argument(
        "--gcs-dest",
        default=None,
        help="gs://bucket/prefix - each package is mirrored to <dest>/<package-name>/.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _download_snapshot(args: argparse.Namespace) -> str:
    from huggingface_hub import snapshot_download

    print(f"Downloading snapshot of {args.hf_repo} (weights only)...")
    return snapshot_download(
        args.hf_repo,
        revision=args.revision,
        ignore_patterns=SNAPSHOT_IGNORE,
    )


def _list_snapshot_files(snapshot_dir: str) -> list:
    collected = []
    for root, _, files in os.walk(snapshot_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, snapshot_dir)
            if rel.startswith(".cache"):
                continue
            collected.append(rel)
    return sorted(collected)


def _select_files(available: list, patterns: list) -> list:
    return [
        rel
        for rel in available
        if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)
    ]


def _materialize(snapshot_dir: str, package_dir: str, selected: list) -> None:
    for rel in selected:
        src = os.path.realpath(os.path.join(snapshot_dir, rel))
        dst = os.path.join(package_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _inject_runtime_module(args: argparse.Namespace, package_dir: str) -> None:
    if not args.runtime_module:
        print(
            f"  note: world package has no {RUNTIME_MODULE_NAME}; "
            "Cosmos3EdgeWorldModel.from_pretrained will refuse to load it "
            "(pass --runtime-module)."
        )
        return
    dst = os.path.join(package_dir, RUNTIME_MODULE_NAME)
    print(f"  runtime module: {args.runtime_module} -> {dst}")
    if not args.dry_run:
        os.makedirs(package_dir, exist_ok=True)
        shutil.copy2(args.runtime_module, dst)


def _upload_to_gcs(package_dir: str, gcs_dest: str, dry_run: bool) -> None:
    tool = shutil.which("gcloud")
    if tool:
        cmd = ["gcloud", "storage", "cp", "-r", f"{package_dir.rstrip('/')}/*", gcs_dest + "/"]
    elif shutil.which("gsutil"):
        cmd = ["gsutil", "-m", "cp", "-r", f"{package_dir.rstrip('/')}/*", gcs_dest + "/"]
    else:
        raise SystemExit("Neither gcloud nor gsutil found on PATH for --gcs-dest.")
    print(f"  upload: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(" ".join(cmd), shell=True, check=True)


if __name__ == "__main__":
    sys.exit(main())
