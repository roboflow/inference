"""Assemble a Cosmos AnomalyGen model package from a paidf-anomalygen checkout.

Sibling of pull_weights.py for the anomalygen tower. The base weights do not
live in one HF repo - they are the frozen towers that NVIDIA's
`scripts.download_checkpoints` materializes under the GA repo's `checkpoints/`
tree - so this tool assembles the package from:

- ``--checkpoints-dir``: a GA ``checkpoints/`` tree holding the frozen towers
  (Cosmos-Predict2-2B-Text2Image DiT + VAE tokenizer, google-t5/t5-large,
  NVDINOV2). Optionally also ``nvidia/Cosmos-Guardrail1`` if the package
  should ship the image content-safety guardrail.
- ``--run-dir``: a finished training run directory
  (``.../anomaly_gen/<category>/<run_name>``) holding ``ag_config.yaml`` and
  ``checkpoints/model/iter_*.pt``; the newest iteration is packaged.
- ``--runtime-module``: the self-contained runtime file, copied in as
  ``cosmos_anomalygen_runtime.py`` (see that file for the package layout and
  load contract).

Files are hardlinked when possible (copy fallback), never overwritten.
``class_names.txt`` and ``inference_config.json`` are derived from the run's
``ag_config.yaml`` (anomaly types = the trained ``<category>+<class>`` pairs;
generation defaults = the production recipe: guidance 1.5, 35 steps,
crop_ratio 4.0, crop-and-paste on, Poisson off).

Usage:
    python pull_anomalygen_weights.py \\
        --checkpoints-dir /workspace/paidf-anomalygen/checkpoints \\
        --run-dir /workspace/paidf-anomalygen/checkpoints/anomaly_gen/tube/tube_hole_webapp_7a6f33a9 \\
        --runtime-module cosmos_anomalygen_runtime.py \\
        --output-dir checkpoints/packages \\
        [--gcs-dest gs://bucket/prefix] [--dry-run]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

PACKAGE_NAME = "cosmos-anomalygen"
RUNTIME_MODULE_NAME = "cosmos_anomalygen_runtime.py"

# Frozen base towers, relative to --checkpoints-dir, packaged under
# checkpoints/<same path>. Guardrail files are optional (see docstring).
BASE_FILES = [
    "nvidia/Cosmos-Predict2-2B-Text2Image/model.pt",
    "nvidia/Cosmos-Predict2-2B-Text2Image/tokenizer/tokenizer.pth",
    "google-t5/t5-large/config.json",
    "google-t5/t5-large/model.safetensors",
    "google-t5/t5-large/spiece.model",
    "google-t5/t5-large/tokenizer.json",
    "NVDINOV2/nv_dinov2_classification_model.ckpt",
]
OPTIONAL_BASE_PATTERNS = [
    "nvidia/Cosmos-Guardrail1/*",
]

INFERENCE_DEFAULTS = {
    "architecture": "cosmos-anomalygen",
    "variant": "cosmos-anomalygen-2b",
    "task_type": "image-generation",
    "experiment": "predict2_anomaly_gen_ddp_2b",
    "guidance": 1.5,
    "num_steps": 35,
    "crop_ratio": 4.0,
    "crop_and_paste": True,
    "poisson_blend": False,
}


def main() -> int:
    args = _parse_args()
    package_dir = os.path.join(args.output_dir, PACKAGE_NAME)
    print(f"Assembling {PACKAGE_NAME} package -> {package_dir}")

    anomaly_types = _read_anomaly_types(args.run_dir)
    adapter_path = _find_latest_adapter(args.run_dir)
    print(f"  adapter: {adapter_path}")
    print(f"  anomaly types: {anomaly_types}")

    base_files = list(BASE_FILES)
    for pattern in OPTIONAL_BASE_PATTERNS:
        matches = glob.glob(os.path.join(args.checkpoints_dir, pattern))
        base_files.extend(
            os.path.relpath(m, args.checkpoints_dir)
            for m in matches
            if os.path.isfile(m)
        )
    missing = [
        rel
        for rel in BASE_FILES
        if not os.path.isfile(os.path.join(args.checkpoints_dir, rel))
    ]
    if missing:
        raise SystemExit(
            f"Missing base weights under {args.checkpoints_dir}: {missing}"
        )

    plan = [
        (os.path.join(args.checkpoints_dir, rel), os.path.join("checkpoints", rel))
        for rel in base_files
    ]
    plan.append(
        (
            adapter_path,
            os.path.join("checkpoints", "model", os.path.basename(adapter_path)),
        )
    )
    plan.append((os.path.join(args.run_dir, "ag_config.yaml"), "ag_config.yaml"))
    plan.append((args.runtime_module, RUNTIME_MODULE_NAME))

    for src, rel in plan:
        print(f"  {rel}")
        if not args.dry_run:
            _materialize(src, os.path.join(package_dir, rel))

    if not args.dry_run:
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "class_names.txt"), "w") as fp:
            fp.write("\n".join(anomaly_types) + "\n")
        inference_config = dict(INFERENCE_DEFAULTS)
        inference_config["anomaly_types"] = anomaly_types
        with open(os.path.join(package_dir, "inference_config.json"), "w") as fp:
            json.dump(inference_config, fp, indent=2)
            fp.write("\n")

    if args.gcs_dest:
        _upload_to_gcs(
            package_dir=package_dir,
            gcs_dest=f"{args.gcs_dest.rstrip('/')}/{PACKAGE_NAME}",
            dry_run=args.dry_run,
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoints-dir",
        required=True,
        help="A GA checkpoints/ tree holding the frozen base towers.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run dir with ag_config.yaml and checkpoints/model/iter_*.pt.",
    )
    parser.add_argument(
        "--runtime-module",
        default=os.path.join(os.path.dirname(__file__), RUNTIME_MODULE_NAME),
        help=f"Python file copied into the package as {RUNTIME_MODULE_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/packages",
        help="Directory receiving the package subdirectory.",
    )
    parser.add_argument(
        "--gcs-dest",
        default=None,
        help="gs://bucket/prefix - the package is mirrored to <dest>/cosmos-anomalygen/.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_anomaly_types(run_dir: str) -> list:
    import yaml

    ag_config_path = os.path.join(run_dir, "ag_config.yaml")
    with open(ag_config_path) as fp:
        ag_config = yaml.safe_load(fp)
    try:
        pairs = ag_config["dataloader_train"]["dataset"]["anomaly_types"]
    except (KeyError, TypeError):
        raise SystemExit(f"{ag_config_path} has no dataloader_train anomaly_types")
    return [f"{category}+{name}" for category, name in pairs]


def _find_latest_adapter(run_dir: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(run_dir, "checkpoints", "model", "iter_*.pt"))
    )
    if not candidates:
        raise SystemExit(f"No iter_*.pt under {run_dir}/checkpoints/model/")
    return candidates[-1]


def _materialize(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(os.path.realpath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _upload_to_gcs(package_dir: str, gcs_dest: str, dry_run: bool) -> None:
    tool = shutil.which("gcloud")
    if tool:
        cmd = [
            "gcloud",
            "storage",
            "cp",
            "-r",
            f"{package_dir.rstrip('/')}/*",
            gcs_dest + "/",
        ]
    elif shutil.which("gsutil"):
        cmd = [
            "gsutil",
            "-m",
            "cp",
            "-r",
            f"{package_dir.rstrip('/')}/*",
            gcs_dest + "/",
        ]
    else:
        raise SystemExit("Neither gcloud nor gsutil found on PATH for --gcs-dest.")
    print(f"  upload: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(" ".join(cmd), shell=True, check=True)


if __name__ == "__main__":
    sys.exit(main())
