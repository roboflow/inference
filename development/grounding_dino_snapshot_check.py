"""Diagnose Grounding DINO's Hugging Face dependency loading.

This script is intended to run inside an Inference production image. It compares
the current unrestricted BERT snapshot download with two restricted variants,
loads BERT entirely from the resulting local snapshot, and performs online and
offline Grounding DINO forward passes using the first viable restricted variant.
Results are written to JSON so remote-runner can fetch them after the container
exits.
"""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

SNAPSHOT_CASES = {
    "current-unrestricted": {
        "repo_id": "bert-base-uncased",
    },
    "alias-restricted": {
        "repo_id": "bert-base-uncased",
        "allow_patterns": BERT_ALLOW_PATTERNS,
    },
    "canonical-restricted": {
        "repo_id": "google-bert/bert-base-uncased",
        "allow_patterns": BERT_ALLOW_PATTERNS,
    },
}


def snapshot_download_kwargs(
    case_name: str,
    cache_dir: Path,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    """Build deterministic snapshot arguments for one diagnostic case."""
    case = SNAPSHOT_CASES[case_name]
    result = {
        "repo_id": case["repo_id"],
        "revision": BERT_REVISION,
        "cache_dir": str(cache_dir),
        "local_files_only": local_files_only,
    }
    if "allow_patterns" in case:
        result["allow_patterns"] = list(case["allow_patterns"])
    return result


def exception_result(error: BaseException) -> Dict[str, Any]:
    return {
        "ok": False,
        "exception_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


def installed_versions() -> Dict[str, str]:
    packages = [
        "inference",
        "inference-models",
        "huggingface-hub",
        "hf-xet",
        "transformers",
        "torch",
        "rf-groundingdino",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def list_snapshot_files(snapshot_path: Path) -> List[str]:
    return sorted(
        str(path.relative_to(snapshot_path))
        for path in snapshot_path.rglob("*")
        if path.is_file()
    )


def run_snapshot_case(
    case_name: str,
    cache_root: Path,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    case_cache = cache_root / case_name
    case_cache.mkdir(parents=True, exist_ok=True)
    kwargs = snapshot_download_kwargs(
        case_name=case_name,
        cache_dir=case_cache,
        local_files_only=local_files_only,
    )
    print(f"SNAPSHOT_START case={case_name} kwargs={kwargs}", flush=True)
    try:
        snapshot_path = Path(snapshot_download(**kwargs))
        result = {
            "ok": True,
            "snapshot_path": str(snapshot_path),
            "files": list_snapshot_files(snapshot_path),
        }
        print(
            f"SNAPSHOT_OK case={case_name} files={len(result['files'])}",
            flush=True,
        )
        return result
    except Exception as error:
        result = exception_result(error)
        print(
            f"SNAPSHOT_FAILED case={case_name} "
            f"type={result['exception_type']} error={result['error']}",
            flush=True,
        )
        return result


def validate_bert_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    print(f"BERT_LOAD_START path={snapshot_path}", flush=True)
    try:
        import torch
        from transformers import AutoTokenizer, BertModel

        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
        )
        model = BertModel.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
        )
        model.eval()
        inputs = tokenizer("apple", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        result = {
            "ok": True,
            "hidden_state_shape": list(output.last_hidden_state.shape),
        }
        print(
            f"BERT_LOAD_OK shape={result['hidden_state_shape']}",
            flush=True,
        )
        return result
    except Exception as error:
        result = exception_result(error)
        print(
            f"BERT_LOAD_FAILED type={result['exception_type']} "
            f"error={result['error']}",
            flush=True,
        )
        return result


def run_grounding_dino_forward(
    case_name: str,
    cache_root: Path,
) -> Dict[str, Any]:
    print(f"GROUNDING_DINO_START case={case_name}", flush=True)
    try:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from inference.models.grounding_dino import grounding_dino

        candidate_cache = cache_root / case_name

        def candidate_snapshot_download(**kwargs):
            return snapshot_download(
                **snapshot_download_kwargs(
                    case_name=case_name,
                    cache_dir=candidate_cache,
                    local_files_only=bool(kwargs.get("local_files_only", False)),
                )
            )

        grounding_dino.snapshot_download = candidate_snapshot_download
        model = grounding_dino.GroundingDINO()
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        with torch.no_grad():
            response = model.infer(
                image=image,
                text=["apple"],
                box_threshold=0.99,
                text_threshold=0.99,
            )
        result = {
            "ok": True,
            "device": model.model.device,
            "prediction_count": len(response.predictions),
            "image": response.image.dict(),
        }
        print(
            f"GROUNDING_DINO_OK device={result['device']} "
            f"predictions={result['prediction_count']}",
            flush=True,
        )
        return result
    except Exception as error:
        result = exception_result(error)
        print(
            f"GROUNDING_DINO_FAILED type={result['exception_type']} "
            f"error={result['error']}",
            flush=True,
        )
        return result


def run_offline_subprocess(
    case_name: str,
    output_dir: Path,
    cache_root: Path,
    model_cache: Path,
) -> Dict[str, Any]:
    offline_result_path = output_dir / "offline-result.json"
    environment = os.environ.copy()
    environment.update(
        {
            "OFFLINE_MODE": "True",
            "MODEL_CACHE_DIR": str(model_cache),
            "HF_HOME": str(cache_root / "offline-hf-home"),
            "HF_HUB_CACHE": str(cache_root / case_name),
        }
    )
    environment.pop("API_KEY", None)
    environment.pop("ROBOFLOW_API_KEY", None)
    environment.pop(
        "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START",
        None,
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "offline-forward",
        "--candidate",
        case_name,
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(cache_root.parent),
        "--result-path",
        str(offline_result_path),
    ]
    print(f"OFFLINE_SUBPROCESS_START command={command}", flush=True)
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    result = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }
    if offline_result_path.is_file():
        result["details"] = json.loads(offline_result_path.read_text())
    return result


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_full_diagnostic(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    cache_workspace = Path(args.cache_dir).resolve()
    cache_root = cache_workspace / "snapshot-caches"
    model_cache = cache_workspace / "model-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MODEL_CACHE_DIR"] = str(model_cache)

    summary = {
        "versions": installed_versions(),
        "snapshot_cases": {},
    }
    case_names = list(SNAPSHOT_CASES)
    if args.skip_unrestricted:
        case_names.remove("current-unrestricted")

    for case_name in case_names:
        summary["snapshot_cases"][case_name] = run_snapshot_case(
            case_name=case_name,
            cache_root=cache_root,
        )

    selected_case = next(
        (
            case_name
            for case_name in ("alias-restricted", "canonical-restricted")
            if summary["snapshot_cases"].get(case_name, {}).get("ok")
        ),
        None,
    )
    summary["selected_case"] = selected_case
    if selected_case is None:
        write_json(output_dir / "diagnostic-summary.json", summary)
        return 1

    snapshot_path = Path(summary["snapshot_cases"][selected_case]["snapshot_path"])
    summary["bert_validation"] = validate_bert_snapshot(snapshot_path)
    if not summary["bert_validation"]["ok"]:
        write_json(output_dir / "diagnostic-summary.json", summary)
        return 1

    summary["grounding_dino_online"] = run_grounding_dino_forward(
        case_name=selected_case,
        cache_root=cache_root,
    )
    if not summary["grounding_dino_online"]["ok"]:
        write_json(output_dir / "diagnostic-summary.json", summary)
        return 1

    if not args.skip_offline:
        summary["grounding_dino_offline"] = run_offline_subprocess(
            case_name=selected_case,
            output_dir=output_dir,
            cache_root=cache_root,
            model_cache=model_cache,
        )

    write_json(output_dir / "diagnostic-summary.json", summary)
    offline_ok = summary.get("grounding_dino_offline", {"ok": True})["ok"]
    return 0 if offline_ok else 1


def run_offline_forward(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    cache_root = Path(args.cache_dir).resolve() / "snapshot-caches"
    result = {
        "versions": installed_versions(),
        "offline_mode": os.getenv("OFFLINE_MODE"),
        "forward": run_grounding_dino_forward(
            case_name=args.candidate,
            cache_root=cache_root,
        ),
    }
    write_json(Path(args.result_path), result)
    return 0 if result["forward"]["ok"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/output")
    parser.add_argument(
        "--cache-dir",
        default="/tmp/grounding-dino-snapshot-check",
    )
    parser.add_argument(
        "--phase",
        choices=["full", "offline-forward"],
        default="full",
    )
    parser.add_argument(
        "--candidate",
        choices=list(SNAPSHOT_CASES),
        default="canonical-restricted",
    )
    parser.add_argument("--result-path", default="/output/offline-result.json")
    parser.add_argument("--skip-unrestricted", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "offline-forward":
        return run_offline_forward(args)
    return run_full_diagnostic(args)


if __name__ == "__main__":
    raise SystemExit(main())
