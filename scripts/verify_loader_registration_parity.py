"""Prove `core_steps/loader.py` registers the same things after a refactor.

This is a BEFORE/AFTER comparison of the WORKING TREE - record before the edit,
compare after. It does not read git history.

    python scripts/verify_loader_registration_parity.py --record  <out.json>
    python scripts/verify_loader_registration_parity.py --compare <out.json> \
        [--expect-added-initializers configuration]

One child per tensor mode, with USE_INFERENCE_MODELS pinned True (env.py:1486 ANDs
it into the flag) and every other registration-gating variable pinned so the answer
depends only on the flag. The child also reports the EFFECTIVE
flag it observed, and the parent asserts it matches the mode it asked for -
otherwise a broken configuration hand-off would silently compare a mode against
itself.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PAYLOAD_KEYS = ("blocks", "kinds", "serializers", "deserializers", "initializers")

CHILD = r"""
import json
from inference.core.workflows.core_steps import loader

print(json.dumps({
    "flag": loader.ENABLE_TENSOR_DATA_REPRESENTATION,
    "blocks": sorted(f"{b.__module__}.{b.__name__}" for b in loader.load_blocks()),
    "kinds": sorted(k.name for k in loader.load_kinds()),
    "serializers": sorted(loader.KINDS_SERIALIZERS),
    "deserializers": sorted(loader.KINDS_DESERIALIZERS),
    "initializers": sorted(loader.REGISTERED_INITIALIZERS),
}))
"""


def snapshot(tensor_mode: bool) -> dict:
    child_env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "inference_models")}
    child_env["ENABLE_TENSOR_DATA_REPRESENTATION"] = "True" if tensor_mode else "False"
    # `env.py:1486` ANDs USE_INFERENCE_MODELS into the tensor flag (it defaults
    # to False on Windows and may be pinned False elsewhere), so the requested
    # mode is only reachable with BOTH pinned - round-3 defect 5. The child
    # still reports the EFFECTIVE flag and the parent asserts it below.
    child_env["USE_INFERENCE_MODELS"] = "True"
    child_env["SAM3_3D_OBJECTS_ENABLED"] = "False"
    child_env["WORKFLOW_DISABLED_BLOCK_TYPES"] = ""
    child_env["WORKFLOW_DISABLED_BLOCK_PATTERNS"] = ""
    child_env.pop("WORKFLOWS_PLUGINS", None)
    completed = subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stdout + completed.stderr)
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if payload["flag"] is not tensor_mode:
        raise SystemExit(
            f"asked for tensor={tensor_mode} but the loader observed "
            f"{payload['flag']!r} - the configuration hand-off is broken"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", metavar="PATH")
    parser.add_argument("--compare", metavar="PATH")
    parser.add_argument(
        "--expect-added-initializers",
        nargs="*",
        default=[],
        help="initializer names this refactor deliberately ADDS",
    )
    args = parser.parse_args()
    current = {"off": snapshot(False), "on": snapshot(True)}
    if args.record:
        Path(args.record).write_text(json.dumps(current, indent=1))
        for mode, payload in current.items():
            print(
                f"recorded tensor={mode}: {len(payload['blocks'])} blocks, "
                f"{len(payload['kinds'])} kinds, "
                f"{len(payload['serializers'])} serializers, "
                f"{len(payload['initializers'])} initializers "
                f"{payload['initializers']}"
            )
        return 0
    recorded = json.loads(Path(args.compare).read_text())
    expected_added = set(args.expect_added_initializers)
    ok = True
    for mode in ("off", "on"):
        for key in PAYLOAD_KEYS:
            before, after = set(recorded[mode][key]), set(current[mode][key])
            added, removed = after - before, before - after
            if key == "initializers":
                # The ONE intended change: this task adds `configuration`.
                if added != expected_added or removed:
                    ok = False
                    print(
                        f"tensor={mode} initializers: expected exactly "
                        f"+{sorted(expected_added)}, got +{sorted(added)} "
                        f"-{sorted(removed)}"
                    )
                continue
            if added or removed:
                ok = False
                print(f"tensor={mode} {key}: -{sorted(removed)} +{sorted(added)}")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
