"""Guard for the shared CI concurrency policy.

Every workflow triggered by ``pull_request`` or ``push`` must declare a
top-level ``concurrency`` block whose ``group`` is keyed by
``github.workflow`` and which sets ``cancel-in-progress``. Because
``github.workflow`` resolves to the workflow ``name:``, names must also be
unique, otherwise two workflows would share a concurrency group and cancel
each other.

Usage: ``python3 .github/scripts/check_workflow_concurrency.py [workflows_dir]``
Exit code 0 when every workflow complies, 1 otherwise (offenders are listed).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import yaml

GUARDED_TRIGGERS = {"pull_request", "push"}
DEFAULT_WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / "workflows"


def _triggers(document: dict) -> set[str]:
    # PyYAML parses a bare ``on`` key as boolean True.
    on = document.get(True, document.get("on"))
    if isinstance(on, dict):
        return set(on)
    if isinstance(on, list):
        return set(on)
    if isinstance(on, str):
        return {on}
    return set()


def find_offenders(workflows_dir: Path) -> list[str]:
    offenders: list[str] = []
    names: dict[str, list[str]] = defaultdict(list)
    for path in sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml")):
        with path.open() as handle:
            document = yaml.safe_load(handle)
        if not isinstance(document, dict):
            offenders.append(f"{path.name}: not a mapping at top level")
            continue
        name = document.get("name")
        if name:
            names[str(name)].append(path.name)
        if not _triggers(document) & GUARDED_TRIGGERS:
            continue
        concurrency = document.get("concurrency")
        if not isinstance(concurrency, dict):
            offenders.append(
                f"{path.name}: triggered by pull_request/push but has no top-level concurrency block"
            )
            continue
        group = str(concurrency.get("group", ""))
        if "github.workflow" not in group:
            offenders.append(f"{path.name}: concurrency.group must be keyed by github.workflow")
        if "cancel-in-progress" not in concurrency:
            offenders.append(f"{path.name}: concurrency block must set cancel-in-progress")
    for name, files in sorted(names.items()):
        if len(files) > 1:
            offenders.append(
                f"duplicate workflow name {name!r} in {', '.join(files)}: "
                "github.workflow is the concurrency-group key, names must be unique"
            )
    return offenders


def main(argv: list[str]) -> int:
    workflows_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_WORKFLOWS_DIR
    offenders = find_offenders(workflows_dir)
    if offenders:
        print("Concurrency policy violations:")
        for offender in offenders:
            print(f"  - {offender}")
        print(
            "\nAdd the shared concurrency block (copy it from any workflow under "
            ".github/workflows) or fix the duplicate name."
        )
        return 1
    print(f"OK: concurrency policy holds for every workflow in {workflows_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
