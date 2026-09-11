"""No module Phase 5 owns may import `inference.core.env`.

This is narrower and louder than the decontamination lint, whose baseline
tolerates whatever is listed. It reuses the lint's own `collect_violations()`
so relative imports (`from ...core.env import X`), function-local imports and
the exec'd-string form are all handled by the one implementation that already
gets them right - and so a mere docstring mention of the module name is NOT a
violation.

It also freezes the inventory of direct environment reads left inside
workflows, so a new `os.getenv` / `os.environ[...]` cannot appear silently.
The scanner is `environment_reads` from Task 5.1's test module - ONE rule for
the whole phase (round-3 defect 7: the round-2 copy here started with
`if not isinstance(node, ast.Call): continue` and missed every subscript).
"""

import ast

from tests.workflows.unit_tests.test_configuration import environment_reads
from tests.workflows.unit_tests.test_decontamination_lint import (
    REPO_ROOT,
    WORKFLOWS_ROOT,
    collect_violations,
)

# Owned by Phase 9 (controller ruling R-S): these trees move to
# `inference/roboflow_workflows_plugin/`, where importing the server's env
# module is legitimate. Under R-U Phase 9 has landed, the paths do not exist
# and this tuple matches nothing; if Phase 5 runs first it skips them.
PHASE_9_PREFIXES = (
    "inference/core/workflows/core_steps/sinks/roboflow/",
    "inference/core/workflows/core_steps/integrations/roboflow/",
)

# Every direct environment read that remains inside `inference/core/workflows`
# after Phase 5, with its owner. Phase 5 removes `inference.core.env` IMPORTS;
# it does not touch these. Adding a row here is a deliberate act.
PERMITTED_ENVIRONMENT_READS = {
    ("inference/core/workflows/execution_engine/introspection/blocks_loader.py", 1),
    ("inference/core/workflows/execution_engine/v1/core.py", 1),
    ("inference/core/workflows/execution_engine/v1/debugger/core.py", 2),
    (
        "inference/core/workflows/execution_engine/v1/dynamic_blocks/modal_executor.py",
        3,
    ),
    (
        "inference/core/workflows/core_steps/secrets_providers/environment_secrets_store/v1.py",
        1,
    ),
    # Phase 9 relocates these two files; if Phase 5 runs first their reads stay.
    ("inference/core/workflows/core_steps/sinks/roboflow/vision_events/v1.py", 1),
    (
        "inference/core/workflows/core_steps/sinks/roboflow/vision_events/v1_tensor.py",
        1,
    ),
}


def test_no_owned_module_imports_inference_core_env() -> None:
    offenders = sorted(
        (path, module)
        for path, module in collect_violations()
        if module.startswith("inference.core.env")
        and not path.startswith(PHASE_9_PREFIXES)
    )
    assert not offenders, offenders


def test_the_environment_read_inventory_is_frozen() -> None:
    found = {}
    for path in sorted(WORKFLOWS_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_bytes().decode("utf-8"), filename=str(path))
        reads = len(environment_reads(tree))
        if reads:
            found[path.relative_to(REPO_ROOT).as_posix()] = reads
    expected = {path: count for path, count in PERMITTED_ENVIRONMENT_READS}
    # Phase 9 may already have relocated its two files.
    expected = {
        path: count for path, count in expected.items() if (REPO_ROOT / path).exists()
    }
    assert found == expected, {
        "unexpected": {k: v for k, v in found.items() if expected.get(k) != v},
        "missing": {k: v for k, v in expected.items() if found.get(k) != v},
    }


def test_the_inventory_scanner_counts_a_subscript_read() -> None:
    # Round-3 defect 7: `os.environ["X"]` is a read the import lint cannot see
    # either (it scans imports and generated import strings, `lint:91`).
    assert environment_reads(
        ast.parse('import os\nX = os.environ["SOME_CONFIG"]\n')
    ) == [2]
