"""Guard the direct environment reads allowed inside the Workflows package.

Server imports are covered by test_decontamination_lint.
"""

import ast

from tests.unit_tests.test_configuration import environment_reads
from tests.unit_tests.test_decontamination_lint import PROJECT_ROOT, WORKFLOWS_ROOT

# Every direct environment read that remains inside `inference/core/workflows`
# after Phase 5, with its owner. Phase 5 removes `inference.core.env` IMPORTS;
# it does not touch these. Adding a row here is a deliberate act.
PERMITTED_ENVIRONMENT_READS = {
    ("roboflow_workflows/enterprise_blocks/sinks/event_writer/v1.py", 1),
    ("roboflow_workflows/enterprise_blocks/sinks/opc_writer/v1.py", 1),
    ("roboflow_workflows/execution_engine/introspection/blocks_loader.py", 1),
    ("roboflow_workflows/execution_engine/v1/core.py", 1),
    ("roboflow_workflows/execution_engine/v1/debugger/core.py", 2),
    (
        "roboflow_workflows/execution_engine/v1/dynamic_blocks/modal_executor.py",
        3,
    ),
    (
        "roboflow_workflows/core_steps/secrets_providers/environment_secrets_store/v1.py",
        1,
    ),
    # Phase 9 relocates these two files; if Phase 5 runs first their reads stay.
    ("roboflow_workflows/core_steps/sinks/roboflow/vision_events/v1.py", 1),
    (
        "roboflow_workflows/core_steps/sinks/roboflow/vision_events/v1_tensor.py",
        1,
    ),
}


def test_the_environment_read_inventory_is_frozen() -> None:
    found = {}
    for path in sorted(WORKFLOWS_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_bytes().decode("utf-8"), filename=str(path))
        reads = len(environment_reads(tree))
        if reads:
            found[path.relative_to(PROJECT_ROOT).as_posix()] = reads
    expected = {path: count for path, count in PERMITTED_ENVIRONMENT_READS}
    # Phase 9 may already have relocated its two files.
    expected = {
        path: count
        for path, count in expected.items()
        if (PROJECT_ROOT / path).exists()
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
