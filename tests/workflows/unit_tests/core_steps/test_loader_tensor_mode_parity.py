"""The loader must follow the INSTALLED configuration's tensor flag.

The exhaustive before/after identity proof is
`scripts/verify_loader_registration_parity.py` (a one-shot migration gate, run
in Task 5.3). What must keep holding forever is the mechanism: the flag the
loader branches on comes from the configuration, and each mode registers its
own variant of a block that has both.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

CHILD = r"""
import json
from inference.core.workflows.core_steps import loader

print(json.dumps({
    "flag": loader.ENABLE_TENSOR_DATA_REPRESENTATION,
    "blocks": sorted(f"{b.__module__}.{b.__name__}" for b in loader.load_blocks()),
    "serializers": sorted(loader.KINDS_SERIALIZERS),
    "initializers": sorted(loader.REGISTERED_INITIALIZERS),
}))
"""


def _load(tensor_mode: bool) -> dict:
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
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


BLUR_NUMPY = (
    "inference.core.workflows.core_steps.classical_cv.image_blur.v1.ImageBlurBlockV1"
)
BLUR_TENSOR = "inference.core.workflows.core_steps.classical_cv.image_blur.v1_tensor.ImageBlurBlockV1"


@pytest.mark.parametrize("tensor_mode", [False, True], ids=["numpy", "tensor"])
def test_loader_follows_the_installed_configuration(tensor_mode: bool) -> None:
    payload = _load(tensor_mode)
    assert payload["flag"] is tensor_mode
    assert (BLUR_TENSOR in payload["blocks"]) is tensor_mode
    assert (BLUR_NUMPY in payload["blocks"]) is not tensor_mode
    assert len(payload["blocks"]) == len(set(payload["blocks"])), "duplicate blocks"
    assert "configuration" in payload["initializers"]
    # The tensor-native producers add a classification serialiser the numpy
    # path has no counterpart for (`loader.py:1615-1640`).
    assert ("classification_prediction" in payload["serializers"]) is tensor_mode


def test_configuration_resolves_for_core_sourced_blocks_only() -> None:
    """`REGISTERED_INITIALIZERS["configuration"]` is registered as
    `workflows_core.configuration` (`blocks_loader.py:372-376`), and resolution
    tries `{block_source}.{param}` then the bare name
    (`steps_initialiser.py:124-133`). So it resolves for `workflows_core` -
    including a plugin that declares `BLOCKS_SOURCE = "workflows_core"` - and
    NOT for an ordinary plugin. The plan promises exactly that; this pins it.
    """
    from inference.core.workflows.configuration import WorkflowsConfiguration
    from inference.core.workflows.errors import BlockInitParameterNotProvidedError
    from inference.core.workflows.execution_engine.introspection.blocks_loader import (
        load_initializers,
    )
    from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
        retrieve_init_parameter_values,
    )

    initializers = load_initializers()
    resolved = retrieve_init_parameter_values(
        block_name="step",
        block_init_parameter="configuration",
        block_source="workflows_core",
        explicit_init_parameters={},
        initializers=initializers,
    )
    assert isinstance(resolved, WorkflowsConfiguration)

    for foreign_source in ("my_plugin", "dynamic_workflows_blocks"):
        with pytest.raises(BlockInitParameterNotProvidedError):
            retrieve_init_parameter_values(
                block_name="step",
                block_init_parameter="configuration",
                block_source=foreign_source,
                explicit_init_parameters={},
                initializers=initializers,
            )
