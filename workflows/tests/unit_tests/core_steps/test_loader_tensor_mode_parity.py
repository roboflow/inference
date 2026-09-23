"""The loader must follow the INSTALLED configuration's tensor flag.

The flag the loader branches on comes from the configuration, and each mode
registers its own variant of a block that has both. The child runs in a
scratch cwd inside the workflows project (never above it) with PYTHONPATH
scrubbed - `inference_models` and every other dep must come from the
installed environment, not from a sibling monorepo checkout.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CHILD = r"""
import dataclasses
import json
import os
from roboflow_workflows.configuration import (
    configure_process, default_configuration, resolve_image_tensor_device,
)

tensor_mode = os.environ["ENABLE_TENSOR_DATA_REPRESENTATION"] == "True"
base = default_configuration()
configure_process(dataclasses.replace(
    base,
    tensor=dataclasses.replace(
        base.tensor, representation_enabled=tensor_mode,
        image_tensor_device=resolve_image_tensor_device(tensor_mode),
    ),
))
from roboflow_workflows.core_steps import loader

print(json.dumps({
    "flag": loader.ENABLE_TENSOR_DATA_REPRESENTATION,
    "blocks": sorted(f"{b.__module__}.{b.__name__}" for b in loader.load_blocks()),
    "serializers": sorted(loader.KINDS_SERIALIZERS),
    "initializers": sorted(loader.REGISTERED_INITIALIZERS),
}))
"""


def _load(tensor_mode: bool, scratch: Path) -> dict:
    child_env = {**os.environ}
    # Standalone package tests must not depend on a sibling monorepo
    # checkout. Any inherited PYTHONPATH is dropped; the child resolves
    # everything from the installed environment.
    child_env.pop("PYTHONPATH", None)
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
        cwd=scratch,
        env=child_env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


BLUR_NUMPY = "roboflow_workflows.core_steps.classical_cv.image_blur.v1.ImageBlurBlockV1"
BLUR_TENSOR = (
    "roboflow_workflows.core_steps.classical_cv.image_blur.v1_tensor.ImageBlurBlockV1"
)


@pytest.mark.parametrize("tensor_mode", [False, True], ids=["numpy", "tensor"])
def test_loader_follows_the_installed_configuration(
    tensor_mode: bool, tmp_path
) -> None:
    payload = _load(tensor_mode, tmp_path)
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
    from roboflow_workflows.configuration import WorkflowsConfiguration
    from roboflow_workflows.errors import BlockInitParameterNotProvidedError
    from roboflow_workflows.execution_engine.introspection.blocks_loader import (
        load_initializers,
    )
    from roboflow_workflows.execution_engine.v1.compiler.steps_initialiser import (
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
