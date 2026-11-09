import os
import subprocess
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize("target", ["start_test_docker_cpu", "start_test_docker_gpu"])
@pytest.mark.parametrize("torch_preprocessing", [None, "False", "True"])
def test_container_receives_preprocessing_mode(
    target: str, torch_preprocessing: Optional[str], tmp_path: Path
) -> None:
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    env = dict(os.environ)
    env.pop("USE_PYTORCH_FOR_PREPROCESSING", None)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    if torch_preprocessing is not None:
        env["USE_PYTORCH_FOR_PREPROCESSING"] = torch_preprocessing

    result = subprocess.run(
        [
            "make",
            "--silent",
            target,
            "PORT=9101",
            "INFERENCE_TEST_IMAGE=test-image",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = result.stdout.splitlines()
    flag = f"USE_PYTORCH_FOR_PREPROCESSING={torch_preprocessing or 'False'}"
    assert flag in args
    assert args[args.index(flag) - 1] == "-e"
