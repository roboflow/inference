from pathlib import Path

import yaml

from development.profiling.config import parse_profile_config
from development.profiling.main import build_nsys_command, run_profile


def test_smoke_profile_execution_writes_expected_manifest_data(tmp_path):
    config = parse_profile_config(
        {
            "profile_name": "smoke",
            "target": {"name": "smoke-tensor"},
            "data_source": {
                "name": "dummy",
                "record_count": 2,
                "tensor_shape": [2, 2],
            },
            "device": "cpu",
            "warmup": 1,
            "iterations": 2,
            "repetitions": 1,
            "output_dir": str(tmp_path),
        }
    )
    run_dir = Path(tmp_path) / "smoke" / "runs" / "test-run"

    manifest = run_profile(
        config=config,
        run_id="test-run",
        run_dir=run_dir,
        argv=["development/profiling/main.py", "--config", "config.yaml"],
    )

    assert manifest["profile_name"] == "smoke"
    assert manifest["target"]["name"] == "smoke-tensor"
    assert manifest["data_source"]["name"] == "dummy"
    assert manifest["record_ids"] == ["dummy-0", "dummy-1"]
    assert manifest["workload"]["iterations"] == 2
    assert len(manifest["output_summaries"]) == 2


def test_main_writes_manifest_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "profile_name": "smoke",
                "target": {"name": "smoke-tensor"},
                "data_source": {"name": "dummy", "record_count": 1},
                "device": "cpu",
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    from development.profiling.main import main

    exit_code = main(["--config", str(config_path), "--run-id", "test-run"])

    assert exit_code == 0
    assert (tmp_path / "smoke" / "runs" / "test-run" / "manifest.yaml").exists()


def test_nsys_command_uses_capture_range_and_does_not_execute_nsys(tmp_path):
    config = parse_profile_config(
        {
            "profile_name": "smoke",
            "target": {"name": "smoke-tensor"},
            "data_source": {"name": "dummy"},
            "capture_range": "custom-target",
        }
    )

    command = build_nsys_command(
        config=config,
        run_dir=tmp_path / "smoke" / "runs" / "run-1",
        config_path="config.yaml",
    )

    assert command.startswith("nsys profile")
    assert "--nvtx-capture=custom-target@*" in command
    assert "uv run python development/profiling/main.py" in command
