from pathlib import Path
import textwrap

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
            "seed": 123,
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
    assert manifest["workload"]["record_loading"] == "eager"
    assert manifest["workload"]["seed"] == 123
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
            "record_loading": "lazy",
            "seed": 7,
        }
    )

    command = build_nsys_command(
        config=config,
        run_dir=tmp_path / "smoke" / "runs" / "run-1",
        config_path="configs/profile config.yaml",
    )

    assert command.startswith("nsys profile")
    assert "--nvtx-capture=custom-target@*" in command
    assert "--record-loading \\\n  lazy" in command
    assert "--seed \\\n  7" in command
    assert "'configs/profile config.yaml'" in command
    assert "uv run python development/profiling/main.py" in command


def test_validation_runs_during_warmup_only(tmp_path):
    counter_path = tmp_path / "validate_count.txt"
    target_file = tmp_path / "target.py"
    target_file.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            import torch


            COUNTER_PATH = Path({str(counter_path)!r})


            class ValidationCounterTarget:
                name = "validation-counter"

                def prepare(self, record, *, device):
                    return torch.tensor([1.0], device=device)

                def run(self, prepared):
                    return prepared

                def validate(self, output):
                    current_count = 0
                    if COUNTER_PATH.exists():
                        current_count = int(COUNTER_PATH.read_text())
                    COUNTER_PATH.write_text(str(current_count + 1))

                def summarize(self, output):
                    return {{"shape": list(output.shape)}}


            target = ValidationCounterTarget()
            """
        ),
        encoding="utf-8",
    )
    config = parse_profile_config(
        {
            "profile_name": "validation-counter",
            "target": {
                "name": "validation-counter",
                "import_path": f"{target_file}:target",
            },
            "data_source": {"name": "dummy", "record_count": 2},
            "device": "cpu",
            "warmup": 1,
            "iterations": 3,
        }
    )

    run_profile(
        config=config,
        run_id="test-run",
        run_dir=tmp_path / "validation-counter" / "runs" / "test-run",
        argv=["development/profiling/main.py", "--config", "config.yaml"],
    )

    assert counter_path.read_text() == "2"
