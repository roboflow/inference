import pytest

from development.profiling.config import parse_profile_config


def test_parse_profile_config_with_defaults():
    config = parse_profile_config(
        {
            "profile_name": "example",
            "target": {"name": "smoke-tensor"},
            "data_source": {"name": "dummy", "record_count": 3},
        }
    )

    assert config.profile_name == "example"
    assert config.target.name == "smoke-tensor"
    assert config.data_source.name == "dummy"
    assert config.data_source.parameters == {"record_count": 3}
    assert config.record_loading == "eager"
    assert config.seed is None
    assert config.capture_range == "profile-target"
    assert config.cuda.synchronize_before_capture is True
    assert config.cuda.synchronize_each_iteration is False


def test_parse_profile_config_accepts_workload_and_cuda_overrides():
    config = parse_profile_config(
        {
            "profile": {"name": "example"},
            "target": {
                "name": "generated",
                "import_path": "inference_profiling/snippets/generated/target.py:target",
                "profile_prepare": True,
            },
            "data": {"source": "dummy"},
            "workload": {
                "device": "cuda",
                "warmup": 2,
                "iterations": 5,
                "capture_range": "custom-capture",
                "record_loading": "lazy",
                "seed": 42,
            },
            "cuda": {"synchronize_each_iteration": True},
        }
    )

    assert config.profile_name == "example"
    assert config.target.profile_prepare is True
    assert config.device == "cuda"
    assert config.warmup == 2
    assert config.iterations == 5
    assert config.capture_range == "custom-capture"
    assert config.record_loading == "lazy"
    assert config.seed == 42
    assert config.cuda.synchronize_each_iteration is True


def test_parse_profile_config_rejects_empty_profile_name():
    with pytest.raises(ValueError, match="String config values must not be empty"):
        parse_profile_config(
            {
                "profile_name": " ",
                "target": {"name": "smoke-tensor"},
                "data_source": {"name": "dummy"},
            }
        )


def test_parse_profile_config_rejects_invalid_iteration_count():
    with pytest.raises(ValueError, match="iterations must be greater than"):
        parse_profile_config(
            {
                "profile_name": "example",
                "target": {"name": "smoke-tensor"},
                "data_source": {"name": "dummy"},
                "iterations": 0,
            }
        )
