import json
from pathlib import Path

import pytest

from inference_models.models.grounding_dino.config import data_only_config


@pytest.mark.parametrize(
    "source",
    [
        "import os\nvalue = 1",
        'value = print("must not execute")',
        "value = (1).__class__",
        '_base_ = "/tmp/outside.py"',
        "config.value = 1",
        "value = [x for x in range(2)]",
        'value = 1\nraise RuntimeError("not data")',
    ],
)
def test_rejects_executable_config_before_loading(tmp_path, source):
    path = tmp_path / "config.py"
    path.write_text(source)
    with pytest.raises(ValueError):
        with data_only_config(str(path)):
            pytest.fail("Unsafe config reached the loader")


def test_preserves_literal_data_and_cleans_temporary_config(tmp_path):
    path = tmp_path / "config.py"
    path.write_text(
        '"docstring"\nbackbone = "swin_B_384_22k"\nconfig = {"layers": [1, 2], "enabled": True}\n'
    )
    with data_only_config(str(path)) as safe_path:
        assert Path(safe_path).suffix == ".json"
        assert json.loads(Path(safe_path).read_text()) == {
            "backbone": "swin_B_384_22k",
            "config": {"layers": [1, 2], "enabled": True},
        }
    assert not Path(safe_path).exists()


@pytest.mark.parametrize(
    "filename", ["GroundingDINO_SwinT_OGC.py", "GroundingDINO_SwinB_cfg.py"]
)
def test_installed_shipped_configs_preserve_dependency_configuration(filename):
    import groundingdino
    from groundingdino.util.slconfig import SLConfig

    config = Path(groundingdino.__file__).parent / "config" / filename
    expected = SLConfig.fromfile(str(config))
    with data_only_config(str(config)) as safe_path:
        actual = SLConfig.fromfile(safe_path)
        assert actual._cfg_dict == expected._cfg_dict
        assert actual.modelname == "groundingdino"


def test_config_size_is_bounded(tmp_path):
    path = tmp_path / "config.py"
    path.write_bytes(b"#" * 1_000_001)
    with pytest.raises(ValueError, match="1 MB"):
        with data_only_config(str(path)):
            pytest.fail("Oversized config accepted")
