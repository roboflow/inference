import builtins
import os
import subprocess
import sys

import pytest
from streamvision.camera.dgpu_producer import PyNvVideoCodecFrameProducer


def test_constructor_raises_import_error_naming_the_nvdec_extra_when_package_missing(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)

    with pytest.raises(ImportError, match=r"streamvision\[nvdec\]"):
        PyNvVideoCodecFrameProducer("video.mp4")


def test_constructor_keeps_original_message_for_non_missing_import_failures(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def _raise_for_pynvvideocodec(name, *args, **kwargs):
        if name == "PyNvVideoCodec":
            raise OSError("libnvidia-encode.so.1: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_for_pynvvideocodec)

    with pytest.raises(ImportError) as exc_info:
        PyNvVideoCodecFrameProducer("video.mp4")

    assert "streamvision[nvdec]" not in str(exc_info.value)


def test_camera_modules_import_cleanly_when_pynvvideocodec_is_missing() -> None:
    script = (
        "import sys\n"
        "sys.modules['PyNvVideoCodec'] = None\n"
        "import streamvision.camera.dgpu_producer\n"
        "import streamvision.camera.discoverability\n"
        "import streamvision.camera.video_source\n"
        "print('ok')\n"
    )
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
