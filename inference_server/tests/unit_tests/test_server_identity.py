import re

import pytest

from inference_server import configuration, server_identity

RANDOM_PART = r"[A-Za-z0-9]{6}"
REAL_GPU_INDEX = server_identity._gpu_index
REAL_JETSON_SERIAL = server_identity._jetson_serial


@pytest.fixture(autouse=True)
def fresh_identity(monkeypatch):
    monkeypatch.setattr(configuration, "INFERENCE_SERVER_ID", None)
    monkeypatch.setattr(server_identity, "_gpu_index", lambda: None)
    monkeypatch.setattr(server_identity, "_jetson_serial", lambda: None)
    server_identity._generated_server_id.cache_clear()
    yield
    server_identity._generated_server_id.cache_clear()


def test_configured_id_is_returned_verbatim(monkeypatch):
    monkeypatch.setattr(configuration, "INFERENCE_SERVER_ID", "srv-42")

    assert server_identity.get_inference_server_id() == "srv-42"


def test_gpu_suffix(monkeypatch):
    monkeypatch.setattr(server_identity, "_gpu_index", lambda: 0)

    assert re.fullmatch(
        RANDOM_PART + "-GPU-0", server_identity.get_inference_server_id()
    )


def test_jetson_suffix_when_serial_is_readable(monkeypatch):
    monkeypatch.setattr(server_identity, "_jetson_serial", lambda: "1423abc")

    assert re.fullmatch(
        RANDOM_PART + "-JETSON-1423abc", server_identity.get_inference_server_id()
    )


def test_jetson_wins_over_gpu_because_torch_sees_the_integrated_gpu(monkeypatch):
    monkeypatch.setattr(server_identity, "_gpu_index", lambda: 0)
    monkeypatch.setattr(server_identity, "_jetson_serial", lambda: "1423abc")

    assert server_identity.get_inference_server_id().endswith("-JETSON-1423abc")


def test_bare_random_part_without_hardware():
    assert re.fullmatch(RANDOM_PART, server_identity.get_inference_server_id())


def test_any_failure_gives_unknown(monkeypatch):
    def boom():
        raise RuntimeError("no")

    monkeypatch.setattr(server_identity, "_gpu_index", boom)

    assert server_identity.get_inference_server_id() == "UNKNOWN"


def test_generated_id_is_stable_within_a_process(monkeypatch):
    first = server_identity.get_inference_server_id()
    monkeypatch.setattr(server_identity, "_gpu_index", lambda: 0)

    assert server_identity.get_inference_server_id() == first


def test_jetson_serial_is_read_from_the_device_tree(monkeypatch, tmp_path):
    serial = tmp_path / "serial-number"
    serial.write_text("1423abc\n")
    monkeypatch.setattr(server_identity, "JETSON_SERIAL_PATH", serial)

    assert REAL_JETSON_SERIAL() == "1423abc"


def test_missing_or_empty_device_tree_serial_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr(server_identity, "JETSON_SERIAL_PATH", tmp_path / "absent")
    assert REAL_JETSON_SERIAL() is None

    empty = tmp_path / "empty"
    empty.write_text("  \n")
    monkeypatch.setattr(server_identity, "JETSON_SERIAL_PATH", empty)
    assert REAL_JETSON_SERIAL() is None


def test_gpu_index_is_none_when_torch_cannot_be_imported(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    assert REAL_GPU_INDEX() is None


def test_gpu_index_follows_torch_cuda_availability(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert REAL_GPU_INDEX() == 0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert REAL_GPU_INDEX() is None
