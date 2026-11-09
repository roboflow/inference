"""Exercise the generic worker contract independently of any connector."""

import time
from pathlib import Path

import pytest
from roboflow_workflows.utils.subprocess_supervisor import run_in_subprocess


def test_result_callback_precedes_cleanup_and_cleanup_survives_invalid_result(
    tmp_path, monkeypatch
):
    (tmp_path / "worker_fixture.py").write_text(
        "import sys\nfrom pathlib import Path\nPath(sys.argv[1], 'result').write_text('ok')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    directories = []

    def read_result(directory):
        directories.append(directory)
        assert (directory / "result").read_text() == "ok"
        return "loaded"

    assert (
        run_in_subprocess(
            {},
            worker_module="worker_fixture",
            deadline=time.monotonic() + 5,
            read_result=read_result,
        )
        == "loaded"
    )
    assert not directories[-1].exists()

    def invalid_result(directory):
        read_result(directory)
        raise ValueError("invalid result")

    with pytest.raises(ValueError, match="invalid result"):
        run_in_subprocess(
            {},
            worker_module="worker_fixture",
            deadline=time.monotonic() + 5,
            read_result=invalid_result,
        )
    assert not directories[-1].exists()


@pytest.mark.parametrize(
    "report", ["not json", '{"message": ["wrong type"]}', "x" * 3000]
)
def test_malformed_or_oversized_worker_errors_are_bounded(
    tmp_path, monkeypatch, report
):
    (tmp_path / "worker_fixture.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        + f"Path(sys.argv[1], 'error.json').write_text({report!r})\nsys.exit(1)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="valid error report"):
        run_in_subprocess(
            {},
            worker_module="worker_fixture",
            deadline=time.monotonic() + 5,
            read_result=lambda _: pytest.fail("read failed worker"),
        )
