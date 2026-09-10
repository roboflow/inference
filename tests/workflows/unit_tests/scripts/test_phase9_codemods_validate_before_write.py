"""Phase 9 final fix wave, F1: `phase9_endpoint_type_constant.py`,
`phase9_platform_client_tests.py` and `phase9_platform_errors_imports.py` used
to write each file as soon as its own `patch()` ran, and only check the
aggregate `--expected*` totals in `main()` afterwards - a wrong count left
earlier files rewritten with a non-zero exit code. All three now compute every
file's rewrite in memory, validate the aggregate counts, and only then write -
matching `scripts/phase9_move_roboflow_plugin.py` and
`scripts/phase9_platform_client.py`.

Each script gets a wrong-count case (asserts nothing is written) and a
happy-path case (asserts the file changed), driven through `main()` exactly
as the CLI is: `subprocess.run([sys.executable, script, ...])`.
"""

import subprocess
import sys
from pathlib import Path

import scripts.phase9_endpoint_type_constant as endpoint_codemod
import scripts.phase9_platform_client_tests as client_tests_codemod
import scripts.phase9_platform_errors_imports as errors_imports_codemod


def _run(script_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
    )


# --- phase9_endpoint_type_constant.py -------------------------------------

ENDPOINT_SOURCE = (
    "from inference.core.roboflow_api import ModelEndpointType\n"
    "\n"
    "\n"
    "def endpoint_type():\n"
    "    return ModelEndpointType.CORE_MODEL\n"
)


def test_endpoint_codemod_wrong_expected_count_writes_nothing(tmp_path):
    target = tmp_path / "module.py"
    target.write_text(ENDPOINT_SOURCE, encoding="utf-8")
    before = target.read_bytes()

    result = _run(
        Path(endpoint_codemod.__file__),
        str(target),
        "--expected-imports",
        "0",  # wrong: the fixture has 1
        "--expected-usages",
        "1",
    )

    assert result.returncode != 0
    assert target.read_bytes() == before


def test_endpoint_codemod_happy_path_writes_and_exits_zero(tmp_path):
    target = tmp_path / "module.py"
    target.write_text(ENDPOINT_SOURCE, encoding="utf-8")
    before = target.read_bytes()

    result = _run(
        Path(endpoint_codemod.__file__),
        str(target),
        "--expected-imports",
        "1",
        "--expected-usages",
        "1",
    )

    assert result.returncode == 0
    after = target.read_bytes()
    assert after != before
    assert b"ModelEndpointType" not in after
    assert b"CORE_MODEL_ENDPOINT_TYPE" in after


def test_endpoint_codemod_second_run_is_a_no_op(tmp_path):
    target = tmp_path / "module.py"
    target.write_text(ENDPOINT_SOURCE, encoding="utf-8")
    _run(
        Path(endpoint_codemod.__file__),
        str(target),
        "--expected-imports",
        "1",
        "--expected-usages",
        "1",
    )
    migrated = target.read_bytes()

    result = _run(
        Path(endpoint_codemod.__file__),
        str(target),
        "--expected-imports",
        "0",
        "--expected-usages",
        "0",
    )

    assert result.returncode == 0
    assert "SKIP (already repointed)" in result.stdout
    assert target.read_bytes() == migrated


# --- phase9_platform_errors_imports.py -------------------------------------

ERRORS_SOURCE = (
    "from inference.core.exceptions import RoboflowAPIRequestError\n"
    "\n"
    "\n"
    "def use():\n"
    "    raise RoboflowAPIRequestError()\n"
)


def test_errors_imports_codemod_wrong_expected_count_writes_nothing(tmp_path):
    target = tmp_path / "module.py"
    target.write_text(ERRORS_SOURCE, encoding="utf-8")
    before = target.read_bytes()

    result = _run(
        Path(errors_imports_codemod.__file__),
        str(target),
        "--expected",
        "0",  # wrong: the fixture has 1
    )

    assert result.returncode != 0
    assert target.read_bytes() == before


def test_errors_imports_codemod_happy_path_writes_and_exits_zero(tmp_path):
    target = tmp_path / "module.py"
    target.write_text(ERRORS_SOURCE, encoding="utf-8")
    before = target.read_bytes()

    result = _run(
        Path(errors_imports_codemod.__file__),
        str(target),
        "--expected",
        "1",
    )

    assert result.returncode == 0
    after = target.read_bytes()
    assert after != before
    assert b"inference.core.exceptions" not in after
    assert b"inference.core.workflows.prototypes.platform_errors" in after


def test_errors_imports_codemod_second_run_is_a_no_op(tmp_path):
    target = tmp_path / "module.py"
    target.write_text(ERRORS_SOURCE, encoding="utf-8")
    _run(Path(errors_imports_codemod.__file__), str(target), "--expected", "1")
    migrated = target.read_bytes()

    result = _run(Path(errors_imports_codemod.__file__), str(target), "--expected", "0")

    assert result.returncode == 0
    assert "SKIP (already repointed)" in result.stdout
    assert target.read_bytes() == migrated


# --- phase9_platform_client_tests.py ----------------------------------------

CLIENT_TEST_SOURCE = (
    "import pytest\n"
    "\n"
    "from fake.touched.module import execute_claude_request as aliased_call\n"
    "\n"
    "\n"
    "def test_aliased_call_gets_client():\n"
    '    aliased_call(roboflow_api_key="y")\n'
)


def _touched_file(tmp_path: Path) -> Path:
    touched = tmp_path / "touched.txt"
    touched.write_text("fake/touched/module.py\n", encoding="utf-8")
    return touched


def test_client_tests_codemod_wrong_expected_count_writes_nothing(tmp_path):
    target = tmp_path / "test_fixture.py"
    target.write_text(CLIENT_TEST_SOURCE, encoding="utf-8")
    before = target.read_bytes()
    touched = _touched_file(tmp_path)

    result = _run(
        Path(client_tests_codemod.__file__),
        str(target),
        "--touched",
        str(touched),
        "--classes",
        "SomeUnusedClass",
        "--expected-calls",
        "0",  # wrong: the fixture has 1
        "--expected-constructions",
        "0",
    )

    assert result.returncode != 0
    assert target.read_bytes() == before


def test_client_tests_codemod_happy_path_writes_and_exits_zero(tmp_path):
    target = tmp_path / "test_fixture.py"
    target.write_text(CLIENT_TEST_SOURCE, encoding="utf-8")
    before = target.read_bytes()
    touched = _touched_file(tmp_path)

    result = _run(
        Path(client_tests_codemod.__file__),
        str(target),
        "--touched",
        str(touched),
        "--classes",
        "SomeUnusedClass",
        "--expected-calls",
        "1",
        "--expected-constructions",
        "0",
    )

    assert result.returncode == 0
    after = target.read_bytes()
    assert after != before
    assert b"platform_client=platform_client" in after


def test_client_tests_codemod_second_run_is_a_no_op(tmp_path):
    target = tmp_path / "test_fixture.py"
    target.write_text(CLIENT_TEST_SOURCE, encoding="utf-8")
    touched = _touched_file(tmp_path)
    _run(
        Path(client_tests_codemod.__file__),
        str(target),
        "--touched",
        str(touched),
        "--classes",
        "SomeUnusedClass",
        "--expected-calls",
        "1",
        "--expected-constructions",
        "0",
    )
    migrated = target.read_bytes()

    result = _run(
        Path(client_tests_codemod.__file__),
        str(target),
        "--touched",
        str(touched),
        "--classes",
        "SomeUnusedClass",
        "--expected-calls",
        "0",
        "--expected-constructions",
        "0",
    )

    assert result.returncode == 0
    assert "TOTAL 0 helper calls, 0 block constructions edited" in result.stdout
    assert target.read_bytes() == migrated
