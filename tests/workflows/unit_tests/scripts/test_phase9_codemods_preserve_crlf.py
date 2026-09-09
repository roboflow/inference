"""Fix round 1, F2/F3: both Phase 9 codemods must round-trip CRLF files verbatim
and resolve call/construction origins correctly (aliased imports; untouched-module
same-named classes).

`Path.read_text()` performs universal-newline translation, so the scripts' own
`"\r\n" in source` detection was always False even on a genuinely CRLF file - the
file would silently come back out as LF. These tests write real CRLF fixtures to
disk, run the codemods exactly as `main()` does (`read_bytes().decode("utf-8")`
in, `open(..., newline="")` out), and assert every line ending survives.
"""

import ast

import scripts.phase9_platform_client as platform_client_codemod
import scripts.phase9_platform_client_tests as test_codemod


def _crlf(text: str) -> str:
    return text.replace("\n", "\r\n")


def _assert_all_crlf(text: str) -> None:
    assert "\r\n" in text, "fixture lost its CRLF endings entirely"
    assert text.count("\n") == text.count("\r\n"), "a bare \\n survived the codemod"


MODULE_SOURCE = _crlf(
    "from typing import Optional\n"
    "\n"
    "\n"
    "def helper(roboflow_api_key: Optional[str]) -> None:\n"
    "    inner(roboflow_api_key=roboflow_api_key)\n"
    "\n"
    "\n"
    "def inner(roboflow_api_key: Optional[str]) -> None:\n"
    "    pass\n"
    "\n"
    "\n"
    "class Block:\n"
    "    def __init__(self, api_key: Optional[str]):\n"
    "        self._roboflow_api_key = api_key\n"
    "\n"
    "    @classmethod\n"
    "    def get_init_parameters(cls):\n"
    '        return ["api_key"]\n'
)


def test_platform_client_transform_preserves_crlf_line_endings(tmp_path):
    path = tmp_path / "module.py"
    path.write_bytes(MODULE_SOURCE.encode("utf-8"))
    source = path.read_bytes().decode("utf-8")

    updated, stats = platform_client_codemod.transform(source, "Block", str(path))

    assert stats["defs"] == 2
    assert stats["calls"] == 1
    assert stats["ctor"] == 1
    assert stats["gip"] == 1
    _assert_all_crlf(updated)
    ast.parse(updated)  # still parses


TEST_SOURCE = _crlf(
    "import pytest\n"
    "\n"
    "from fake.touched.module import execute_claude_request as aliased_call\n"
    "from fake.untouched.module import SomeBlock\n"
    "\n"
    "\n"
    "def test_aliased_call_gets_client():\n"
    '    aliased_call(roboflow_api_key="y")\n'
    "\n"
    "\n"
    "def test_construction_from_untouched_module_is_left_alone():\n"
    '    SomeBlock(some_kwarg="z")\n'
)


def test_test_codemod_preserves_crlf_and_resolves_origins_correctly(tmp_path):
    path = tmp_path / "test_fixture.py"
    path.write_bytes(TEST_SOURCE.encode("utf-8"))

    calls, ctors = test_codemod.patch(
        path, touched={"fake.touched.module"}, classes={"SomeBlock"}
    )

    # F3(a): the alias's ORIGINAL imported name ("execute_claude_request") is in
    # CHAIN, not the local alias ("aliased_call") - the call must still be edited.
    assert calls == 1
    # F3(b): SomeBlock matches by name but its origin module ("fake.untouched.module")
    # is not in `touched` - the construction must be left alone.
    assert ctors == 0

    updated = path.read_bytes().decode("utf-8")
    assert "platform_client=platform_client" in updated
    assert 'SomeBlock(some_kwarg="z")' in updated  # untouched: byte-for-byte unchanged
    _assert_all_crlf(updated)
    ast.parse(updated)  # still parses
