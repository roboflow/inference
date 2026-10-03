import json
import logging
import os
from pathlib import Path

import pytest

from inference_server import configuration
from inference_server.workflows import definition_cache

TENANT_K = "c69dcfbddd7fcd94c54e3bd861c6b991bb3676c1b6dad1b2573e2d913e3ac531"
TENANT_ANONYMOUS = "cedf8cfe6a81077bee33fc8f55a395e10be4a9eed0a656f786f4898af02521f7"
VERSIONED_STEM = "wf_d770916e7e0da1d842b6bb3a5e356d98823929ba0e6313dfbe76629a74a9e785"


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(configuration, "SINGLE_TENANT_WORKFLOW_CACHE", False)
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", False)
    return tmp_path / "cache" / "workflow"


@pytest.fixture
def single_tenant(cache_root, monkeypatch):
    monkeypatch.setattr(configuration, "SINGLE_TENANT_WORKFLOW_CACHE", True)
    return cache_root


@pytest.fixture
def offline(single_tenant, monkeypatch):
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", True)
    return single_tenant


@pytest.fixture
def case_sensitive_lookup(monkeypatch):
    monkeypatch.setattr(definition_cache, "_has_exact_case", lambda path: path.exists())


def _response(tag):
    return {
        "workflow": {
            "id": tag,
            "config": json.dumps({"specification": {"tag": tag}}),
        }
    }


def _put(root, relative, tag):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_response(tag)))
    return path


def _load(api_key="k", version=None):
    response = definition_cache.load_definition(
        "ws", "wf", api_key=api_key, workflow_version_id=version
    )
    return None if response is None else response["workflow"]["id"]


def _path(api_key="k", version=None, workspace="ws", workflow="wf"):
    return definition_cache.cache_file_path(
        workspace, workflow, api_key=api_key, workflow_version_id=version
    )


def test_multi_tenant_unversioned_with_key_path(cache_root):
    assert _path() == cache_root / "ws" / ".tenanted-v2" / f"wf_{TENANT_K}.json"
    _put(cache_root, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "tenant-k")
    assert _load() == "tenant-k"


@pytest.mark.parametrize("api_key", [None, "", "local"])
def test_multi_tenant_unversioned_without_key_path(cache_root, api_key):
    expected = cache_root / "ws" / ".tenanted-v2" / f"wf_{TENANT_ANONYMOUS}.json"
    assert _path(api_key=api_key) == expected
    _put(cache_root, f"ws/.tenanted-v2/wf_{TENANT_ANONYMOUS}.json", "anonymous")
    assert _load(api_key=api_key) == "anonymous"


def test_multi_tenant_versioned_with_key_path(cache_root):
    expected = (
        cache_root
        / "ws"
        / ".tenanted-v2"
        / "versions"
        / f"{VERSIONED_STEM}_{TENANT_K}.json"
    )
    assert _path(version="3") == expected
    _put(cache_root, f"ws/.tenanted-v2/versions/{VERSIONED_STEM}_{TENANT_K}.json", "v")
    assert _load(version="3") == "v"


def test_multi_tenant_versioned_without_key_path(cache_root):
    expected = (
        cache_root
        / "ws"
        / ".tenanted-v2"
        / "versions"
        / f"{VERSIONED_STEM}_{TENANT_ANONYMOUS}.json"
    )
    assert _path(api_key=None, version="3") == expected
    _put(
        cache_root,
        f"ws/.tenanted-v2/versions/{VERSIONED_STEM}_{TENANT_ANONYMOUS}.json",
        "anonymous-v",
    )
    assert _load(api_key=None, version="3") == "anonymous-v"


def test_multi_tenant_reads_neither_canonical_nor_oldest_layout(
    cache_root, case_sensitive_lookup
):
    _put(cache_root, "ws/.canonical-v2/wf.json", "canonical")
    _put(cache_root, "ws/wf.json", "oldest")
    _put(cache_root, f"ws/.tenanted-v2/wf_{TENANT_ANONYMOUS}.json", "other-tenant")

    assert _load() is None


def test_empty_version_means_latest(cache_root):
    assert _path(version="") == _path(version=None)


def test_single_tenant_unversioned_order(single_tenant, case_sensitive_lookup):
    assert _path() == single_tenant / "ws" / ".canonical-v2" / "wf.json"
    _put(single_tenant, "ws/wf.json", "oldest")
    _put(single_tenant, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "tenant-k")
    assert _load() == "oldest"

    _put(single_tenant, "ws/.canonical-v2/wf.json", "canonical")
    assert _load() == "canonical"


def test_single_tenant_versioned_has_no_fallback(single_tenant, case_sensitive_lookup):
    expected = (
        single_tenant / "ws" / ".canonical-v2" / "versions" / f"{VERSIONED_STEM}.json"
    )
    assert _path(version="3") == expected
    _put(single_tenant, "ws/wf.json", "oldest")
    _put(
        single_tenant, f"ws/.tenanted-v2/versions/{VERSIONED_STEM}_{TENANT_K}.json", "t"
    )
    assert _load(version="3") is None

    _put(single_tenant, f"ws/.canonical-v2/versions/{VERSIONED_STEM}.json", "canonical")
    assert _load(version="3") == "canonical"


def test_offline_unversioned_with_key_order(offline, case_sensitive_lookup):
    assert _path() == offline / "ws" / ".canonical-v2" / "wf.json"
    _put(offline, "ws/wf.json", "oldest")
    assert _load() == "oldest"

    _put(offline, f"ws/.tenanted-v2/wf_{TENANT_ANONYMOUS}.json", "anonymous")
    assert _load() == "oldest"

    _put(offline, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "tenant-k")
    assert _load() == "tenant-k"

    _put(offline, "ws/.canonical-v2/wf.json", "canonical")
    assert _load() == "canonical"


def test_offline_unversioned_without_key_takes_the_sole_hashed_entry(
    offline, case_sensitive_lookup, caplog
):
    _put(offline, "ws/wf.json", "oldest")
    _put(offline, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "tenant-k")
    assert _load(api_key=None) == "tenant-k"

    _put(offline, f"ws/.tenanted-v2/wf_{'0' * 64}.json", "other")
    with caplog.at_level(logging.WARNING):
        assert _load(api_key=None) == "oldest"
    assert "Cannot choose among 2 hashed offline Workflow cache entries" in caplog.text


def test_offline_hashed_entry_must_match_the_key_exactly(offline):
    _put(offline, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "tenant-k")

    assert _load(api_key="other") is None


def test_offline_versioned_with_key_order(offline, case_sensitive_lookup):
    _put(offline, "ws/wf.json", "oldest")
    assert _load(version="3") is None

    _put(offline, f"ws/.tenanted-v2/versions/{VERSIONED_STEM}_{TENANT_K}.json", "t")
    assert _load(version="3") == "t"

    _put(offline, f"ws/.canonical-v2/versions/{VERSIONED_STEM}.json", "canonical")
    assert _load(version="3") == "canonical"


def test_offline_versioned_without_key_takes_the_sole_hashed_entry(offline):
    _put(offline, f"ws/.tenanted-v2/versions/{VERSIONED_STEM}_{TENANT_K}.json", "t")

    assert _load(api_key=None, version="3") == "t"


def test_oldest_layout_is_refused_on_a_case_insensitive_filesystem(single_tenant):
    _put(single_tenant, "ws/wf.json", "oldest")
    probe = single_tenant / "ws" / "probe"
    probe.write_text("")
    case_insensitive = (single_tenant / "ws" / "PROBE").exists()

    assert _load() == (None if case_insensitive else "oldest")


def test_oldest_layout_requires_lowercase_canonical_ids(
    single_tenant, case_sensitive_lookup
):
    _put(single_tenant, "ws_1/wf.json", "oldest")

    assert (
        definition_cache.load_definition(
            "ws_1", "wf", api_key="k", workflow_version_id=None
        )
        is None
    )


@pytest.mark.parametrize(
    "workspace,segment",
    [
        (
            "My Ws",
            "~My_Ws_cb9323aba50af4af265760b33ac8c3121267145614a5e8d74d5ffe77bdf3a4d6",
        ),
        (
            "con",
            "~con_fad4ef1880f54ccb7471daf2c0cb82cdb6eb8a6204d7fa3c8d7beccd29cad35f",
        ),
        ("", "~empty_1d5d5d4bd4c802b99428d7609c449522dd686f52568fcc87b35e43a3d5241af1"),
        (
            "a" * 97,
            f"~{'a' * 48}_82a498b340b062a3c0536ee058686b1dea2068b496ab485cd204202c0234b50a",
        ),
        ("a" * 96, "a" * 96),
        ("..", "~___47dfe32e0882feb6884562ca7b46703b2b158795ea503e33817902f0906d4862"),
        ("wf_v1", "wf_v1"),
    ],
)
def test_workspace_segment_matches_legacy(cache_root, workspace, segment):
    assert _path(workspace=workspace).parent.parent == cache_root / segment


def test_workflow_segment_rejects_legacy_filename_shapes(cache_root):
    fingerprinted = (
        "~wf_v1_97bb62dd79388ca67690ee26d4339fad474a21d73f688caa1b02a49a3f0e07ab"
    )
    assert _path(workflow="wf_v1").name == f"{fingerprinted}_{TENANT_K}.json"
    assert _path(workflow="wf_v").name == f"wf_v_{TENANT_K}.json"
    assert _path(workflow=f"wf_{'a' * 64}").name.startswith("~wf_")
    assert _path(workflow="a" * 97).name.startswith(f"~{'a' * 48}_")


@pytest.mark.parametrize(
    "value",
    ["..", "../..", "/etc", "/", "a/b", "a\0b", "x" * 5000, "..\\..", "."],
)
@pytest.mark.parametrize("field", ["workspace", "workflow", "version"])
def test_request_values_cannot_escape_the_cache_root(cache_root, field, value):
    arguments = {"workspace": "ws", "workflow": "wf", "version": "3"}
    arguments[field] = value

    path = _path(
        workspace=arguments["workspace"],
        workflow=arguments["workflow"],
        version=arguments["version"],
    )

    assert cache_root in path.parents
    assert all(len(part.encode()) < 255 for part in path.relative_to(cache_root).parts)
    assert "\0" not in str(path)


def test_escape_check_refuses_an_unsanitised_segment(cache_root, monkeypatch):
    monkeypatch.setattr(definition_cache, "_path_segment", lambda value, **_: value)

    with pytest.raises(ValueError, match="insecure location"):
        _path(workspace="../../outside")


def test_non_string_version_is_a_type_error(cache_root):
    with pytest.raises(TypeError):
        _path(version=3)


def test_store_writes_the_legacy_path_atomically(cache_root, monkeypatch):
    seen = {}
    original_replace = os.replace

    def _replace(source, destination):
        seen["source"] = Path(source)
        seen["destination"] = Path(destination)
        original_replace(source, destination)

    monkeypatch.setattr(definition_cache.os, "replace", _replace)

    definition_cache.store_definition(
        "ws", "wf", api_key="k", workflow_version_id=None, response=_response("live")
    )

    target = cache_root / "ws" / ".tenanted-v2" / f"wf_{TENANT_K}.json"
    assert json.loads(target.read_text()) == _response("live")
    assert seen["destination"] == target
    assert seen["source"].parent == target.parent
    assert seen["source"].name.startswith(".workflow.")
    assert list(target.parent.iterdir()) == [target]


def test_partial_write_never_reaches_the_cache_file(cache_root, monkeypatch):
    target = _put(cache_root, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "previous")

    def _partial_dump(response, file_handle):
        file_handle.write('{"workflow": {"id": "partial"')
        raise OSError("disk full")

    monkeypatch.setattr(definition_cache.json, "dump", _partial_dump)
    definition_cache.store_definition(
        "ws", "wf", api_key="k", workflow_version_id=None, response=_response("live")
    )

    assert json.loads(target.read_text()) == _response("previous")
    assert list(target.parent.iterdir()) == [target]
    assert _load() == "previous"


def test_store_failure_is_logged_without_the_path_and_does_not_raise(
    cache_root, caplog
):
    cache_root.parent.mkdir(parents=True)
    cache_root.write_text("not a directory")

    with caplog.at_level(logging.WARNING):
        definition_cache.store_definition(
            "ws", "wf", api_key="k", workflow_version_id=None, response=_response("x")
        )

    assert len(caplog.records) == 1
    assert "Could not write the Workflow definition cache file" in caplog.text
    assert str(cache_root) not in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        {"workflow": {"config": {"specification": {}}}},
        {"workflow": {"config": json.dumps({"specification": []})}},
        {"workflow": {"config": "not json"}},
        {"workflow": "x"},
        [],
    ],
)
def test_malformed_response_is_not_stored(cache_root, caplog, response):
    with caplog.at_level(logging.WARNING):
        definition_cache.store_definition(
            "ws", "wf", api_key="k", workflow_version_id=None, response=response
        )

    assert not cache_root.exists()
    assert "Refusing to cache a malformed Workflow response" in caplog.text


def test_malformed_cached_file_is_a_miss(cache_root):
    path = _put(cache_root, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "x")
    path.write_text("{not json")

    assert _load() is None


def test_oversized_cached_file_is_a_miss_with_a_warning(
    cache_root, monkeypatch, caplog
):
    monkeypatch.setattr(definition_cache, "MAX_CACHED_DEFINITION_BYTES", 64)
    path = _put(cache_root, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "x" * 100)

    with caplog.at_level(logging.WARNING):
        assert _load() is None

    assert path.stat().st_size > 64
    assert "Ignoring an oversized Workflow cache file" in caplog.text


def test_symlinked_cached_file_is_a_miss(cache_root, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_response("outside")))
    link = cache_root / "ws" / ".tenanted-v2" / f"wf_{TENANT_K}.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    assert _load() is None


def test_file_that_grows_past_the_bound_after_the_size_check_is_a_miss(
    cache_root, monkeypatch, caplog
):
    monkeypatch.setattr(definition_cache, "MAX_CACHED_DEFINITION_BYTES", 64)
    path = _put(cache_root, f"ws/.tenanted-v2/wf_{TENANT_K}.json", "x")
    path.write_text(json.dumps(_response("grown")))
    assert path.stat().st_size > 64
    real_fstat = os.fstat

    class _SmallStat:
        def __init__(self, status):
            self._status = status

        def __getattr__(self, name):
            return 1 if name == "st_size" else getattr(self._status, name)

    monkeypatch.setattr(os, "fstat", lambda fd: _SmallStat(real_fstat(fd)))

    with caplog.at_level(logging.WARNING):
        assert _load() is None

    assert "Ignoring an oversized Workflow cache file" in caplog.text
