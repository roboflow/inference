"""Legacy Workflows import compatibility tests.

Covers Phase D of the Workflows extraction (see MOVE_WORKFLOWS_PLAN.MD).
The tests are HARD failures — the `roboflow_workflows` package is a runtime
dependency of `inference`, so an ``ImportError`` here is a real regression,
not an environment gap. Baseline inventory lives next to this file
(`workflows_compat_inventory.json`); it is FROZEN, not regenerated from the
canonical tree, so a subpackage added to `roboflow_workflows` after the move
can never leak in under a historic dotted name.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from inference._workflows_compat_inventory import (
    _ENTERPRISE_MODULES,
    _INVENTORY_LEGACY,
    _INVENTORY_PACKAGES,
)

_INVENTORY_PATH = Path(__file__).with_name("workflows_compat_inventory.json")
_INVENTORY = json.loads(_INVENTORY_PATH.read_text())


def _iter_pairs(kind: str):
    for entry in _INVENTORY[kind]:
        yield entry["legacy"], entry["canonical"]


def test_inventory_baseline_matches_runtime_module() -> None:
    # Runtime finder and the baseline JSON must agree. Regenerating either
    # from a future `roboflow_workflows` tree would leak new APIs under the
    # historic prefix; regenerating from the JSON only in isolation would
    # split the source of truth.
    from_json = {e["legacy"] for e in _INVENTORY["packages"] + _INVENTORY["modules"]}
    assert from_json == set(_INVENTORY_LEGACY)
    assert _INVENTORY_PACKAGES == {entry["legacy"] for entry in _INVENTORY["packages"]}
    assert _ENTERPRISE_MODULES == {
        entry["legacy"]
        for entry in _INVENTORY["modules"]
        if entry["legacy"].startswith("inference.enterprise.")
    }


def test_inventory_pins_baseline_revision() -> None:
    assert _INVENTORY["generated_from_revision"] == (
        "1a9ce2c4218ef0d68c4ae3c1076a3ff3c5095387"
    )


def test_legacy_font_helper_exports_and_monkeypatches_work(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from build_scripts import download_fonts as helper

    assert helper.DEFAULT_TARGET_DIR == helper.FONTS_PACKAGE_DIR / "assets"
    assert "roboflow_workflows" in helper.FONTS_PACKAGE_DIR.parts
    calls = []

    def load_module(name, path):
        calls.append(path)
        return SimpleNamespace(FONTS_REGISTRY={})

    monkeypatch.setattr(helper, "_load_module_by_path", load_module)
    assert helper.download_fonts(tmp_path, only=["missing-font"]) == 1
    assert len(calls) == 2


@pytest.mark.parametrize(
    "legacy,canonical",
    list(_iter_pairs("packages")) + list(_iter_pairs("modules")),
    ids=lambda v: v,
)
def test_legacy_import_returns_canonical_module(legacy: str, canonical: str) -> None:
    canonical_module = importlib.import_module(canonical)
    legacy_module = importlib.import_module(legacy)
    assert legacy_module is canonical_module
    assert sys.modules[legacy] is sys.modules[canonical]
    # Only the temporary alias receives legacy metadata; the shared module
    # retains its canonical loader and spec, including on reload.
    assert legacy_module.__name__ == canonical
    assert legacy_module.__spec__ is not None
    assert legacy_module.__spec__.name == canonical
    assert legacy_module.__loader__ is not None


def test_empty_enterprise_root_does_not_alias_to_canonical() -> None:
    legacy_root = importlib.import_module("inference.enterprise.workflows")
    canonical_root = importlib.import_module("roboflow_workflows")
    assert legacy_root is not canonical_root


def test_missing_legacy_module_raises_module_not_found() -> None:
    # Baseline gate — even if the canonical tree grew a matching subpackage
    # (e.g. `roboflow_workflows.enterprise_blocks`) the historic dotted name
    # under `inference.core.workflows.enterprise_blocks` NEVER existed and
    # must not resolve.
    for missing in (
        "inference.core.workflows._definitely_not_a_real_module_",
        "inference.core.workflows.enterprise_blocks",
        "inference.core.workflows.enterprise_blocks.loader",
        "inference.enterprise.workflows.enterprise_blocks._nope_",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(missing)


def test_string_plugin_paths_still_resolve() -> None:
    canonical = importlib.import_module("roboflow_workflows.enterprise_blocks.loader")
    plugin = importlib.import_module(
        "inference.enterprise.workflows.enterprise_blocks.loader"
    )
    assert plugin is canonical


def test_legacy_import_preserves_class_identity_and_errors() -> None:
    from roboflow_workflows.errors import (
        WorkflowEnvironmentConfigurationError as canonical_error,
    )

    from inference.core.workflows.errors import (
        WorkflowEnvironmentConfigurationError as legacy_error,
    )

    assert legacy_error is canonical_error


def test_legacy_class_module_stays_canonical_for_pickling() -> None:
    # `class.__module__` is intentionally NOT rewritten. Standalone installs
    # of `roboflow_workflows` do not have `inference.*`, and pickle uses
    # `__module__` to re-import the class on load.
    from roboflow_workflows.errors import WorkflowError
    from roboflow_workflows.execution_engine.entities.base import WorkflowImageData

    assert WorkflowImageData.__module__.startswith("roboflow_workflows.")
    assert WorkflowError.__module__.startswith("roboflow_workflows.")
    # Class-level pickle round-trip goes via __module__/__qualname__: if the
    # class module were rewritten to `inference.core.workflows.*`, pickle in
    # a standalone install (no `inference.*` present) would raise on load.
    assert pickle.loads(pickle.dumps(WorkflowError)) is WorkflowError
    assert (
        pickle.loads(b"cinference.core.workflows.errors\nWorkflowError\n.")
        is WorkflowError
    )


def test_historic_type_display_translates_module_path() -> None:
    from roboflow_workflows._compat_names import historic_type_display, to_legacy_module
    from roboflow_workflows.enterprise_blocks.sinks.postgresql.v1 import (
        PostgreSQLSinkBlockV1,
    )
    from roboflow_workflows.execution_engine.entities.base import WorkflowImageData

    core = historic_type_display(WorkflowImageData)
    assert core.startswith("inference.core.workflows.")
    assert core.endswith(".WorkflowImageData")

    ent = historic_type_display(PostgreSQLSinkBlockV1)
    assert ent.startswith(
        "inference.enterprise.workflows.enterprise_blocks.sinks.postgresql."
    )

    assert historic_type_display(int) == "builtins.int"

    # Non-workflow modules pass through untouched.
    class _Local:  # noqa: D401
        pass

    _Local.__module__ = "some.other.pkg"
    assert to_legacy_module("some.other.pkg") == "some.other.pkg"


def test_get_full_type_name_returns_historic_path() -> None:
    from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
    from roboflow_workflows.execution_engine.introspection.utils import (
        get_full_type_name,
    )

    name = get_full_type_name(selected_type=WorkflowImageData)
    assert name.startswith("inference.core.workflows.")
    assert name.endswith(".WorkflowImageData")


def test_legacy_logger_names_root_under_inference() -> None:
    # Every `get_logger(__name__)` in the moved tree resolves to a logger in
    # the `inference.*` tree, so handlers configured on `inference` in
    # `inference.core.logger` propagate as they did before the move.
    import roboflow_workflows.core_steps.common.serializers as serializers

    assert serializers.logger.name.startswith("inference.core.workflows.")

    import roboflow_workflows.enterprise_blocks.sinks.postgresql.v1 as pg

    assert pg.logger.name.startswith(
        "inference.enterprise.workflows.enterprise_blocks."
    )


def test_reload_after_finder_module_reload_stays_singular() -> None:
    # Reloading the compat module must NOT double-register the finder.
    # The reload-safety check keys off a stable marker attribute rather than
    # class identity, so a fresh class from the reload still matches.
    from inference import _workflows_compat as compat

    compat.install()
    before = sum(
        1
        for f in sys.meta_path
        if getattr(f, "_roboflow_workflows_compat_finder", False)
    )
    assert before == 1
    importlib.reload(compat)
    compat.install()
    after = sum(
        1
        for f in sys.meta_path
        if getattr(f, "_roboflow_workflows_compat_finder", False)
    )
    assert after == 1


def test_module_level_monkeypatch_survives_via_shared_identity(monkeypatch) -> None:
    import roboflow_workflows.configuration as canonical_cfg

    import inference.core.workflows.configuration as legacy_cfg

    sentinel = object()
    monkeypatch.setattr(legacy_cfg, "_PATCH_PROBE", sentinel, raising=False)
    assert getattr(canonical_cfg, "_PATCH_PROBE") is sentinel


def test_server_configuration_builder_includes_enterprise_policy() -> None:
    # Phase B parity — no configuration reset required: the builder does not
    # install anything, and the memoised server object stays valid because it
    # is byte-for-byte the same value on repeated env resolution.
    from inference.core import env
    from inference.core.interfaces import workflows_configuration

    cfg = workflows_configuration.build_configuration_from_env()

    assert (
        cfg.engine.allow_postgresql_sink_to_non_global_addresses
        is env.ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES
    )
    assert cfg.platform.lambda_runtime is env.LAMBDA
    if env.POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES is None:
        assert cfg.engine.postgresql_sink_blacklisted_addresses is None
    else:
        assert set(cfg.engine.postgresql_sink_blacklisted_addresses or ()) == set(
            env.POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES
        )
    if env.POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES is None:
        assert cfg.engine.postgresql_sink_whitelisted_addresses is None
    else:
        assert set(cfg.engine.postgresql_sink_whitelisted_addresses or ()) == set(
            env.POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES
        )


def test_environment_facade_exposes_enterprise_policy_types() -> None:
    from roboflow_workflows import environment as facade

    assert isinstance(
        facade.ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES, bool
    )
    assert isinstance(facade.LAMBDA, bool)
    for name in (
        "POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES",
        "POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES",
    ):
        value = getattr(facade, name)
        assert value is None or isinstance(value, set)


def test_postgres_sink_imports_from_workflows_facade_not_server() -> None:
    import roboflow_workflows

    sink_source = (
        Path(roboflow_workflows.__file__).parent
        / "enterprise_blocks"
        / "sinks"
        / "postgresql"
        / "v1.py"
    ).read_text()
    assert "from inference.core.env" not in sink_source
    assert "from inference.core.logger" not in sink_source
    assert "from roboflow_workflows.environment import" in sink_source


# ---------------------------------------------------------------------------
# Fresh-subprocess tests. Each spawns a clean interpreter to exercise a load
# order the parent test process cannot re-create (`inference.*` is already
# imported here, which would mask the P0 bootstrap gate).
# ---------------------------------------------------------------------------

_RUN_ROOT = Path(__file__).resolve().parents[3]


def _run_subprocess(source: str, env: dict) -> subprocess.CompletedProcess:
    child_env = os.environ.copy()
    child_env.update(env)
    # Force restrictive standalone defaults; server env re-normalises them.
    child_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=_RUN_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("suffix", ["loader", "sinks.postgresql.v1"])
def test_subprocess_direct_legacy_enterprise_import_triggers_server_bootstrap(
    suffix,
) -> None:
    # P0: importing an enterprise legacy submodule COLD must load inference.core
    # first so the server config wins over the standalone defaults. Restrictive
    # env vars set here must be visible to the sink module.
    env = {
        "ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES": "False",
        "POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES": "10.0.0.1,10.0.0.2",
        "POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES": "203.0.113.5",
        "LAMBDA": "True",
        "API_LOGGING_ENABLED": "False",
        "WORKFLOWS_COMPAT_TEST_MODULE": (
            "inference.enterprise.workflows.enterprise_blocks." + suffix
        ),
    }
    result = _run_subprocess(
        """
        import os, sys
        # Import ONLY the legacy enterprise path first — no `inference.core`
        # touched by hand. The compat shim must bootstrap the server.
        loader = __import__(
            os.environ["WORKFLOWS_COMPAT_TEST_MODULE"],
            fromlist=["_"],
        )
        assert "inference.core" in sys.modules, "server bootstrap did not run"
        from roboflow_workflows import environment as facade
        assert facade.ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES is False
        assert facade.LAMBDA is True
        assert facade.POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES == {
            "10.0.0.1", "10.0.0.2",
        }
        assert facade.POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES == {
            "203.0.113.5",
        }
        print("ok")
        """,
        env,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("ok")


def test_subprocess_canonical_first_then_legacy_shares_identity() -> None:
    result = _run_subprocess(
        """
        import sys, importlib
        canonical = importlib.import_module("roboflow_workflows.configuration")
        legacy = importlib.import_module("inference.core.workflows.configuration")
        assert legacy is canonical
        assert legacy.__spec__.name == "roboflow_workflows.configuration"
        print("ok")
        """,
        env={"API_LOGGING_ENABLED": "False"},
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_subprocess_legacy_first_then_canonical_shares_identity() -> None:
    result = _run_subprocess(
        """
        import sys, importlib
        legacy = importlib.import_module("inference.core.workflows.configuration")
        canonical = importlib.import_module("roboflow_workflows.configuration")
        assert legacy is canonical
        assert canonical.__spec__.name == "roboflow_workflows.configuration"
        print("ok")
        """,
        env={"API_LOGGING_ENABLED": "False"},
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_subprocess_empty_enterprise_packages_do_not_bootstrap_server() -> None:
    result = _run_subprocess(
        """
        import importlib.util, sys
        spec = importlib.util.find_spec(
            "inference.enterprise.workflows.enterprise_blocks.sinks.postgresql"
        )
        assert spec.submodule_search_locations is not None
        package = importlib.import_module(
            "inference.enterprise.workflows.enterprise_blocks.sinks.postgresql"
        )
        assert "inference.core" not in sys.modules
        assert package.__name__ == "roboflow_workflows.enterprise_blocks.sinks.postgresql"
        """,
        env={},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("first", ["inference.core.workflows", "roboflow_workflows"])
def test_subprocess_reload_parent_binding_and_string_patch(first) -> None:
    result = _run_subprocess(
        """
        import importlib, os, sys
        from unittest.mock import patch
        importlib.import_module(os.environ["WORKFLOWS_COMPAT_FIRST"] + ".errors")
        legacy = importlib.import_module("inference.core.workflows.errors")
        canonical = importlib.import_module("roboflow_workflows.errors")
        assert legacy is canonical
        assert importlib.reload(legacy) is canonical
        assert canonical.__spec__.name == "roboflow_workflows.errors"
        assert sys.modules["inference.core.workflows"].errors is canonical
        sentinel = object()
        with patch("inference.core.workflows.errors.WorkflowError", sentinel):
            assert canonical.WorkflowError is sentinel
        """,
        env={"API_LOGGING_ENABLED": "False", "WORKFLOWS_COMPAT_FIRST": first},
    )
    assert result.returncode == 0, result.stderr


def test_subprocess_standalone_defaults_conflict_with_server_install() -> None:
    # If a caller reads `get_configuration()` before the server installs its
    # own (which materialises the sticky default), a subsequent server install
    # with different values must fail with a clear message rather than silently
    # accepting an ignored value.
    result = _run_subprocess(
        """
        from roboflow_workflows import configuration as cfg
        stickied = cfg.get_configuration()
        assert stickied is cfg.default_configuration() or stickied == cfg.default_configuration()
        try:
            import inference.core
        except cfg.WorkflowEnvironmentConfigurationError:
            print("ok")
        else:
            raise SystemExit("server bootstrap did not refuse differing config")
        """,
        env={
            "API_LOGGING_ENABLED": "False",
            "ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES": "False",
        },
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
