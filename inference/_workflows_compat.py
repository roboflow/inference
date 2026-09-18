"""Legacy import compatibility for the moved Workflows package.

The Workflows source moved out of ``inference.core.workflows`` and
``inference.enterprise.workflows.enterprise_blocks`` into the standalone
``roboflow_workflows`` distribution. This module keeps every OLD dotted import
resolving to the SAME module object as the canonical name.

Rules (see MOVE_WORKFLOWS_PLAN.MD Phase D):

- One meta-path finder, registered exactly once. Detection uses a stable
  marker attribute on the finder instance so ``importlib.reload`` of this
  module does not double-register (a fresh class identity would otherwise
  slip past a ``isinstance`` guard).
- Legacy prefix mapping is gated by the BASELINE inventory: only paths that
  existed under ``inference.*`` before the move are aliased. A canonical
  subtree added later (e.g. ``roboflow_workflows.enterprise_blocks`` reached
  under the historically absent ``inference.core.workflows.enterprise_blocks``)
  must raise ``ModuleNotFoundError`` when resolved as a dotted import.
- The inventory gate applies to *dotted module resolution* only. Because an
  aliased legacy package IS its canonical module, an already-imported
  canonical child is visible as an attribute on the aliased parent — i.e.
  ``from inference.core.workflows import <child>`` succeeds when
  ``roboflow_workflows.<child>`` is already loaded. This shared-attribute
  exposure is an accepted consequence of preserving module identity and
  monkeypatch behavior (see MOVE_WORKFLOWS_PLAN.MD Phase D). Do not add
  proxy packages, global import hooks, attribute deletion, or caller-frame
  tricks to hide it.
- ``sys.modules[legacy] is sys.modules[canonical]`` after the alias so
  module-level state, caches, monkeypatches and configuration singletons stay
  singular. The loader replaces its temporary legacy module in ``sys.modules``
  after importing the canonical module, leaving canonical metadata untouched.
- Aliasing an inventoried ENTERPRISE module triggers ``import inference.core``
  first so the server env / configuration installation runs before any
  enterprise sink freezes standalone security defaults (P0 bootstrap gate).
- The empty legacy ``inference.enterprise.workflows`` root is intentionally
  absent from the map: aliasing it to the canonical root would expose new
  core APIs under an enterprise dotted path.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
from typing import Optional, Tuple

from inference._workflows_compat_inventory import (
    _ENTERPRISE_MODULES,
    _INVENTORY_LEGACY,
    _INVENTORY_PACKAGES,
)

# Longest-prefix first so the enterprise subtree wins the match.
_PREFIX_MAP: Tuple[Tuple[str, str], ...] = (
    (
        "inference.enterprise.workflows.enterprise_blocks",
        "roboflow_workflows.enterprise_blocks",
    ),
    ("inference.core.workflows", "roboflow_workflows"),
)

_FINDER_MARKER = "_roboflow_workflows_compat_finder"
_INSTALL_LOCK = threading.RLock()


def _under_legacy_prefix(fullname: str) -> bool:
    for legacy, _ in _PREFIX_MAP:
        if fullname == legacy or fullname.startswith(legacy + "."):
            return True
    return False


def _canonical_name(fullname: str) -> Optional[str]:
    for legacy, canonical in _PREFIX_MAP:
        if fullname == legacy:
            return canonical
        if fullname.startswith(legacy + "."):
            return canonical + fullname[len(legacy) :]
    return None


def _bootstrap_for_enterprise(legacy_name: str) -> None:
    # Import inference.core FIRST for any enterprise module: it wires
    # `install_workflows_configuration()` (see inference/core/__init__.py).
    # Without it, a direct `import inference.enterprise.workflows.enterprise_blocks.*`
    # loads sinks that would freeze standalone defaults (permissive PostgreSQL
    # policy, LAMBDA=False, etc.) despite restrictive server env.
    if legacy_name not in _ENTERPRISE_MODULES:
        return
    # Let importlib wait for another thread's in-progress core initialization.
    # Presence in sys.modules alone does not mean configuration is installed.
    importlib.import_module("inference.core")


class _WorkflowsCompatLoader(importlib.abc.Loader):
    """Replace the temporary legacy module without altering canonical metadata."""

    def __init__(self, legacy_name: str, canonical_name: str) -> None:
        self._legacy_name = legacy_name
        self._canonical_name = canonical_name

    def create_module(self, spec):  # type: ignore[override]
        return None

    def exec_module(self, module) -> None:  # type: ignore[override]
        _bootstrap_for_enterprise(self._legacy_name)
        sys.modules[self._legacy_name] = importlib.import_module(self._canonical_name)


class _WorkflowsCompatFinder(importlib.abc.MetaPathFinder):
    # Stable duck-typed marker: survives reload of this module (which would
    # otherwise mint a new class object and make `isinstance` checks fail).
    _roboflow_workflows_compat_finder = True

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[override]
        if not _under_legacy_prefix(fullname):
            return None
        if fullname not in _INVENTORY_LEGACY:
            # Baseline gate. Raising (rather than returning None) is deliberate:
            # `sys.modules[legacy_parent]` is the aliased canonical module, so
            # PathFinder would happily discover a canonical subpackage under
            # the aliased parent's `__path__` and load it under the historic
            # dotted name we want to leave dead.
            raise ModuleNotFoundError(
                f"No module named {fullname!r}",
                name=fullname,
            )
        canonical = _canonical_name(fullname)
        if canonical is None:  # pragma: no cover - inventory + prefix guarantee this
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _WorkflowsCompatLoader(fullname, canonical),
            origin="workflows-compat",
            is_package=fullname in _INVENTORY_PACKAGES,
        )


def _existing_finder_index() -> int:
    for i, finder in enumerate(sys.meta_path):
        if getattr(finder, _FINDER_MARKER, False):
            return i
    return -1


def install() -> None:
    """Register the compat finder. Idempotent and reload-safe."""
    with _INSTALL_LOCK:
        if _existing_finder_index() >= 0:
            return
        sys.meta_path.insert(0, _WorkflowsCompatFinder())


def alias_legacy_root(legacy_name: str) -> None:
    """Point ``sys.modules[legacy_name]`` at its canonical module.

    Called from the legacy package ``__init__.py`` after Python's default
    loader has placed a stub in ``sys.modules``. Replacing that stub keeps
    identity singular from the first import onward and prevents the stub
    from lingering as a distinct object.
    """
    install()
    if legacy_name not in _INVENTORY_LEGACY:
        raise ImportError(f"{legacy_name!r} is not a baseline legacy Workflows path")
    canonical = _canonical_name(legacy_name)
    if canonical is None:
        raise ImportError(f"{legacy_name!r} is not a mapped legacy Workflows prefix")
    _bootstrap_for_enterprise(legacy_name)
    canonical_module = importlib.import_module(canonical)
    sys.modules[legacy_name] = canonical_module
    parent_name, _, attr = legacy_name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attr, canonical_module)
