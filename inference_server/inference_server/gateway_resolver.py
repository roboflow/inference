"""Gateway resolution — entry-point registry for the gateway duck surface.

GATEWAY_API_VERSION is the versioned seam external gateway factories assert
against. Any change to EXPECTED_GATEWAY_SIGNATURES (test_gateway_contract.py)
bumps this version.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import threading
from typing import Any, Callable, Dict

from inference_server import configuration as cfg

GATEWAY_API_VERSION = 1

GATEWAY_FACTORIES: Dict[str, Callable[[], Any]] = {}
_EPS_LOADED = False
_EPS_LOCK = threading.Lock()


def _build_direct_gateway() -> Any:
    from inference_model_manager.model_manager import ModelManager
    from inference_server.gateway import ModelManagerGateway

    return ModelManagerGateway(ModelManager())


GATEWAY_FACTORIES["direct"] = _build_direct_gateway


def _iter_gateway_entry_points():
    return md.entry_points(group="inference_server.gateway")


def _reset_entry_point_cache_for_tests() -> None:
    global _EPS_LOADED
    _EPS_LOADED = False


def resolve_gateway() -> Any:
    name = os.environ.get(cfg.INFERENCE_GATEWAY_ENV, cfg.INFERENCE_GATEWAY_DEFAULT)
    factory = GATEWAY_FACTORIES.get(name)
    if factory is None:
        global _EPS_LOADED
        if not _EPS_LOADED:
            with _EPS_LOCK:
                if not _EPS_LOADED:
                    for ep in _iter_gateway_entry_points():
                        if ep.name not in GATEWAY_FACTORIES:
                            GATEWAY_FACTORIES[ep.name] = ep.load()
                    _EPS_LOADED = True
            factory = GATEWAY_FACTORIES.get(name)
    if factory is None:
        raise RuntimeError(
            f"Unknown INFERENCE_GATEWAY={name!r}. Available: "
            f"{sorted(GATEWAY_FACTORIES)}. Additional gateways are provided by "
            "the Roboflow enterprise runtime package."
        )
    return factory()
