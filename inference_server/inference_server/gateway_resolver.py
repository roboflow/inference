"""Gateway resolution — entry-point registry for the gateway duck surface.

GATEWAY_API_VERSION is the versioned seam external gateway factories assert
against. Any change to EXPECTED_GATEWAY_SIGNATURES (test_gateway_contract.py)
bumps this version.

Contract under version 2: ``ensure_loaded`` returns ``("model_ready",)`` both
when the model was already loaded and when this call loaded it; callers read
only the first element, so a gateway returning extra elements is ignored,
not reported. Reporting a load (cold start, load time, requested id) is the
gateway's responsibility: the in-process gateway calls
``inference_server.middlewares.model_load.record_model_load`` from inside the
load it runs for the request that started it, and an out-of-process gateway
must record its own loads the same way, or no cold-start headers appear for
it. Callers record the attempted model id before the load themselves.

A failed load is reported as ``("error", code)`` by ``ensure_loaded`` and
``load``. A gateway may append a third element, a dict of JSON-serialisable
values describing the error that failed the load: ``error_type`` (class
name), ``message`` (without the help link), ``help_url`` (or None),
``status_code`` (of a model access error, else None) and ``restricted``
(true when the load was refused by a limit of the server configuration).
The element is optional: callers that read it treat a two-element tuple as
a failure of unknown cause. A gateway may also expose
``last_load_failure(model_id, instance="")`` returning the failure tuple of
the latest load of a model while no newer load has been started, or None;
callers treat a gateway without it as having nothing to report.

Version 3 adds the stream pipeline surface: ``model_supports_stream_pipeline``,
``get_model_pipeline_depth``, ``flush_model_stream_pipeline`` and
``shutdown_model_stream_pipeline``, each taking the model's routing key. A
gateway without a pipelined model answers False, 1, None and None; ``stats``
entries carry ``stream_pipeline_depth`` (1 when not pipelined) so route
resolution reads the depth without a round trip.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import threading
from typing import Any, Callable, Dict

from inference_server import configuration as cfg

GATEWAY_API_VERSION = 3

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
