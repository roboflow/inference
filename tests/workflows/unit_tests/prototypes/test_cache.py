import inspect

from inference.core.cache.base import BaseCache
from inference.core.workflows.prototypes.cache import WorkflowsCache


def test_protocol_covers_only_what_blocks_use() -> None:
    assert hasattr(WorkflowsCache, "get")
    assert hasattr(WorkflowsCache, "set")


def test_server_cache_satisfies_the_protocol() -> None:
    # Transitional: pins the server class to the port while both exist.
    assert hasattr(BaseCache, "get")
    assert hasattr(BaseCache, "set")


def _params(method):
    # Skip `self`; the port is the source of truth for which names to iterate.
    return list(inspect.signature(method).parameters.values())[1:]


def test_get_and_set_signatures_match_the_server_cache() -> None:
    # Guards against a drifted port (e.g. a dropped `expire` or a renamed
    # `key`) going undetected. model_monitoring_inference_aggregator/v1.py:413
    # calls cache.set(key=..., value=..., expire=900) by keyword, so kind and
    # default matter, not just presence. Annotations deliberately not
    # compared: the port uses Any/Optional[float], the server uses str/float.
    for method_name in ("get", "set"):
        port_params = _params(getattr(WorkflowsCache, method_name))
        real_params = _params(getattr(BaseCache, method_name))
        assert [p.name for p in port_params] == [p.name for p in real_params]
        for port_param, real_param in zip(port_params, real_params):
            assert port_param.kind == real_param.kind
            assert port_param.default == real_param.default
