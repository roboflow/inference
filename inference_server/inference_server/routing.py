"""Routing keys shared by the gateway and the request framework."""


def routing_key(model_id: str, instance: str = "") -> str:
    """Key a model instance is registered and routed under.

    Matches the MMP wire format exactly: the bare ``model_id`` when no instance
    is requested, ``model_id:instance`` otherwise.
    """
    return f"{model_id}:{instance}" if instance else model_id


def routed_model_id(key: str) -> str:
    """Weights identifier behind a routing key (drops the ``:instance`` suffix)."""
    return key.rsplit(":", 1)[0]
