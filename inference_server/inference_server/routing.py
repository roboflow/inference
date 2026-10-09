"""Routing keys shared by the gateway and the request framework."""

from typing import Iterable, Optional, Tuple

IMAGE_EMBEDDINGS = "image_embeddings"
DEFAULT_EMBEDDING_OUTPUT_TYPE = "feature_vector"
_CAPABILITIES_FIELD = "capabilities"
_OUTPUT_TYPE_FIELD = "output_type"
_CAPABILITY_MARKER = f"{_CAPABILITIES_FIELD}="
_CAPABILITY_FIELD_SEPARATOR = ";"


def routing_key(model_id: str, instance: str = "") -> str:
    """Key a model instance is registered and routed under.

    Matches the MMP wire format exactly: the bare ``model_id`` when no instance
    is requested, ``model_id:instance`` otherwise.
    """
    return f"{model_id}:{instance}" if instance else model_id


def routed_model_id(key: str) -> str:
    """Weights identifier behind a routing key (drops the ``:instance`` suffix)."""
    return key.rsplit(":", 1)[0]


def routed_instance(key: str) -> str:
    """Instance part of a routing key, empty for a bare model id."""
    return key.rsplit(":", 1)[1] if ":" in key else ""


def has_capability_marker(value: str) -> bool:
    """Whether user input names a capability instance anywhere in its text."""
    return _CAPABILITY_MARKER in value


def is_capability_instance(instance: str) -> bool:
    """Whether an instance name is one ``capability_instance`` builds."""
    return instance.startswith(_CAPABILITY_MARKER)


def capability_instance(
    required_capabilities: Optional[Iterable[str]] = None,
    output_type: str = DEFAULT_EMBEDDING_OUTPUT_TYPE,
) -> str:
    """Instance under which a model loaded with extra capabilities is routed.

    Args:
        required_capabilities: Operations the loaded instance must support.
        output_type: Feature or logit representation for image embeddings.

    Returns:
        ``capabilities=<a,b>`` with ``;output_type=<type>`` appended for image
        embeddings other than feature vectors; empty without capabilities, so
        plain registrations keep the bare model id.
    """
    if not required_capabilities:
        return ""

    capabilities = sorted(set(required_capabilities))
    instance = f"{_CAPABILITY_MARKER}{','.join(capabilities)}"
    if IMAGE_EMBEDDINGS in capabilities and output_type != (
        DEFAULT_EMBEDDING_OUTPUT_TYPE
    ):
        instance += f"{_CAPABILITY_FIELD_SEPARATOR}{_OUTPUT_TYPE_FIELD}={output_type}"

    return instance


def capability_load_kwargs(key: str) -> dict:
    """Load arguments encoded in the instance part of a routing key.

    Every route refuses request input carrying a capability marker, so a key
    whose instance is a capability instance was built by the server.

    Args:
        key: Routing key as produced by ``routing_key``.

    Returns:
        ``required_capabilities`` and, when present, ``output_type`` for a key
        built with ``capability_instance``; empty for any other key.
    """
    instance = routed_instance(key)
    if not is_capability_instance(instance):
        return {}

    kwargs = {}
    for field in instance.split(_CAPABILITY_FIELD_SEPARATOR):
        name, _, value = field.partition("=")
        if name == _CAPABILITIES_FIELD:
            kwargs["required_capabilities"] = value.split(",")
        elif name == _OUTPUT_TYPE_FIELD:
            kwargs["output_type"] = value

    return kwargs


def plain_model_id(key: str) -> str:
    """Identity a capability key reports for; other keys as given."""
    if is_capability_instance(routed_instance(key)):
        return routed_model_id(key)
    return key


def registration_key(model_id: str, instance: str = "") -> str:
    """Identity a registration is listed and removed under, as legacy wrote it.

    Args:
        model_id: Model id as the request named it.
        instance: Instance of the registration, empty for a plain one.

    Returns:
        ``model_id:capabilities=<a,b>[:output_type=<type>]`` for a capability
        instance, otherwise the routing key.
    """
    if not is_capability_instance(instance):
        return routing_key(model_id, instance)

    return routing_key(model_id, instance.replace(_CAPABILITY_FIELD_SEPARATOR, ":"))


def parse_registration_key(key: str) -> Tuple[str, str]:
    """Split a registration identity into the model id and the instance.

    Args:
        key: Identity as ``registration_key`` wrote it.

    Returns:
        The model id and the ``capability_instance`` form of its capability
        instance; a key without one is returned whole with an empty instance.
    """
    model_id, marker, fields = key.partition(f":{_CAPABILITY_MARKER}")
    if not marker:
        return key, ""

    instance = _CAPABILITY_MARKER + fields.replace(":", _CAPABILITY_FIELD_SEPARATOR)

    return model_id, instance
