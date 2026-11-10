"""Credentials as payloads: the ``Secret`` carrier and the ``secret`` kind.

A ``Secret`` shows its name everywhere and its value only through ``reveal()``::

    host env / mapping ─▶ environment secrets store ─▶ Secret("SLACK_TOKEN")
    workflow input of kind ["secret"] ───────────────▶ Secret("input")
                                     │
          consuming block:  revealed = reveal_secrets(params)
                            send(revealed.value); message = revealed.scrub(text)
                                     │
    serialized output ─▶ "<secret SLACK_TOKEN>"    Python output ─▶ the Secret
    recording ─────────▶ {"name": "SLACK_TOKEN"}   replay ─▶ unavailable Secret
    managed state, native wire format, pickle ─▶ rejected

What this does not protect: a string that was never wrapped (a literal in a
workflow definition, an input declared ``string``), plaintext after a block
called ``reveal()``, and text produced by third-party code (for example a
library's own logging). ``Revealed.scrub`` replaces exact occurrences only;
encoded forms such as base64 or URL-encoding pass through unchanged.

This module does not import the engine. The recording codec object is built
from ``encode_secret_record`` and ``decode_secret_record`` by the secrets block
package, next to the other catalogue registrations.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from roboflow_workflows.v2.data.kinds import Kind

SECRET_CODEC_NAME = "v2/secret@1"
"""Recording codec identity of ``Secret``; the JSON form is ``{"name": NAME}``."""

INPUT_SECRET_NAME = "input"
"""Name given to a secret supplied as a plain string workflow input."""


class SecretUnavailableError(RuntimeError):
    """A secret without a value was revealed, e.g. one read back from a recording."""


class Secret:
    """A credential whose value appears only through ``reveal()``.

    ``repr``, ``str`` and f-strings show the name only. Copies return the same
    object; equality is identity. Pickling raises ``TypeError`` because a
    pickle would carry the plaintext into files, queues or other processes.

    Args:
        value: The plaintext; ``None`` for an unavailable secret.
        name: Name shown instead of the value, e.g. the environment variable.

    Raises:
        TypeError: When ``name`` is not a non-empty string or ``value`` is not
            a string or ``None``. The message never contains ``value``.
    """

    __slots__ = ("_name", "_value")

    def __init__(self, value: Optional[str], *, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("a secret name must be a non-empty string")
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"secret {name!r} must hold a str, got {type(value).__name__}"
            )

        self._name = name
        self._value = value

    @classmethod
    def unavailable(cls, name: str) -> "Secret":
        """Create a secret known only by name, as recordings restore it.

        Args:
            name: Name of the secret.

        Returns:
            A secret whose ``reveal()`` raises ``SecretUnavailableError``.
        """
        secret = cls(None, name=name)

        return secret

    @property
    def name(self) -> str:
        """Name shown instead of the value."""
        return self._name

    @property
    def available(self) -> bool:
        """Whether ``reveal()`` returns a value."""
        return self._value is not None

    def reveal(self) -> str:
        """Return the plaintext.

        Returns:
            The secret value.

        Raises:
            SecretUnavailableError: When the secret holds no value, e.g. it was
                read from a recording, which stores names only.
        """
        if self._value is None:
            raise SecretUnavailableError(
                f"secret {self._name!r} has no value here: recordings and replays "
                "keep only its name. Supply the credential again from the host "
                "to use it."
            )

        return self._value

    def __repr__(self) -> str:
        state = "" if self.available else ", unavailable"
        return f"Secret({self._name!r}{state})"

    def __str__(self) -> str:
        return f"<secret {self._name}>"

    def __format__(self, format_spec: str) -> str:
        return str(self)

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise TypeError(f"{self!r} cannot be pickled")

    def __copy__(self) -> "Secret":
        return self

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Secret":
        return self


def to_secret(value: Any, *, name: str) -> Optional[Secret]:
    """Wrap a credential parameter that may be a plain string.

    Blocks call this at entry for parameters that accept a ``str`` literal or
    a ``secret``, so later messages can be scrubbed of either.

    Args:
        value: A ``Secret``, a ``str`` or ``None``.
        name: Name for a wrapped string, e.g. the parameter name.

    Returns:
        ``value`` itself when it is a ``Secret`` or ``None``; otherwise a new
        ``Secret``.

    Raises:
        TypeError: For any other type. The message never contains ``value``.
    """
    if value is None or isinstance(value, Secret):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a secret or a str, got {type(value).__name__}")

    secret = Secret(value, name=name)

    return secret


@dataclass(frozen=True, repr=False)
class Revealed:
    """Plaintext copy of a value tree and the secrets found in it.

    Args:
        value: The tree with every ``Secret`` replaced by its plaintext.
        secrets: The revealed secrets, for ``scrub``.
    """

    value: Any
    secrets: Tuple[Secret, ...]

    def scrub(self, text: str) -> str:
        """Replace each exact plaintext occurrence with the secret's marker.

        Longer values are replaced first. Empty values are ignored. Encoded or
        partial forms of a value are not found.

        Args:
            text: Message that may contain plaintext, e.g. a library error.

        Returns:
            ``text`` with every revealed value replaced by ``<secret NAME>``.
        """
        ordered = sorted(self.secrets, key=lambda secret: len(secret.reveal()))
        for secret in reversed(ordered):
            plain = secret.reveal()
            if plain:
                text = text.replace(plain, str(secret))

        return text

    def __repr__(self) -> str:
        names = [secret.name for secret in self.secrets]
        return f"Revealed(secrets={names})"


def reveal_secrets(value: Any) -> Revealed:
    """Replace every ``Secret`` in nested dicts, lists and tuples by its value.

    The input is never mutated. Containers that hold a secret are rebuilt as
    ``dict``, ``list`` or ``tuple``; other values are kept as they are.

    Args:
        value: A ``Secret`` or a tree of mappings, lists and tuples.

    Returns:
        The plaintext tree and the secrets found in it.

    Raises:
        SecretUnavailableError: When a found secret holds no value.
    """
    found = []

    def walk(node: Any) -> Any:
        if isinstance(node, Secret):
            plain = node.reveal()
            found.append(node)
            return plain
        if isinstance(node, dict):
            return {key: walk(item) for key, item in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        return node

    plain_tree = walk(value)
    revealed = Revealed(value=plain_tree, secrets=tuple(found))

    return revealed


def encode_secret_record(value: Secret, sink: Any, encoder: Any) -> Dict[str, str]:
    """Recording encoder: keep the name, never the value.

    Args:
        value: The secret being recorded.
        sink: Blob sink of the recording; unused.
        encoder: Payload encoder of the recording; unused.

    Returns:
        ``{"name": value.name}``.
    """
    node = {"name": value.name}

    return node


def decode_secret_record(node: Any, source: Any, decoder: Any) -> Secret:
    """Recording decoder: restore an unavailable secret by name.

    Args:
        node: JSON node written by ``encode_secret_record``.
        source: Blob source of the recording; unused.
        decoder: Payload decoder of the recording; unused.

    Returns:
        ``Secret.unavailable(name)``.

    Raises:
        ValueError: When ``node`` is not ``{"name": <non-empty str>}``.
    """
    well_formed = (
        isinstance(node, dict)
        and set(node) == {"name"}
        and isinstance(node["name"], str)
        and bool(node["name"])
    )
    if not well_formed:
        raise ValueError('a recorded secret must be {"name": <non-empty str>}')

    secret = Secret.unavailable(node["name"])

    return secret


def _is_secret(payload: Any) -> bool:
    return isinstance(payload, Secret)


def _secret_from_input(value: Any) -> Secret:
    if isinstance(value, Secret):
        return value
    if isinstance(value, str):
        return Secret(value, name=INPUT_SECRET_NAME)

    raise TypeError(f"a secret must be a str, got {type(value).__name__}")


def _serialize_secret(payload: Any) -> str:
    if not isinstance(payload, Secret):
        raise TypeError(f"not a Secret: {type(payload).__name__}")

    return str(payload)


SECRET_KIND = Kind(
    name="secret",
    description=(
        "Credential, as V1 'secret'. Payload: Secret. A str input is wrapped; "
        "serialized outputs show '<secret NAME>', never the value."
    ),
    validate=_is_secret,
    deserialize=_secret_from_input,
    serialize=_serialize_secret,
)
