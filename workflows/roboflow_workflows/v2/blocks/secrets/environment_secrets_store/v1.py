"""Environment secrets store: port of V1 ``roboflow_core/environment_secrets_store@v1``.

Each configured name becomes one ``secret`` output, lowercased as in V1::

    variables_storing_secrets=["SLACK_TOKEN"]
        ─▶ secrets.lookup("SLACK_TOKEN") ─▶ output slack_token = Secret("SLACK_TOKEN")
                                            (None when the variable is unset)

The ``secrets`` resource decides where values come from:

* ``EnvironmentSecrets()`` (default) reads ``os.environ`` only when the V1 flag
  ``ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES`` is true, and
  raises otherwise. Hosts that disable it for V1 disable it here too.
* ``StaticSecrets({...})`` holds named credentials the host supplies; it never
  touches the environment.

Differences from V1: outputs are ``Secret`` payloads, not strings, so results
and recordings show names only. Names that are not valid selector segments
after lowercasing, or that collide after lowercasing, are rejected (V1 kept
the last of colliding names).
"""

import os
from typing import Dict, List, Mapping, Optional, Protocol
from pydantic import Field, field_validator
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    is_selector_segment,
)


BLOCK_TYPE = "roboflow_core/environment_secrets_store@v1"

class EnvironmentAccessError(RuntimeError):
    """The host does not allow blocks to read environment variables."""

class SecretSource(Protocol):
    """Where the store reads named secret values from."""

    def lookup(self, name: str) -> Optional[str]:
        """Return the value stored under ``name``; ``None`` when absent."""

class EnvironmentSecrets:
    """Read ``os.environ``, gated by the V1 environment-access flag.

    The flag is read at every lookup. Its module is imported lazily, so
    building a catalogue does not freeze the process configuration.
    """

    def lookup(self, name: str) -> Optional[str]:
        """Return the environment variable ``name``.

        Args:
            name: Environment variable name.

        Returns:
            Its value; ``None`` when it is unset.

        Raises:
            EnvironmentAccessError: When the host disabled environment access.
        """
        from roboflow_workflows import environment

        if not environment.ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES:
            raise EnvironmentAccessError(
                f"{BLOCK_TYPE} cannot read environment variables: this host "
                "disabled it (ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES"
                "=False). Ask the host to provide a 'secrets' resource instead."
            )

        value = os.environ.get(name)

        return value

    def __repr__(self) -> str:
        return "EnvironmentSecrets()"

class StaticSecrets:
    """Named credentials held by the host; no environment access.

    Args:
        values: Secret value by name. Copied at construction.

    Raises:
        TypeError: When a name or value is not a string. The message names
            the offending key only.
    """

    def __init__(self, values: Mapping[str, str]) -> None:
        for name, value in values.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError(
                    f"StaticSecrets needs str names and values; check {name!r}"
                )

        self._values = dict(values)

    def lookup(self, name: str) -> Optional[str]:
        """Return the value stored under ``name``.

        Args:
            name: Secret name.

        Returns:
            The value; ``None`` when the host supplied none.
        """
        value = self._values.get(name)

        return value

    def __repr__(self) -> str:
        return f"StaticSecrets({sorted(self._values)})"

DEFAULT_SECRET_SOURCE = EnvironmentSecrets()

class EnvironmentSecretsStoreBlock(Block):
    """Expose named host secrets as ``secret`` outputs.

    Example: ``variables_storing_secrets=["SLACK_TOKEN"]`` outputs
    ``slack_token``; connect it to a parameter that accepts ``secret``, such
    as a sink's credential.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Environment Secrets Store",
        "port_of": "roboflow_core/environment_secrets_store@v1",
        "section": "advanced",
    }
    output_fields = ("variables_storing_secrets",)
    prunable = True

    class Params(BlockParams):
        variables_storing_secrets: List[str] = Field(
            min_length=1,
            description=(
                "Names of the secrets to read, e.g. environment variables. Each "
                "becomes one output named in lowercase."
            ),
            examples=[["MY_API_KEY", "OTHER_API_KEY"]],
        )

        @field_validator("variables_storing_secrets")
        @classmethod
        def _output_names_are_valid(cls, names: List[str]) -> List[str]:
            _check_output_names(names)

            return names

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Mapping[str, Output]:
        """Declare one ``secret`` output per configured name, lowercased.

        Args:
            params: Validated step parameters.

        Returns:
            Outputs by name.
        """
        outputs = {
            name.lower(): Output(
                SECRET_KIND, description=f"Secret {name!r}; None when it is unset."
            )
            for name in params.variables_storing_secrets
        }

        return outputs

    def __init__(self, *, secrets: SecretSource = DEFAULT_SECRET_SOURCE) -> None:
        self._secrets = secrets

    def run(
        self, *, variables_storing_secrets: List[str]
    ) -> Dict[str, Optional[Secret]]:
        """Read every configured secret.

        Args:
            variables_storing_secrets: Names to read.

        Returns:
            ``Secret`` (or ``None`` when absent) by lowercased name.
        """
        outputs = {}
        for name in variables_storing_secrets:
            value = self._secrets.lookup(name)
            outputs[name.lower()] = None if value is None else Secret(value, name=name)

        return outputs

def _check_output_names(names: List[str]) -> None:
    owners: Dict[str, str] = {}
    for name in names:
        output = name.lower()
        if not is_selector_segment(output):
            raise ValueError(
                f"secret name {name!r} cannot name an output: use letters, digits, "
                "_ or -"
            )
        if output in owners:
            raise ValueError(
                f"secret names {owners[output]!r} and {name!r} both become output "
                f"{output!r}; list each name once"
            )
        owners[output] = name
