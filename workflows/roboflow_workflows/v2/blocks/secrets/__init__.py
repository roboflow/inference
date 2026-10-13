"""Secret providers and the ``Secret`` recording codec.

Register with a catalogue together with the kind::

    Catalogue(
        [EnvironmentSecretsStoreBlock, ...],
        kinds=[SECRET_KIND, ...],
        codecs=[SECRET_RECORDING_CODEC, ...],
    )

The codec writes a secret's name only; reading a recording restores an
unavailable ``Secret`` whose ``reveal()`` raises ``SecretUnavailableError``.
"""

from roboflow_workflows.v2.blocks.secrets.environment_store import (
    EnvironmentAccessError,
    EnvironmentSecrets,
    EnvironmentSecretsStoreBlock,
    SecretSource,
    StaticSecrets,
)
from roboflow_workflows.v2.data.secrets import (
    SECRET_CODEC_NAME,
    Secret,
    decode_secret_record,
    encode_secret_record,
)
from roboflow_workflows.v2.engine.recording.codecs import PayloadCodec, type_name_of

SECRET_RECORDING_CODEC = PayloadCodec(
    name=SECRET_CODEC_NAME,
    type_name=type_name_of(Secret),
    encode=encode_secret_record,
    decode=decode_secret_record,
)

__all__ = [
    "BLOCKS",
    "EnvironmentAccessError",
    "EnvironmentSecrets",
    "EnvironmentSecretsStoreBlock",
    "SECRET_RECORDING_CODEC",
    "SecretSource",
    "StaticSecrets",
]

BLOCKS = (EnvironmentSecretsStoreBlock,)
