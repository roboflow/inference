"""Payload kind of language model answers.

``language_model_output`` matches the V1 kind name. Its payload is what a
language or vision-language model block returned: the answer text (``str``),
or a dictionary when the block already decoded a structured answer. The kind
keeps payloads as they are; it parses nothing. Blocks that read answers, such
as the VLM parsers in ``blocks/language``, decide how to interpret them.

Workflow inputs and serialized outputs use the payload unchanged.
"""

from collections.abc import Mapping
from typing import Any

from roboflow_workflows.v2.data.kinds import Kind


def _is_language_model_output(payload: Any) -> bool:
    return isinstance(payload, (str, Mapping))


LANGUAGE_MODEL_OUTPUT_KIND = Kind(
    name="language_model_output",
    description=(
        "Answer of a language model: the answer text (str) or an already "
        "decoded structured answer (dict)."
    ),
    validate=_is_language_model_output,
)

__all__ = ["LANGUAGE_MODEL_OUTPUT_KIND"]
