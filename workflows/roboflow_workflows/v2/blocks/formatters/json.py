"""JSON field extraction: port of V1 ``roboflow_core/json_parser@v1``.

Parsing is V1's, unchanged::

    raw_json ── first ```json fence (any case), else the whole text ──▶ json.loads
      not JSON                         every field None, error_status True
      JSON object                      field = object.get(key)
                                       error_status = some key is absent
      other JSON, no fields expected   error_status False
      other JSON (array, number, ...)  every field None, error_status True

A present ``null`` is a found field. Only the first fence is read, even when it
is invalid and a later one is valid.

``expected_fields`` names the outputs. The V1 list form uses each JSON key as
its output name. The mapping form ``{output_name: json_key}`` reaches keys a
selector cannot name, such as ``"class name"``, ``"a.b"`` or ``"error_status"``.
Output names follow selector rules; ``error_status`` is reserved. Field
outputs declare no kind, so a typed consumer checks each value at run time.
"""

import json
import re
from typing import Any, Dict, List, Mapping, Tuple, Union

from pydantic import Field, field_validator
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    is_selector_segment,
)

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)
ERROR_STATUS = "error_status"
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9_\-]")

ExpectedFields = Union[List[str], Dict[str, str]]


class JsonParserBlock(Block):
    """Parse a JSON string and expose chosen fields as outputs.

    Example: ``expected_fields={"label": "class name", "confidence":
    "confidence"}`` turns ``'{"class name": "car", "confidence": 0.9}'`` into
    ``label="car"``, ``confidence=0.9`` and ``error_status=False``.
    """

    type = "v2/json_parser@v1"
    metadata = {
        "name": "JSON Parser",
        "port_of": "roboflow_core/json_parser@v1",
        "section": "advanced",
    }
    output_fields = ("expected_fields",)

    class Params(BlockParams):
        raw_json: str | Ref(STRING_KIND) = Field(
            description=(
                "JSON text, raw or inside a ```json Markdown fence as language "
                "models often return it. Only the first fence is parsed."
            ),
            examples=["$steps.lmm.output", '{"model_id": "yolov8n-640"}'],
        )
        expected_fields: ExpectedFields = Field(
            description=(
                "Fields to expose as outputs. A list uses each JSON key as the "
                "output name. A mapping {output_name: json_key} also reaches "
                "keys with spaces, dots or other characters a selector cannot "
                "name. A missing field outputs None and sets error_status."
            ),
            examples=[
                ["model_id", "confidence"],
                {"label": "class name", "confidence": "confidence"},
            ],
        )

        @field_validator("expected_fields")
        @classmethod
        def _output_names_are_valid(cls, value: ExpectedFields) -> ExpectedFields:
            if isinstance(value, list):
                _check_list_entries(value)
            for name in value:
                _check_output_name(name)

            return value

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Mapping[str, Output]:
        """Declare one wildcard-kind output per field, then ``error_status``.

        Args:
            params: Validated step parameters.

        Returns:
            Outputs by name.
        """
        outputs = {
            name: Output(description=f"Value of JSON key {key!r}, any kind.")
            for name, key in output_json_keys(params.expected_fields).items()
        }
        outputs[ERROR_STATUS] = Output(
            BOOLEAN_KIND,
            description="True when the text is not JSON or a field is missing.",
        )

        return outputs

    def run(self, *, raw_json: str, expected_fields: ExpectedFields) -> Dict[str, Any]:
        """Parse ``raw_json`` and pick the expected fields.

        Args:
            raw_json: JSON text, optionally inside a Markdown fence.
            expected_fields: Validated list or ``{output_name: json_key}``.

        Returns:
            One value per output name, plus ``error_status``.
        """
        json_keys = output_json_keys(expected_fields)
        error_status, values = parse_json_fields(raw_json, json_keys=json_keys)
        values[ERROR_STATUS] = error_status

        return values


def output_json_keys(expected_fields: ExpectedFields) -> Dict[str, str]:
    """Return the JSON key read for each output name.

    Args:
        expected_fields: A list of keys or a ``{output_name: json_key}`` mapping.

    Returns:
        ``{output_name: json_key}`` in declaration order.
    """
    if isinstance(expected_fields, Mapping):
        return dict(expected_fields)

    json_keys = {key: key for key in expected_fields}

    return json_keys


def parse_json_fields(
    raw_json: str, *, json_keys: Mapping[str, str]
) -> Tuple[bool, Dict[str, Any]]:
    """Parse JSON text as V1 ``roboflow_core/json_parser@v1`` does.

    Args:
        raw_json: JSON text, optionally inside a ```json Markdown fence.
        json_keys: JSON key read for each output name.

    Returns:
        ``(error_status, values by output name)``.
    """
    fences = JSON_MARKDOWN_BLOCK_PATTERN.findall(raw_json)
    content = fences[0] if fences else raw_json
    try:
        parsed = json.loads(content)
    except (ValueError, RecursionError):
        return True, dict.fromkeys(json_keys)

    if not isinstance(parsed, dict):
        return bool(json_keys), dict.fromkeys(json_keys)

    values = {name: parsed.get(key) for name, key in json_keys.items()}
    missing = any(key not in parsed for key in json_keys.values())

    return missing, values


def _check_list_entries(entries: List[str]) -> None:
    duplicates = sorted({entry for entry in entries if entries.count(entry) > 1})
    if duplicates:
        raise ValueError(f"duplicate expected_fields entries: {duplicates}")

    for entry in entries:
        if entry != ERROR_STATUS and not is_selector_segment(entry):
            safe_name = _UNSAFE_CHARACTERS.sub("_", entry) or "field"
            raise ValueError(
                f"JSON key {entry!r} is not a valid output name (letters, digits, "
                f"_ or -). Name its output with the mapping form, e.g. "
                f"{{{safe_name!r}: {entry!r}}}"
            )


def _check_output_name(name: str) -> None:
    if name == ERROR_STATUS:
        raise ValueError(
            f"output name {ERROR_STATUS!r} is reserved for the parse status. "
            f"To read a JSON key with that name, map it, e.g. "
            f"{{'upstream_error': {ERROR_STATUS!r}}}"
        )
    if not is_selector_segment(name):
        raise ValueError(
            f"output name {name!r} must use letters, digits, _ or - so a "
            "selector can address it"
        )
