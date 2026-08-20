#!/usr/bin/env python3
"""Validate the video benchmark decision registry and fail-closed semantics.

The repository does not require ``jsonschema`` for this development-only
campaign. This module implements the small JSON Schema 2020-12 subset used by
``decision-framework.schema.json`` and then enforces cross-reference and
certification rules that JSON Schema alone cannot express clearly.
"""

import argparse
import copy
import datetime
import json
import re
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DOCUMENT = DIRECTORY / "decision-framework.json"
DEFAULT_SCHEMA = DIRECTORY / "decision-framework.schema.json"
REPO_ROOT = Path(__file__).resolve().parents[4]


class FrameworkValidationError(ValueError):
    """Raised when a framework cannot support its declared state."""


def _path(parent, child):
    return f"{parent}.{child}" if parent else str(child)


def _resolve_ref(root_schema, reference):
    if not reference.startswith("#/"):
        raise FrameworkValidationError(f"unsupported schema reference: {reference}")
    value = root_schema
    for part in reference[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[part]
    return value


def _json_type_matches(value, expected):
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise FrameworkValidationError(f"unsupported schema type: {expected}")


def _validate_schema(value, schema, root_schema, path="$"):
    if "$ref" in schema:
        referenced_schema = _resolve_ref(root_schema, schema["$ref"])
        _validate_schema(value, referenced_schema, root_schema, path)
        return

    if "const" in schema and value != schema["const"]:
        raise FrameworkValidationError(
            f"{path}: expected constant {schema['const']!r}, got {value!r}"
        )
    if "enum" in schema and value not in schema["enum"]:
        raise FrameworkValidationError(
            f"{path}: {value!r} is not one of {schema['enum']!r}"
        )

    expected_type = schema.get("type")
    if expected_type and not _json_type_matches(value, expected_type):
        raise FrameworkValidationError(
            f"{path}: expected {expected_type}, got {type(value).__name__}"
        )

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise FrameworkValidationError(f"{path}: missing required keys {missing!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise FrameworkValidationError(f"{path}: unexpected keys {extra!r}")
        for name, child in value.items():
            if name in properties:
                _validate_schema(
                    child,
                    properties[name],
                    root_schema,
                    _path(path, name),
                )

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise FrameworkValidationError(
                f"{path}: expected at least {schema['minItems']} items"
            )
        if schema.get("uniqueItems"):
            canonical = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            ]
            if len(canonical) != len(set(canonical)):
                raise FrameworkValidationError(f"{path}: items must be unique")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(
                    item,
                    schema["items"],
                    root_schema,
                    f"{path}[{index}]",
                )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise FrameworkValidationError(f"{path}: string is too short")
        pattern = schema.get("pattern")
        if pattern and re.fullmatch(pattern, value) is None:
            raise FrameworkValidationError(
                f"{path}: {value!r} does not match {pattern!r}"
            )
        if schema.get("format") == "date":
            try:
                parsed = datetime.date.fromisoformat(value)
            except ValueError as error:
                raise FrameworkValidationError(
                    f"{path}: {value!r} is not an ISO date"
                ) from error
            if parsed.isoformat() != value:
                raise FrameworkValidationError(
                    f"{path}: {value!r} is not a canonical ISO date"
                )


def _unique_ids(items, path):
    ids = [item["id"] for item in items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise FrameworkValidationError(f"{path}: duplicate ids {duplicates!r}")


def _validate_semantics(document, repo_root):
    _unique_ids(document["evidence"], "$.evidence")
    _unique_ids(document["decisions"], "$.decisions")
    evidence = {item["id"]: item for item in document["evidence"]}
    allowed_strength = {
        "verified_observation": {"committed_summary"},
        "preliminary_observation": {
            "committed_summary_without_campaign_reports",
            "committed_summary_without_raw_reports",
        },
        "design_only": {"reviewable_design"},
        "pending": {"none"},
    }

    for item in document["evidence"]:
        item_id = item["id"]
        status = item["status"]
        if item["strength"] not in allowed_strength[status]:
            raise FrameworkValidationError(
                f"evidence {item_id}: strength {item['strength']!r} does not match "
                f"status {status!r}"
            )
        if status == "verified_observation" and not item.get("facts"):
            raise FrameworkValidationError(
                f"evidence {item_id}: verified observations require nonempty facts"
            )
        source = item["source"]
        if source.startswith(("/", "http://", "https://")):
            raise FrameworkValidationError(
                f"evidence {item_id}: source must be a repository-relative file"
            )
        if not (repo_root / source).is_file():
            raise FrameworkValidationError(
                f"evidence {item_id}: source does not exist: {source}"
            )

    for decision in document["decisions"]:
        decision_id = decision["id"]
        evidence_ids = set(decision["evidenceIds"])
        missing = sorted(evidence_ids - set(evidence))
        if missing:
            raise FrameworkValidationError(
                f"decision {decision_id}: unknown evidence ids {missing!r}"
            )
        _unique_ids(
            decision["certificationRequirements"],
            f"decision {decision_id}.certificationRequirements",
        )

        for requirement in decision["certificationRequirements"]:
            requirement_ids = set(requirement["evidenceIds"])
            missing = sorted(requirement_ids - set(evidence))
            if missing:
                raise FrameworkValidationError(
                    f"requirement {decision_id}/{requirement['id']}: unknown evidence "
                    f"ids {missing!r}"
                )
            omitted = sorted(requirement_ids - evidence_ids)
            if omitted:
                raise FrameworkValidationError(
                    f"requirement {decision_id}/{requirement['id']}: evidence ids "
                    f"omitted from parent decision {omitted!r}"
                )
            if requirement["status"] == "satisfied" and any(
                evidence[item_id]["status"] != "verified_observation"
                for item_id in requirement_ids
            ):
                raise FrameworkValidationError(
                    f"requirement {decision_id}/{requirement['id']}: satisfied "
                    "requirements require only verified observations"
                )

        status = decision["certificationStatus"]
        if status == "provisional" and not any(
            evidence[item_id]["status"] == "verified_observation"
            for item_id in evidence_ids
        ):
            raise FrameworkValidationError(
                f"decision {decision_id}: provisional decisions require at least one "
                "verified observation; use design_candidate otherwise"
            )
        if status == "certified":
            if decision["unknowns"]:
                raise FrameworkValidationError(
                    f"decision {decision_id}: certified decisions cannot retain "
                    "unknowns"
                )
            if any(
                requirement["status"] != "satisfied"
                for requirement in decision["certificationRequirements"]
            ):
                raise FrameworkValidationError(
                    f"decision {decision_id}: every certification requirement must be "
                    "satisfied"
                )


def validate_document(document, schema, repo_root=REPO_ROOT):
    """Validate a parsed framework and return an independent deep copy."""
    _validate_schema(document, schema, schema)
    _validate_semantics(document, Path(repo_root))
    return copy.deepcopy(document)


def validate_paths(document_path=DEFAULT_DOCUMENT, schema_path=DEFAULT_SCHEMA):
    document_path = Path(document_path)
    schema_path = Path(schema_path)
    document = json.loads(document_path.read_text())
    schema = json.loads(schema_path.read_text())
    return validate_document(document, schema)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", default=str(DEFAULT_DOCUMENT))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    args = parser.parse_args(argv)
    validate_paths(args.document, args.schema)
    print(f"valid: {args.document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
