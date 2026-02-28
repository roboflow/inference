#!/usr/bin/env python3
"""Get detailed information about a specific workflow block."""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from setup import get_api_key, get_api_url
from http_utils import fetch_blocks_describe


def find_block(blocks, identifier):
    """Find a block by type identifier (exact or partial match)."""
    identifier_lower = identifier.lower()

    # Exact match first
    for block in blocks:
        if block.get("manifest_type_identifier", "").lower() == identifier_lower:
            return block
        for alias in block.get("manifest_type_identifier_aliases", []):
            if alias.lower() == identifier_lower:
                return block

    # Partial match
    matches = []
    for block in blocks:
        tid = block.get("manifest_type_identifier", "").lower()
        name = block.get("human_friendly_block_name", "").lower()
        if identifier_lower in tid or identifier_lower in name:
            matches.append(block)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple blocks match '{identifier}':")
        for m in matches:
            print(f"  - {m.get('human_friendly_block_name')}: {m.get('manifest_type_identifier')}")
        print("\nPlease use the exact type identifier.")
        sys.exit(1)

    return None


def format_schema_property(name, prop, required_fields):
    """Format a single schema property for display."""
    is_required = name in required_fields
    default = prop.get("default")
    description = prop.get("description", "")

    desc_short = description[:150] + "..." if len(description) > 150 else description

    return {
        "name": name,
        "required": is_required,
        "default": default,
        "description": desc_short,
    }


def print_block_details(block):
    """Print formatted block details."""
    name = block.get("human_friendly_block_name", "Unknown")
    identifier = block.get("manifest_type_identifier", "")
    aliases = block.get("manifest_type_identifier_aliases", [])
    source = block.get("block_source", "")
    compat = block.get("execution_engine_compatibility", "")
    schema = block.get("block_schema", {})
    outputs = block.get("outputs_manifest", [])
    dim_offsets = block.get("input_dimensionality_offsets", {})
    output_dim_offset = block.get("output_dimensionality_offset", 0)

    print(f"# {name}")
    print(f"Type: `{identifier}`")
    if aliases:
        print(f"Aliases: {', '.join(f'`{a}`' for a in aliases)}")
    print(f"Source: {source}")
    if compat:
        print(f"Engine Compatibility: {compat}")
    if output_dim_offset != 0:
        print(f"Output Dimensionality Offset: {output_dim_offset:+d}")

    desc = schema.get("description", "")
    if desc:
        print(f"\n## Description\n{desc[:500]}")

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    skip_props = {"type", "name"}

    print("\n## Properties")
    print()
    for prop_name in sorted(properties.keys()):
        if prop_name in skip_props:
            continue
        prop = properties[prop_name]
        info = format_schema_property(prop_name, prop, required)

        req_marker = " (REQUIRED)" if info["required"] else ""
        default_str = "" if info["required"] else f" [default: {json.dumps(info['default'])}]"
        dim_str = ""
        if prop_name in dim_offsets and dim_offsets[prop_name] != 0:
            dim_str = f" [dimensionality offset: {dim_offsets[prop_name]:+d}]"

        print(f"  - `{prop_name}`{req_marker}{default_str}{dim_str}")
        if info["description"]:
            print(f"    {info['description']}")

    print("\n## Outputs")
    print()
    for output in outputs:
        out_name = output.get("name", "")
        out_kinds = output.get("kind", [])
        kind_names = [k.get("name", "wildcard") if isinstance(k, dict) else str(k) for k in out_kinds]
        print(f"  - `{out_name}`: {', '.join(kind_names)}")

    print("\n## Example Step")
    print()
    example = {"type": identifier, "name": "<step_name>"}
    for prop_name in sorted(properties.keys()):
        if prop_name in skip_props:
            continue
        prop = properties[prop_name]
        examples = prop.get("examples", [])
        if examples:
            example[prop_name] = examples[0]
        elif prop_name in required:
            example[prop_name] = f"<{prop_name}>"
    print("```json")
    print(json.dumps(example, indent=2))
    print("```")


def main():
    parser = argparse.ArgumentParser(description="Describe a workflow block")
    parser.add_argument("block_type", help="Block type identifier (e.g., roboflow_core/roboflow_object_detection_model@v2)")
    parser.add_argument("--api-url", help="Override inference server URL")
    parser.add_argument("--api-key", help="Override API key")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    api_url = args.api_url or get_api_url()
    api_key = args.api_key or get_api_key()

    try:
        data = fetch_blocks_describe(api_url, api_key)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    blocks = data.get("blocks", [])
    block = find_block(blocks, args.block_type)

    if not block:
        print(f"Block '{args.block_type}' not found.")
        print("\nDid you mean one of these?")
        search_lower = args.block_type.lower()
        for b in blocks:
            tid = b.get("manifest_type_identifier", "")
            name = b.get("human_friendly_block_name", "")
            for word in search_lower.split("_"):
                if len(word) > 2 and (word in tid.lower() or word in name.lower()):
                    print(f"  - {name}: `{tid}`")
                    break
        sys.exit(1)

    if args.json:
        print(json.dumps(block, indent=2))
    else:
        print_block_details(block)


if __name__ == "__main__":
    main()
