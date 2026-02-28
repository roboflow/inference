#!/usr/bin/env python3
"""Get detailed information about a specific workflow block."""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from setup import get_api_key, get_api_url

try:
    import requests
except ImportError:
    import urllib.request
    import urllib.error

    class _Requests:
        class Response:
            def __init__(self, urllib_response):
                self.status_code = urllib_response.getcode()
                self._data = urllib_response.read()
            def json(self):
                return json.loads(self._data)
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}: {self._data[:200]}")

        def post(self, url, json=None, **kwargs):
            data = __import__("json").dumps(json).encode() if json else None
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            try:
                resp = urllib.request.urlopen(req, timeout=30)
                return self.Response(resp)
            except urllib.error.HTTPError as e:
                r = self.Response(e)
                r.status_code = e.code
                return r

    requests = _Requests()


def fetch_blocks(api_url, api_key):
    url = f"{api_url}/workflows/blocks/describe"
    payload = {}
    if api_key:
        payload["api_key"] = api_key
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


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
    parts = []
    prop_type = prop.get("type", "")
    title = prop.get("title", name)

    # Check if it accepts selectors
    any_of = prop.get("anyOf", [])
    one_of = prop.get("oneOf", [])
    alternatives = any_of or one_of

    accepts_selector = False
    types = []
    if alternatives:
        for alt in alternatives:
            if alt.get("type") == "string" and "pattern" in alt.get("", ""):
                accepts_selector = True
            ref = alt.get("$ref", "")
            if "selector" in ref.lower():
                accepts_selector = True
            alt_type = alt.get("type", alt.get("$ref", ""))
            if alt_type:
                types.append(alt_type)
    else:
        types.append(prop_type)

    is_required = name in required_fields
    default = prop.get("default")
    description = prop.get("description", "")

    # Build display string
    type_str = " | ".join(types) if types else "any"
    req_str = "REQUIRED" if is_required else f"default: {json.dumps(default)}"

    if description:
        # Truncate long descriptions
        desc_short = description[:150]
        if len(description) > 150:
            desc_short += "..."
    else:
        desc_short = ""

    return {
        "name": name,
        "type": type_str,
        "required": is_required,
        "default": default,
        "description": desc_short,
        "accepts_selector": accepts_selector,
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

    # Description
    desc = schema.get("description", "")
    if desc:
        print(f"\n## Description\n{desc[:500]}")

    # Properties
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Separate name/type (always present) from configurable properties
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

    # Outputs
    print("\n## Outputs")
    print()
    for output in outputs:
        out_name = output.get("name", "")
        out_kinds = output.get("kind", [])
        kind_names = [k.get("name", "wildcard") if isinstance(k, dict) else str(k) for k in out_kinds]
        print(f"  - `{out_name}`: {', '.join(kind_names)}")

    # Example step JSON
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
        data = fetch_blocks(api_url, api_key)
    except Exception as e:
        print(f"Error fetching blocks: {e}", file=sys.stderr)
        sys.exit(1)

    blocks = data.get("blocks", [])
    block = find_block(blocks, args.block_type)

    if not block:
        print(f"Block '{args.block_type}' not found.")
        print("\nDid you mean one of these?")
        # Show similar blocks
        search_lower = args.block_type.lower()
        for b in blocks:
            tid = b.get("manifest_type_identifier", "")
            name = b.get("human_friendly_block_name", "")
            # Simple fuzzy: check if any word matches
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
