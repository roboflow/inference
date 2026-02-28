#!/usr/bin/env python3
"""List available workflow blocks from the inference server describe endpoint."""

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
        """Minimal requests-like wrapper around urllib for environments without requests."""
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
    """Fetch block descriptions from the inference server."""
    url = f"{api_url}/workflows/blocks/describe"
    payload = {}
    if api_key:
        payload["api_key"] = api_key
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def categorize_block(block):
    """Extract category from block's fully_qualified_block_class_name or manifest_type_identifier."""
    identifier = block.get("manifest_type_identifier", "")
    # e.g., "roboflow_core/roboflow_object_detection_model@v2"
    if "/" in identifier:
        name_part = identifier.split("/")[1].split("@")[0]
    else:
        name_part = identifier

    fqn = block.get("fully_qualified_block_class_name", "")

    if "visualization" in fqn.lower() or "visualization" in name_part:
        return "Visualization"
    elif "models/foundation" in fqn or "anthropic" in name_part or "gemini" in name_part or "openai" in name_part or "florence" in name_part or "yolo_world" in name_part or "clip" in name_part or "ocr" in name_part or "cogvlm" in name_part or "segment_anything" in name_part:
        return "Models / Foundation"
    elif "models/roboflow" in fqn or "roboflow_object_detection" in name_part or "roboflow_classification" in name_part or "roboflow_instance_segmentation" in name_part or "roboflow_keypoint" in name_part or "roboflow_multi_label" in name_part:
        return "Models / Roboflow"
    elif "models/" in fqn:
        return "Models / Other"
    elif "transformation" in fqn or "dynamic_crop" in name_part or "detections_filter" in name_part or "detections_transformation" in name_part or "image_slicer" in name_part or "perspective" in name_part:
        return "Transformations"
    elif "analytics" in fqn:
        return "Analytics"
    elif "formatter" in fqn or "expression" in name_part or "json_parser" in name_part or "csv_formatter" in name_part or "vlm_as" in name_part or "property_definition" in name_part:
        return "Formatters"
    elif "flow_control" in fqn or "continue_if" in name_part or "rate_limiter" in name_part:
        return "Flow Control"
    elif "sink" in fqn or "webhook" in name_part or "email" in name_part or "dataset_upload" in name_part:
        return "Sinks"
    elif "classical_cv" in fqn:
        return "Classical CV"
    elif "fusion" in fqn or "consensus" in name_part or "stitch" in name_part:
        return "Fusion"
    elif "cache" in fqn:
        return "Cache"
    else:
        return "Other"


def main():
    parser = argparse.ArgumentParser(description="List available workflow blocks")
    parser.add_argument("--category", help="Filter by category keyword (case-insensitive)")
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
        print(f"API URL: {api_url}", file=sys.stderr)
        sys.exit(1)

    blocks = data.get("blocks", [])

    if args.json:
        print(json.dumps(blocks, indent=2))
        return

    # Categorize and sort
    categorized = {}
    for block in blocks:
        category = categorize_block(block)
        if args.category and args.category.lower() not in category.lower():
            continue
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(block)

    if not categorized:
        print("No blocks found matching filter.")
        return

    # Print summary
    total = sum(len(v) for v in categorized.values())
    print(f"Found {total} blocks in {len(categorized)} categories:\n")

    for category in sorted(categorized.keys()):
        blocks_in_cat = categorized[category]
        print(f"## {category} ({len(blocks_in_cat)} blocks)")
        print()
        for block in sorted(blocks_in_cat, key=lambda b: b.get("manifest_type_identifier", "")):
            name = block.get("human_friendly_block_name", "")
            identifier = block.get("manifest_type_identifier", "")
            # Extract short description from schema if available
            schema = block.get("block_schema", {})
            desc = schema.get("description", "")
            if desc:
                # Take first sentence
                first_sentence = desc.split(". ")[0].split("\n")[0][:100]
                print(f"  - {name}: `{identifier}`")
                print(f"    {first_sentence}")
            else:
                print(f"  - {name}: `{identifier}`")
        print()


if __name__ == "__main__":
    main()
