#!/usr/bin/env python3
"""List available workflow blocks from the inference server describe endpoint."""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from setup import get_api_key, get_api_url
from http_utils import fetch_blocks_describe


def categorize_block(block):
    """Extract category from block metadata."""
    identifier = block.get("manifest_type_identifier", "")
    name_part = identifier.split("/")[1].split("@")[0] if "/" in identifier else identifier
    fqn = block.get("fully_qualified_block_class_name", "")

    if "visualization" in fqn.lower() or "visualization" in name_part:
        return "Visualization"
    elif "models/foundation" in fqn or any(kw in name_part for kw in ("anthropic", "gemini", "openai", "florence", "yolo_world", "clip", "ocr", "cogvlm", "segment_anything")):
        return "Models / Foundation"
    elif "models/roboflow" in fqn or any(kw in name_part for kw in ("roboflow_object_detection", "roboflow_classification", "roboflow_instance_segmentation", "roboflow_keypoint", "roboflow_multi_label")):
        return "Models / Roboflow"
    elif "models/" in fqn:
        return "Models / Other"
    elif "transformation" in fqn or any(kw in name_part for kw in ("dynamic_crop", "detections_filter", "detections_transformation", "image_slicer", "perspective")):
        return "Transformations"
    elif "analytics" in fqn:
        return "Analytics"
    elif "formatter" in fqn or any(kw in name_part for kw in ("expression", "json_parser", "csv_formatter", "vlm_as", "property_definition")):
        return "Formatters"
    elif "flow_control" in fqn or any(kw in name_part for kw in ("continue_if", "rate_limiter")):
        return "Flow Control"
    elif "sink" in fqn or any(kw in name_part for kw in ("webhook", "email", "dataset_upload")):
        return "Sinks"
    elif "classical_cv" in fqn:
        return "Classical CV"
    elif "fusion" in fqn or any(kw in name_part for kw in ("consensus", "stitch")):
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
        data = fetch_blocks_describe(api_url, api_key)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    blocks = data.get("blocks", [])

    if args.json:
        print(json.dumps(blocks, indent=2))
        return

    categorized = {}
    for block in blocks:
        category = categorize_block(block)
        if args.category and args.category.lower() not in category.lower():
            continue
        categorized.setdefault(category, []).append(block)

    if not categorized:
        print("No blocks found matching filter.")
        return

    total = sum(len(v) for v in categorized.values())
    print(f"Found {total} blocks in {len(categorized)} categories:\n")

    for category in sorted(categorized.keys()):
        blocks_in_cat = categorized[category]
        print(f"## {category} ({len(blocks_in_cat)} blocks)")
        print()
        for block in sorted(blocks_in_cat, key=lambda b: b.get("manifest_type_identifier", "")):
            name = block.get("human_friendly_block_name", "")
            identifier = block.get("manifest_type_identifier", "")
            schema = block.get("block_schema", {})
            desc = schema.get("description", "")
            if desc:
                first_sentence = desc.split(". ")[0].split("\n")[0][:100]
                print(f"  - {name}: `{identifier}`")
                print(f"    {first_sentence}")
            else:
                print(f"  - {name}: `{identifier}`")
        print()


if __name__ == "__main__":
    main()
