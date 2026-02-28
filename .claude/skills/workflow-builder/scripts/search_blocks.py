#!/usr/bin/env python3
"""Search workflow blocks by keyword across names, descriptions, and schemas."""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from setup import get_api_key, get_api_url
from http_utils import fetch_blocks_describe


def search_blocks(blocks, query):
    """Search blocks by keyword. Returns list of (block, score) tuples."""
    query_lower = query.lower()
    query_words = query_lower.split()
    results = []

    for block in blocks:
        score = 0
        name = block.get("human_friendly_block_name", "").lower()
        identifier = block.get("manifest_type_identifier", "").lower()
        fqn = block.get("fully_qualified_block_class_name", "").lower()
        schema = block.get("block_schema", {})
        description = schema.get("description", "").lower()
        search_keywords = schema.get("search_keywords", [])
        if isinstance(search_keywords, list):
            search_keywords = " ".join(str(k).lower() for k in search_keywords)
        else:
            search_keywords = str(search_keywords).lower()

        for word in query_words:
            if word in name:
                score += 10
            if word in identifier:
                score += 8
            if word in search_keywords:
                score += 6
            if word in description:
                score += 3
            if word in fqn:
                score += 1

        if query_lower in name:
            score += 20
        if query_lower in identifier:
            score += 15
        if query_lower in search_keywords:
            score += 12

        if score > 0:
            results.append((block, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Search workflow blocks by keyword")
    parser.add_argument("query", help="Search query (e.g., 'object detection', 'visualization', 'ocr')")
    parser.add_argument("--limit", type=int, default=15, help="Maximum results (default: 15)")
    parser.add_argument("--api-url", help="Override inference server URL")
    parser.add_argument("--api-key", help="Override API key")
    args = parser.parse_args()

    api_url = args.api_url or get_api_url()
    api_key = args.api_key or get_api_key()

    try:
        data = fetch_blocks_describe(api_url, api_key)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    blocks = data.get("blocks", [])
    results = search_blocks(blocks, args.query)

    if not results:
        print(f"No blocks found matching '{args.query}'.")
        return

    print(f"Found {len(results)} blocks matching '{args.query}'")
    if len(results) > args.limit:
        print(f"(showing top {args.limit})")
    print()

    for block, score in results[:args.limit]:
        name = block.get("human_friendly_block_name", "")
        identifier = block.get("manifest_type_identifier", "")
        schema = block.get("block_schema", {})
        desc = schema.get("description", "")
        first_sentence = desc.split(". ")[0].split("\n")[0][:120] if desc else ""

        outputs = block.get("outputs_manifest", [])
        output_names = [o.get("name", "") for o in outputs]

        print(f"  {name}")
        print(f"    Type: `{identifier}`")
        if first_sentence:
            print(f"    {first_sentence}")
        if output_names:
            print(f"    Outputs: {', '.join(output_names)}")
        print()


if __name__ == "__main__":
    main()
