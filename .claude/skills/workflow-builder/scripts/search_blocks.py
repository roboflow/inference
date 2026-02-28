#!/usr/bin/env python3
"""Search workflow blocks by keyword across names, descriptions, and schemas."""

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

        # Score based on where the match is found
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

        # Exact phrase match bonuses
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
    parser.add_argument("--limit", type=int, default=15, help="Maximum results to show (default: 15)")
    parser.add_argument("--api-url", help="Override inference server URL")
    parser.add_argument("--api-key", help="Override API key")
    args = parser.parse_args()

    api_url = args.api_url or get_api_url()
    api_key = args.api_key or get_api_key()

    try:
        data = fetch_blocks(api_url, api_key)
    except Exception as e:
        print(f"Error fetching blocks: {e}", file=sys.stderr)
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
