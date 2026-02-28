#!/usr/bin/env python3
"""Validate a workflow definition against the inference server."""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from setup import get_api_key, get_api_url
from http_utils import post_json


def validate_workflow(api_url, api_key, specification):
    """Validate a workflow definition. Returns (success, result_dict)."""
    url = f"{api_url}/workflows/validate"
    params = {}
    if api_key:
        params["api_key"] = api_key
    status, data = post_json(url, json_payload={"specification": specification}, params=params)
    return status == 200, data


def main():
    parser = argparse.ArgumentParser(description="Validate a workflow definition")
    parser.add_argument("workflow", help="Path to workflow JSON file, '-' for stdin, or inline JSON string")
    parser.add_argument("--api-url", help="Override inference server URL")
    parser.add_argument("--api-key", help="Override API key")
    args = parser.parse_args()

    api_url = args.api_url or get_api_url()
    api_key = args.api_key or get_api_key()

    # Load workflow
    if args.workflow == "-":
        spec = json.load(sys.stdin)
    elif args.workflow.startswith("{"):
        spec = json.loads(args.workflow)
    elif os.path.isfile(args.workflow):
        with open(args.workflow) as f:
            spec = json.load(f)
    else:
        print(f"Error: '{args.workflow}' is not a valid file path or JSON string.", file=sys.stderr)
        sys.exit(1)

    try:
        success, result = validate_workflow(api_url, api_key, spec)
    except Exception as e:
        print(f"Error connecting to server: {e}", file=sys.stderr)
        print(f"API URL: {api_url}", file=sys.stderr)
        sys.exit(1)

    if success:
        print("VALID - Workflow definition is valid.")
    else:
        print("INVALID - Workflow definition has errors:")
        print()
        if isinstance(result, dict):
            message = result.get("message", result.get("detail", result.get("error", "")))
            if message:
                print(f"  {message}")
            else:
                print(json.dumps(result, indent=2))
        else:
            print(f"  {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
