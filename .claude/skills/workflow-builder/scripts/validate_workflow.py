#!/usr/bin/env python3
"""Validate a workflow definition against the inference server."""

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
            @property
            def text(self):
                return self._data.decode("utf-8", errors="replace")

        def post(self, url, json=None, params=None, **kwargs):
            if params:
                from urllib.parse import urlencode
                url = url + "?" + urlencode(params)
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


def validate_workflow(api_url, api_key, specification):
    """Validate a workflow definition."""
    url = f"{api_url}/workflows/validate"
    params = {}
    if api_key:
        params["api_key"] = api_key

    resp = requests.post(url, json={"specification": specification}, params=params)

    if resp.status_code == 200:
        return True, resp.json()
    else:
        try:
            error_data = resp.json()
        except Exception:
            error_data = {"error": resp.text}
        return False, error_data


def main():
    parser = argparse.ArgumentParser(description="Validate a workflow definition")
    parser.add_argument("workflow", help="Path to workflow JSON file, or '-' for stdin, or inline JSON string")
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
