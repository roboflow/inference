#!/usr/bin/env python3
"""Run a workflow against an image and display results."""

import argparse
import base64
import json
import os
import sys
import tempfile

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

        def post(self, url, json=None, **kwargs):
            data = __import__("json").dumps(json).encode() if json else None
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            try:
                resp = urllib.request.urlopen(req, timeout=120)
                return self.Response(resp)
            except urllib.error.HTTPError as e:
                r = self.Response(e)
                r.status_code = e.code
                return r

    requests = _Requests()


def prepare_image_input(image_arg):
    """Convert image argument to workflow input format."""
    if image_arg.startswith("http://") or image_arg.startswith("https://"):
        return {"type": "url", "value": image_arg}
    elif os.path.isfile(image_arg):
        with open(image_arg, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return {"type": "base64", "value": image_data}
    else:
        # Assume it's already base64
        return {"type": "base64", "value": image_arg}


def save_image_output(image_data, output_name):
    """Save base64 image data to a temp file and return the path."""
    if isinstance(image_data, dict) and "value" in image_data:
        b64 = image_data["value"]
    elif isinstance(image_data, str):
        # Check if it looks like base64
        if len(image_data) > 200 and not image_data.startswith("/"):
            b64 = image_data
        else:
            return None
    else:
        return None

    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        return None

    # Determine extension from magic bytes
    ext = ".png"
    if img_bytes[:2] == b'\xff\xd8':
        ext = ".jpg"
    elif img_bytes[:4] == b'\x89PNG':
        ext = ".png"

    tmp_dir = tempfile.mkdtemp(prefix="workflow_output_")
    path = os.path.join(tmp_dir, f"{output_name}{ext}")
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path


def process_output_value(value, output_name, depth=0):
    """Process an output value, saving images and formatting for display."""
    if depth > 3:
        return str(value)[:200]

    if isinstance(value, dict):
        # Check if it's an image
        if value.get("type") in ("base64", "url") and "value" in value:
            path = save_image_output(value, output_name)
            if path:
                return f"[Image saved to: {path}]"

        # Recursively process dict values
        result = {}
        for k, v in value.items():
            result[k] = process_output_value(v, f"{output_name}_{k}", depth + 1)
        return result

    elif isinstance(value, list):
        return [process_output_value(item, f"{output_name}_{i}", depth + 1) for i, item in enumerate(value)]

    elif isinstance(value, str) and len(value) > 500:
        # Might be base64 image data
        path = save_image_output(value, output_name)
        if path:
            return f"[Image saved to: {path}]"
        return value[:200] + f"... ({len(value)} chars total)"

    return value


def run_workflow(api_url, api_key, specification, inputs):
    """Run a workflow with the given specification and inputs."""
    url = f"{api_url}/workflows/run"

    payload = {
        "specification": specification,
        "inputs": inputs,
    }
    if api_key:
        payload["api_key"] = api_key

    resp = requests.post(url, json=payload)

    if resp.status_code == 200:
        return True, resp.json()
    else:
        try:
            error_data = resp.json()
        except Exception:
            error_data = {"error": resp.text}
        return False, error_data


def main():
    parser = argparse.ArgumentParser(description="Run a workflow against an image")
    parser.add_argument("workflow", help="Path to workflow JSON file, or inline JSON string")
    parser.add_argument("--image", required=True, help="Image path or URL")
    parser.add_argument("--image-name", default="image", help="Name of the image input (default: 'image')")
    parser.add_argument("--param", action="append", default=[], help="Additional parameters as key=value pairs")
    parser.add_argument("--api-url", help="Override inference server URL")
    parser.add_argument("--api-key", help="Override API key")
    parser.add_argument("--raw", action="store_true", help="Output raw JSON response")
    args = parser.parse_args()

    api_url = args.api_url or get_api_url()
    api_key = args.api_key or get_api_key()

    if not api_key:
        print("Error: No API key configured.", file=sys.stderr)
        print("Run: python3 setup.py set --api-key <KEY> --workspace <NAME>", file=sys.stderr)
        sys.exit(1)

    # Load workflow
    if args.workflow.startswith("{"):
        spec = json.loads(args.workflow)
    elif os.path.isfile(args.workflow):
        with open(args.workflow) as f:
            spec = json.load(f)
    else:
        print(f"Error: '{args.workflow}' is not a valid file path or JSON string.", file=sys.stderr)
        sys.exit(1)

    # Build inputs
    inputs = {args.image_name: prepare_image_input(args.image)}

    # Add extra parameters
    for param in args.param:
        if "=" not in param:
            print(f"Error: parameter '{param}' must be in key=value format.", file=sys.stderr)
            sys.exit(1)
        key, value = param.split("=", 1)
        # Try to parse as JSON for numeric/bool/list values
        try:
            inputs[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            inputs[key] = value

    # Run
    try:
        success, result = run_workflow(api_url, api_key, spec, inputs)
    except Exception as e:
        print(f"Error running workflow: {e}", file=sys.stderr)
        sys.exit(1)

    if not success:
        print("WORKFLOW EXECUTION FAILED:")
        print()
        if isinstance(result, dict):
            message = result.get("message", result.get("detail", result.get("error", "")))
            if message:
                print(f"  {message}")
            else:
                print(json.dumps(result, indent=2))
        else:
            print(result)
        sys.exit(1)

    if args.raw:
        print(json.dumps(result, indent=2, default=str))
        return

    # Process outputs
    outputs = result.get("outputs", [])
    if not outputs:
        print("Workflow returned no outputs.")
        return

    print(f"Workflow executed successfully. {len(outputs)} result(s):\n")

    for i, output_dict in enumerate(outputs):
        if len(outputs) > 1:
            print(f"--- Result {i + 1} ---")

        for key, value in output_dict.items():
            processed = process_output_value(value, key)
            if isinstance(processed, (dict, list)):
                print(f"  {key}:")
                formatted = json.dumps(processed, indent=4, default=str)
                # Indent each line
                for line in formatted.split("\n"):
                    print(f"    {line}")
            else:
                print(f"  {key}: {processed}")

        if len(outputs) > 1:
            print()


if __name__ == "__main__":
    main()
