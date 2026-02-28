#!/usr/bin/env python3
"""Run a workflow against an image and display results.

Uses the inference_sdk InferenceHTTPClient for image encoding, request building,
retries, and output decoding. Falls back to the CLI's image extraction utilities
for saving visualization outputs to disk.
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from setup import get_api_key, get_api_url

try:
    import cv2
    import numpy as np
    from inference_sdk import InferenceHTTPClient
    from inference_cli.lib.workflows.common import (
        extract_images_from_result,
        deduct_images,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


def run_with_sdk(api_url, api_key, spec, image_arg, image_name, parameters, raw):
    """Run workflow using the inference SDK client."""
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    images = {image_name: image_arg}

    try:
        outputs = client.run_workflow(
            specification=spec,
            images=images,
            parameters=parameters,
        )
    except Exception as e:
        print(f"WORKFLOW EXECUTION FAILED:\n  {e}", file=sys.stderr)
        sys.exit(1)

    if not outputs:
        print("Workflow returned no outputs.")
        return

    if raw:
        cleaned = [deduct_images(result=o) for o in outputs]
        print(json.dumps(cleaned, indent=2, default=str))
        return

    print(f"Workflow executed successfully. {len(outputs)} result(s):\n")

    for i, output_dict in enumerate(outputs):
        if len(outputs) > 1:
            print(f"--- Result {i + 1} ---")

        # Extract and save any image outputs to temp files
        image_outputs = extract_images_from_result(result=output_dict)
        saved_images = {}
        if image_outputs:
            tmp_dir = tempfile.mkdtemp(prefix="workflow_output_")
            for image_key, image_array in image_outputs:
                safe_key = image_key.replace("/", "_")
                path = os.path.join(tmp_dir, f"{safe_key}.jpg")
                cv2.imwrite(path, image_array)
                saved_images[image_key] = path

        # Display results with images replaced by file paths
        cleaned = deduct_images(result=output_dict)
        for key, value in cleaned.items():
            # Check if this key had an image extracted
            if key in saved_images:
                print(f"  {key}: [Image saved to: {saved_images[key]}]")
            elif value == "<deducted_image>":
                # Image was in a nested position — find matching saved paths
                matching = {k: v for k, v in saved_images.items() if k.startswith(key)}
                if matching:
                    for img_key, img_path in matching.items():
                        print(f"  {img_key}: [Image saved to: {img_path}]")
                else:
                    print(f"  {key}: <image data>")
            elif isinstance(value, (dict, list)):
                print(f"  {key}:")
                formatted = json.dumps(value, indent=4, default=str)
                for line in formatted.split("\n"):
                    print(f"    {line}")
            else:
                print(f"  {key}: {value}")

        if len(outputs) > 1:
            print()


def run_with_http(api_url, api_key, spec, image_arg, image_name, parameters, raw):
    """Fallback: run workflow using raw HTTP when SDK is not available."""
    import base64
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

    # Prepare image input
    if image_arg.startswith("http://") or image_arg.startswith("https://"):
        image_input = {"type": "url", "value": image_arg}
    elif os.path.isfile(image_arg):
        with open(image_arg, "rb") as f:
            image_input = {"type": "base64", "value": base64.b64encode(f.read()).decode("utf-8")}
    else:
        image_input = {"type": "base64", "value": image_arg}

    inputs = {image_name: image_input}
    inputs.update(parameters)

    payload = {"specification": spec, "inputs": inputs}
    if api_key:
        payload["api_key"] = api_key

    resp = requests.post(f"{api_url}/workflows/run", json=payload)
    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("message", err.get("detail", err.get("error", str(err))))
        except Exception:
            msg = resp.text
        print(f"WORKFLOW EXECUTION FAILED:\n  {msg}", file=sys.stderr)
        sys.exit(1)

    result = resp.json()
    outputs = result.get("outputs", [])
    if not outputs:
        print("Workflow returned no outputs.")
        return

    if raw:
        print(json.dumps(outputs, indent=2, default=str))
        return

    print(f"Workflow executed successfully. {len(outputs)} result(s):\n")
    for i, output_dict in enumerate(outputs):
        if len(outputs) > 1:
            print(f"--- Result {i + 1} ---")
        for key, value in output_dict.items():
            if isinstance(value, dict) and value.get("type") == "base64" and "value" in value:
                img_bytes = base64.b64decode(value["value"])
                tmp_dir = tempfile.mkdtemp(prefix="workflow_output_")
                ext = ".jpg" if img_bytes[:2] == b'\xff\xd8' else ".png"
                path = os.path.join(tmp_dir, f"{key}{ext}")
                with open(path, "wb") as f:
                    f.write(img_bytes)
                print(f"  {key}: [Image saved to: {path}]")
            elif isinstance(value, (dict, list)):
                print(f"  {key}:")
                for line in json.dumps(value, indent=4, default=str).split("\n"):
                    print(f"    {line}")
            else:
                print(f"  {key}: {value}")
        if len(outputs) > 1:
            print()


def main():
    parser = argparse.ArgumentParser(description="Run a workflow against an image")
    parser.add_argument("workflow", help="Path to workflow JSON file, or inline JSON string")
    parser.add_argument("--image", required=True, help="Image file path or URL")
    parser.add_argument("--image-name", default="image", help="Name of the image input (default: 'image')")
    parser.add_argument("--param", action="append", default=[], help="Additional parameters as key=value")
    parser.add_argument("--api-url", help="Override inference server URL")
    parser.add_argument("--api-key", help="Override API key")
    parser.add_argument("--raw", action="store_true", help="Output raw JSON (images stripped)")
    args = parser.parse_args()

    api_url = args.api_url or get_api_url()
    api_key = args.api_key or get_api_key()

    if not api_key:
        print("Error: No API key configured.", file=sys.stderr)
        print("Run: python3 setup.py set --api-key <KEY> --workspace <NAME>", file=sys.stderr)
        sys.exit(1)

    # Load workflow spec
    if args.workflow.startswith("{"):
        spec = json.loads(args.workflow)
    elif os.path.isfile(args.workflow):
        with open(args.workflow) as f:
            spec = json.load(f)
    else:
        print(f"Error: '{args.workflow}' is not a valid file or JSON.", file=sys.stderr)
        sys.exit(1)

    # Parse extra parameters
    parameters = {}
    for param in args.param:
        if "=" not in param:
            print(f"Error: parameter '{param}' must be key=value.", file=sys.stderr)
            sys.exit(1)
        key, value = param.split("=", 1)
        try:
            parameters[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            parameters[key] = value

    # Use SDK if available, otherwise fall back to raw HTTP
    runner = run_with_sdk if SDK_AVAILABLE else run_with_http
    runner(api_url, api_key, spec, args.image, args.image_name, parameters, args.raw)


if __name__ == "__main__":
    main()
