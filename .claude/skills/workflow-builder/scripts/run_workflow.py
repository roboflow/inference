#!/usr/bin/env python3
"""Run a workflow against an image and display results.

Uses only Python standard library + optional 'requests'. Image handling and
output processing patterns are adapted from the inference SDK/CLI but
reimplemented without those dependencies.
"""

import argparse
import base64
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from setup import get_api_key, get_api_url
from http_utils import post_json

BASE64_DATA_URI_PATTERN = re.compile(r"^data:image/[a-z]+;base64,")


def prepare_image_input(image_arg):
    """Convert an image file path, URL, or base64 string to API input format.

    Mirrors inference_sdk's load_static_inference_input / inject_nested_batches.
    """
    if image_arg.startswith(("http://", "https://")):
        return {"type": "url", "value": image_arg}
    elif os.path.isfile(image_arg):
        with open(image_arg, "rb") as f:
            return {"type": "base64", "value": base64.b64encode(f.read()).decode("ascii")}
    else:
        # Assume raw base64
        return {"type": "base64", "value": image_arg}


def is_workflow_image(value):
    """Check if a value is a workflow image dict (base64-encoded).

    Mirrors inference_sdk.http.utils.post_processing.is_workflow_image.
    """
    return isinstance(value, dict) and value.get("type") == "base64" and "value" in value


def save_base64_image(b64_data, name):
    """Decode base64 image data, save to a temp file, return the path."""
    cleaned = BASE64_DATA_URI_PATTERN.sub("", b64_data)
    try:
        img_bytes = base64.b64decode(cleaned)
    except Exception:
        return None
    ext = ".jpg" if img_bytes[:2] == b"\xff\xd8" else ".png"
    tmp_dir = tempfile.mkdtemp(prefix="workflow_output_")
    path = os.path.join(tmp_dir, f"{name}{ext}")
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path


def extract_and_save_images(result, key_prefix=""):
    """Walk a result dict/list, save any embedded images to temp files.

    Returns dict mapping key paths to saved file paths.
    Mirrors inference_cli.lib.workflows.common.extract_images_from_result.
    """
    saved = {}
    if is_workflow_image(result):
        path = save_base64_image(result["value"], key_prefix.replace("/", "_") or "image")
        if path:
            saved[key_prefix] = path
    elif isinstance(result, dict):
        for key, value in result.items():
            child_key = f"{key_prefix}/{key}".lstrip("/")
            saved.update(extract_and_save_images(value, child_key))
    elif isinstance(result, list):
        for idx, element in enumerate(result):
            child_key = f"{key_prefix}/{idx}".lstrip("/")
            saved.update(extract_and_save_images(element, child_key))
    return saved


def strip_images(result):
    """Replace image data in a result with a placeholder string.

    Mirrors inference_cli.lib.workflows.common.deduct_images.
    """
    if is_workflow_image(result):
        return "<image>"
    elif isinstance(result, dict):
        return {k: strip_images(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [strip_images(e) for e in result]
    return result


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

    # Build inputs dict
    inputs = {args.image_name: prepare_image_input(args.image)}
    for param in args.param:
        if "=" not in param:
            print(f"Error: parameter '{param}' must be key=value.", file=sys.stderr)
            sys.exit(1)
        key, value = param.split("=", 1)
        try:
            inputs[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            inputs[key] = value

    # Execute
    payload = {"specification": spec, "inputs": inputs}
    if api_key:
        payload["api_key"] = api_key

    status, result = post_json(f"{api_url}/workflows/run", json_payload=payload, timeout=120)

    if status != 200:
        msg = result.get("message", result.get("detail", result.get("error", f"HTTP {status}")))
        print(f"WORKFLOW EXECUTION FAILED:\n  {msg}", file=sys.stderr)
        sys.exit(1)

    outputs = result.get("outputs", [])
    if not outputs:
        print("Workflow returned no outputs.")
        return

    if args.raw:
        print(json.dumps([strip_images(o) for o in outputs], indent=2, default=str))
        return

    print(f"Workflow executed successfully. {len(outputs)} result(s):\n")

    for i, output_dict in enumerate(outputs):
        if len(outputs) > 1:
            print(f"--- Result {i + 1} ---")

        # Extract and save any images
        saved_images = extract_and_save_images(output_dict)

        # Display results
        cleaned = strip_images(output_dict)
        for key, value in cleaned.items():
            # Check for saved images at this key
            matching_paths = {k: v for k, v in saved_images.items()
                             if k == key or k.startswith(key + "/")}
            if matching_paths:
                for img_key, img_path in matching_paths.items():
                    label = key if img_key == key else img_key
                    print(f"  {label}: [Image saved to: {img_path}]")
            elif isinstance(value, (dict, list)):
                print(f"  {key}:")
                for line in json.dumps(value, indent=4, default=str).split("\n"):
                    print(f"    {line}")
            else:
                print(f"  {key}: {value}")

        if len(outputs) > 1:
            print()


if __name__ == "__main__":
    main()
