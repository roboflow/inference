#!/usr/bin/env python3
"""Shared HTTP utilities for workflow skill scripts.

Uses inference_sdk's requests session when available, falls back to urllib.
"""

import json
import os
import sys


def _make_urllib_post(url, json_payload=None, params=None, timeout=30):
    """Make a POST request using urllib (no external dependencies)."""
    import urllib.request
    import urllib.error
    from urllib.parse import urlencode

    if params:
        url = url + "?" + urlencode(params)
    data = json.dumps(json_payload).encode() if json_payload else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.getcode(), json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body.decode("utf-8", errors="replace")}


def post_json(url, json_payload=None, params=None, timeout=30):
    """Make a POST request and return (status_code, response_dict).

    Tries the requests library first (available when inference_sdk is installed),
    then falls back to urllib.
    """
    try:
        import requests
        resp = requests.post(url, json=json_payload, params=params, timeout=timeout)
        try:
            data = resp.json()
        except Exception:
            data = {"error": resp.text}
        return resp.status_code, data
    except ImportError:
        return _make_urllib_post(url, json_payload, params, timeout)


def fetch_blocks_describe(api_url, api_key):
    """Fetch the full blocks description from the inference server."""
    url = f"{api_url}/workflows/blocks/describe"
    payload = {}
    if api_key:
        payload["api_key"] = api_key
    status, data = post_json(url, json_payload=payload)
    if status != 200:
        msg = data.get("message", data.get("detail", data.get("error", f"HTTP {status}")))
        raise RuntimeError(f"Failed to fetch blocks: {msg}")
    return data
