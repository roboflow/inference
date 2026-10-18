"""Offline HTTP contract observations for the V2 API gap report.

Run from the repository root with the selected checkout's packages on PYTHONPATH.
Authentication, registry lookup, URL fetching and model execution are mocked;
the real app, middleware, routers, parsers, dispatch and serializers execute.
No model downloads or production requests are performed. JSON is written to stdout.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.metadata
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np

import inference_server.app as app_module
from inference_server import configuration
from inference_server.framework.registry import DYNAMIC_MODELS_HANDLERS

_IMAGE = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 12 + b"\xff\xd9"
_KEY = "offline-audit-key"
_BASE64_IMAGE = {"type": "base64", "value": base64.b64encode(_IMAGE).decode()}


def _prediction(task: str) -> object:
    if task == "classification":
        return SimpleNamespace(confidence=np.array([0.2, 0.8]), class_id=np.array([1]))
    if task == "multi-label-classification":
        return SimpleNamespace(confidence=np.array([0.2, 0.8]), class_ids=np.array([1]))
    if task == "semantic-segmentation":
        return SimpleNamespace(
            segmentation_map=np.array([[0, 1]]), confidence=np.array([[0.9, 0.8]])
        )
    if task == "depth-estimation":
        return np.array([[0.1, 0.2]])
    if task == "embedding":
        return np.array([[0.1, 0.2]])
    if task == "text-only-ocr":
        return "audit text"
    return SimpleNamespace(
        xyxy=np.array([[1, 2, 3, 4]]),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
        image_metadata={"class_names": ["cat"]},
        mask=np.array([[[True, False], [False, True]]]),
    )


async def _main() -> None:
    proxy = SimpleNamespace(
        ensure_loaded=AsyncMock(return_value=("model_ready",)),
        infer=AsyncMock(),
        load=AsyncMock(return_value=("ok",)),
        unload=AsyncMock(return_value=("ok",)),
        stats=AsyncMock(return_value={"models": {}}),
        interface=AsyncMock(return_value={"model_id": "audit/1", "actions": {}}),
    )
    app_module.app.state.model_manager = proxy
    observations = []
    original_gate = configuration.ENABLE_CONTROL_PLANE_ROUTES
    configuration.ENABLE_CONTROL_PLANE_ROUTES = True

    async def _observe(
        label: str, method: str, url: str, *, task="object-detection", **kwargs
    ):
        proxy.infer.reset_mock()
        proxy.ensure_loaded.reset_mock()
        proxy.load.reset_mock()
        proxy.infer.return_value = _prediction(task)
        kwargs.setdefault("headers", {"authorization": f"Bearer {_KEY}"})
        with (
            patch(
                "inference_server.framework.dispatch.stat_model_while_checking_auth",
                new=AsyncMock(return_value=(task, "infer")),
            ),
            patch(
                "inference_server.routers.v2_models.stat_model_while_checking_auth",
                new=AsyncMock(return_value=(task, "infer")),
            ),
            patch(
                "inference_server.handlers.object_detection.input_parser.fetch_images_from_urls",
                new=AsyncMock(return_value=([_IMAGE], None)),
            ),
        ):
            response = await client.request(method, url, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = response.text
            if len(body) > 300:
                body = {"text_preview": body[:300], "total_characters": len(body)}
        observation = {
            "case": label,
            "method": method,
            "url": url,
            "status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "body": body,
            "inference_calls": proxy.infer.await_count,
        }
        if proxy.infer.await_args:
            observation["forwarded_params"] = proxy.infer.await_args.kwargs.get(
                "params"
            )
        if proxy.ensure_loaded.await_args:
            args = proxy.ensure_loaded.await_args.args
            observation["ensure_loaded_args"] = [
                args[0],
                args[1],
                "<redacted>",
                args[3],
            ]
        if proxy.load.await_args:
            observation["load_model_id"] = proxy.load.await_args.args[0]
        observations.append(observation)

    try:
        with patch.object(
            app_module, "validate_api_key", new=AsyncMock(return_value=(True, None))
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app_module.app, raise_app_exceptions=False
                ),
                base_url="http://offline-audit",
            ) as client:
                routes = [
                    ("POST", "/v2/models/run?model_id=audit/1"),
                    ("GET", "/v2/models/interface?model_id=audit/1"),
                    ("GET", "/v2/models/compatibility"),
                    ("GET", "/v2/models/loaded"),
                    ("POST", "/v2/models/load?model_id=audit/1"),
                    ("DELETE", "/v2/models/unload?model_id=audit/1"),
                    ("POST", "/v2/workflows/run"),
                    ("POST", "/v2/workflows/interface"),
                    ("POST", "/v2/workflows/validate"),
                    ("GET", "/v2/workflows/system/blocks"),
                    ("GET", "/v2/workflows/system/definition-schema"),
                    ("GET", "/v2/workflows/system/engine-versions"),
                    ("GET", "/v2/server/health"),
                    ("GET", "/v2/server/ready"),
                    ("GET", "/v2/server/info"),
                    ("GET", "/v2/server/metrics"),
                    ("GET", "/v2/models"),
                    ("POST", "/v2/models/unload?model_id=audit/1"),
                    ("DELETE", "/v2/models"),
                ]
                for method, url in routes:
                    await _observe(f"route {method} {url.split('?')[0]}", method, url)

                infer_url = "/v2/models/infer?model_id=audit/1"
                await _observe(
                    "default style and envelope", "POST", infer_url, content=_IMAGE
                )
                await _observe(
                    "explicit rich detection",
                    "POST",
                    infer_url + "&response_style=rich",
                    content=_IMAGE,
                )
                await _observe(
                    "query URL image",
                    "POST",
                    infer_url + "&image=https://example.com/a.jpg&confidence=0.5",
                )
                await _observe(
                    "JSON base64 image",
                    "POST",
                    infer_url,
                    json={"inputs": {"image": _BASE64_IMAGE}},
                )
                await _observe(
                    "JSON URL image from proposal",
                    "POST",
                    infer_url,
                    json={
                        "inputs": {
                            "image": {
                                "type": "url",
                                "value": "https://example.com/a.jpg",
                            }
                        }
                    },
                )
                await _observe(
                    "multipart repeated image",
                    "POST",
                    infer_url,
                    files=[
                        ("image", ("a.jpg", _IMAGE, "image/jpeg")),
                        ("image", ("b.jpg", _IMAGE, "image/jpeg")),
                        ("inputs", (None, '{"confidence":0.5}', "application/json")),
                    ],
                )
                await _observe(
                    "multipart named part reference",
                    "POST",
                    infer_url,
                    files=[
                        ("frame", ("a.jpg", _IMAGE, "image/jpeg")),
                        (
                            "inputs",
                            (None, '{"image":"$part.frame"}', "application/json"),
                        ),
                    ],
                )
                await _observe(
                    "multipart response requested",
                    "POST",
                    infer_url + "&response_format=multipart",
                    content=_IMAGE,
                )
                await _observe(
                    "repeated output filter",
                    "POST",
                    infer_url + "&requested_output=first&requested_output=second",
                    content=_IMAGE,
                )
                await _observe(
                    "model package selection",
                    "POST",
                    infer_url + "&model_package_id=chosen-package",
                    content=_IMAGE,
                )
                await _observe(
                    "query API key only",
                    "POST",
                    infer_url + "&api_key=offline-audit-key",
                    content=_IMAGE,
                    headers={},
                )
                await _observe(
                    "body API key only",
                    "POST",
                    infer_url,
                    json={"api_key": _KEY, "inputs": {"image": _BASE64_IMAGE}},
                    headers={},
                )
                await _observe("JSON non-object", "POST", infer_url, json=[])
                await _observe(
                    "malformed multipart inputs",
                    "POST",
                    infer_url,
                    files=[
                        ("image", ("a.jpg", _IMAGE, "image/jpeg")),
                        ("inputs", (None, "{bad-json", "application/json")),
                    ],
                )

                for task in (
                    "classification",
                    "multi-label-classification",
                    "instance-segmentation",
                    "semantic-segmentation",
                    "depth-estimation",
                    "text-only-ocr",
                ):
                    for style in ("compact", "rich"):
                        await _observe(
                            f"serializer {task} {style}",
                            "POST",
                            infer_url + f"&response_style={style}",
                            task=task,
                            content=_IMAGE,
                        )

                proxy.interface.side_effect = RuntimeError("not loaded")
                await _observe(
                    "unloaded interface", "GET", "/v2/models/interface?model_id=audit/1"
                )
                await _observe(
                    "unloaded interface filters",
                    "GET",
                    "/v2/models/interface?model_id=audit/1&response_style=rich&response_format=multipart&request_format=json_payload",
                )
                proxy.interface.side_effect = None
                configuration.ENABLE_CONTROL_PLANE_ROUTES = False
                await _observe(
                    "default gate load", "POST", "/v2/models/load?model_id=audit/1"
                )
                await _observe("default gate info", "GET", "/v2/server/info")
                await _observe("public health", "GET", "/v2/server/health", headers={})
                await _observe("public ready", "GET", "/v2/server/ready", headers={})
    finally:
        configuration.ENABLE_CONTROL_PLANE_ROUTES = original_gate

    result = {
        "scope": "Offline app and contract probes; fake model execution and auth",
        "source_module": app_module.__file__,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("fastapi", "httpx", "numpy", "scipy", "python-multipart")
        },
        "registered_v2_routes": [
            {"path": route.path, "methods": sorted(route.methods)}
            for route in app_module.app.routes
            if getattr(route, "path", "").startswith("/v2/")
        ],
        "registered_handlers": [list(key) for key in sorted(DYNAMIC_MODELS_HANDLERS)],
        "observations": observations,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
