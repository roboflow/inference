import base64
import io
from unittest.mock import AsyncMock

import pytest
from fastapi.routing import APIRoute
from PIL import Image

from inference_server.legacy.errors import NOT_FOUND_MESSAGE
from tests.unit_tests.legacy.conftest import FakeGateway

MARKER = "capabilities="
POISONED_MODEL_IDS = [
    "ds/1:capabilities=image_embeddings;output_type=logits",
    "ds/1:capabilities=image_embeddings:output_type=logits",
    "ds/1:b:capabilities=image_embeddings",
]
POISONED_INSTANCES = [
    "capabilities=image_embeddings",
    "b:capabilities=image_embeddings",
]
MODEL_ID_PATH_PARAMS = {"model_id", "dataset_id", "version_id"}
MODEL_ID_QUERY_PARAMS = {"model_id", "instance"}
UNDECLARED_QUERY_PARAMS = {
    "/v2/models/infer": ("model_id", "instance"),
    "/v2/models/load": ("model_id",),
    "/v2/models/unload": ("model_id",),
    "/v2/models/interface": ("model_id",),
}
RAW_IMAGE_BODY_ROUTES = {"/v2/models/infer"}
FORM_IMAGE_BODY_ROUTES = {"/{dataset_id}/{version_id}"}
ACCEPTED_WITHOUT_LOAD = {"/model/remove"}
ROUTES_RESOLVING_THE_REQUEST_MODEL = {
    "/infer/object_detection",
    "/infer/instance_segmentation",
    "/infer/semantic_segmentation",
    "/infer/classification",
    "/infer/embeddings",
    "/infer/keypoints_detection",
    "/infer/lmm",
    "/infer/lmm/{model_id:path}",
    "/infer/depth-estimation/{model_id:path}",
    "/model/add",
    "/model/remove",
    "/start/{dataset_id}/{version_id}",
    "/{dataset_id}/{version_id}",
    "/v2/models/infer",
    "/v2/models/load",
    "/v2/models/unload",
    "/v2/models/interface",
}
ROUTES_DERIVING_THEIR_MODEL = {
    "/clip/compare",
    "/clip/embed_image",
    "/clip/embed_text",
    "/doctr/ocr",
    "/easy_ocr/ocr",
    "/grounding_dino/infer",
    "/ocr/trocr",
    "/perception_encoder/compare",
    "/perception_encoder/embed_image",
    "/perception_encoder/embed_text",
    "/ocr/pp-ocr",
    "/sam/embed_image",
    "/sam/segment_image",
    "/sam2/embed_image",
    "/sam2/segment_image",
    "/sam3/concept_segment",
    "/sam3/embed_image",
    "/sam3/visual_segment",
    "/yolo_world/infer",
    "/infer/action_recognition",
    "/infer/depth-estimation",
}
ROUTES_TAKING_MODELS_INSIDE_WORKFLOW_SPECS = {
    "/infer/workflows",
    "/infer/workflows/{workspace_name}/{workflow_id}",
    "/workflows/blocks/describe",
    "/workflows/blocks/dynamic_outputs",
    "/workflows/describe_interface",
    "/workflows/describe_workload",
    "/workflows/run",
    "/workflows/validate",
    "/{workspace_name}/workflows/{workflow_id}",
    "/{workspace_name}/workflows/{workflow_id}/describe_interface",
    "/{workspace_name}/workflows/{workflow_id}/describe_workload",
}
ROUTES_WITHOUT_MODEL_INPUT = {
    "/clear_cache",
    "/dashboard.html",
    "/device/stats",
    "/gaze/gaze_detection",
    "/healthz",
    "/info",
    "/logs",
    "/metrics",
    "/model/clear",
    "/model/registry",
    "/notebook/start",
    "/readiness",
    "/v2/models",
    "/v2/models/compatibility",
    "/v2/server/health",
    "/v2/server/info",
    "/v2/server/metrics",
    "/v2/server/ready",
    "/workflows/definition/schema",
    "/workflows/execution_engine/versions",
}


def _jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _filler(name: str):
    image = {"type": "base64", "value": base64.b64encode(_jpeg()).decode()}
    fillers = {
        "image": image,
        "video": image,
        "subject": image,
        "subject_type": "image",
        "prompt": "describe",
        "prompt_type": "text",
        "prompts": [],
        "text": ["a"],
        "mask_input": None,
        "api_key": "k",
    }
    return fillers.get(name, "x")


def _body_model(route: APIRoute):
    for field in route.dependant.body_params:
        annotation = field.field_info.annotation
        fields = getattr(annotation, "model_fields", None)
        if fields and "model_id" in fields:
            return annotation
    return None


def _body(model_class, model_id: str) -> dict:
    body = {"model_id": model_id, "api_key": "k"}
    for name, field in model_class.model_fields.items():
        if name not in body and field.is_required():
            body[name] = _filler(name)
    return body


def _declared_inputs(route: APIRoute):
    path = {p.name for p in route.dependant.path_params} & MODEL_ID_PATH_PARAMS
    query = {q.name for q in route.dependant.query_params} & MODEL_ID_QUERY_PARAMS
    query |= set(UNDECLARED_QUERY_PARAMS.get(route.path, ()))
    body = _body_model(route)
    return path, query, body


def _requests_for(route: APIRoute, model_id: str):
    path_params, query_params, body_model = _declared_inputs(route)
    if not (path_params or query_params or body_model):
        return []
    methods = route.methods - {"HEAD", "OPTIONS"}
    method = "POST" if "POST" in methods else sorted(methods)[0]
    requests = []
    base_body = _body(body_model, "ds/1") if body_model else None
    plain_path = route.path.replace("{model_id:path}", "ds/1")
    plain_path = plain_path.replace("{dataset_id}", "ds").replace("{version_id}", "1")
    body_path = plain_path
    if path_params:
        dataset_id, _, version_id = model_id.partition("/")
        path = route.path.replace("{model_id:path}", model_id)
        path = path.replace("{dataset_id}", dataset_id).replace(
            "{version_id}", version_id
        )
        body_path = path
        poisoned_body = _body(body_model, model_id) if body_model else None
        requests.append((method, path, {"api_key": "k"}, poisoned_body, "model_id"))
    if "model_id" in query_params:
        requests.append(
            (method, plain_path, {"model_id": model_id}, base_body, "model_id")
        )
    if "instance" in query_params:
        for instance in POISONED_INSTANCES:
            params = {"model_id": "ds/1", "instance": instance}
            requests.append((method, plain_path, params, base_body, "instance"))
    if body_model:
        requests.append(
            (
                method,
                body_path,
                {"api_key": "k"},
                _body(body_model, model_id),
                "model_id",
            )
        )
    return requests


def _api_routes(routes):
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _api_routes(inner.routes)
        elif isinstance(route, APIRoute):
            yield route


def _marker_calls(gateway: FakeGateway):
    return [
        call for call in gateway.calls if len(call) > 1 and MARKER in str(call[1])
    ] + [key for key in gateway.loaded if MARKER in key]


def _guard_answer(route: APIRoute, poisoned: str, response) -> bool:
    if route.path in ACCEPTED_WITHOUT_LOAD:
        return response.status_code == 200
    if not response.headers.get("content-type", "").startswith("application/json"):
        return False
    if route.path.startswith("/v2/"):
        body = response.json()
        if poisoned == "instance":
            return (response.status_code, body.get("error_code")) == (
                400,
                "INVALID_PARAM",
            ) and "instance" in body.get("description", "")
        return (response.status_code, body.get("error_code")) == (
            404,
            "MODEL_NOT_FOUND",
        )
    return response.status_code == 404 and response.json() == {
        "message": NOT_FOUND_MESSAGE
    }


def _send(client, route: APIRoute, method, path, params, body):
    if route.path in RAW_IMAGE_BODY_ROUTES:
        return client.request(
            method,
            path,
            params=params,
            content=_jpeg(),
            headers={"Authorization": "Bearer k", "Content-Type": "image/jpeg"},
        )
    if route.path in FORM_IMAGE_BODY_ROUTES:
        return client.request(
            method,
            path,
            params=params,
            content=base64.b64encode(_jpeg()),
            headers={
                "Authorization": "Bearer k",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    if body is not None and "image" in body:
        body["image"] = _filler("image")
    return client.request(
        method, path, params=params, json=body, headers={"Authorization": "Bearer k"}
    )


@pytest.mark.parametrize("model_id", POISONED_MODEL_IDS)
def test_no_route_lets_a_capability_marker_reach_the_gateway(
    legacy_client, fake_stat, monkeypatch, model_id
):
    import inference_server.app as app_module

    monkeypatch.setattr(app_module._cfg, "ENABLE_CONTROL_PLANE_ROUTES", True)
    monkeypatch.setattr(
        app_module, "validate_api_key", AsyncMock(return_value=(True, "ws-1"))
    )
    fake_stat[model_id] = ("classification", "infer")
    fake_stat["ds/1"] = ("classification", "infer")
    gateway = FakeGateway()
    client = legacy_client(gateway)

    all_routes = set()
    swept = set()
    not_refused = {}
    for route in _api_routes(client.app.routes):
        all_routes.add(route.path)
        for method, path, params, body, poisoned in _requests_for(route, model_id):
            response = _send(client, route, method, path, params, body)
            swept.add(route.path)
            if route.path in ROUTES_RESOLVING_THE_REQUEST_MODEL and not _guard_answer(
                route, poisoned, response
            ):
                not_refused[(method, route.path, tuple(sorted(params)))] = (
                    response.status_code,
                    response.text[:120],
                )
        assert _marker_calls(gateway) == [], (route.path, gateway.calls)

    assert not not_refused, "\n".join(
        f"{request}: {answer}" for request, answer in sorted(not_refused.items())
    )
    assert swept == ROUTES_RESOLVING_THE_REQUEST_MODEL | ROUTES_DERIVING_THEIR_MODEL
    assert all_routes == (
        swept | ROUTES_TAKING_MODELS_INSIDE_WORKFLOW_SPECS | ROUTES_WITHOUT_MODEL_INPUT
    )
