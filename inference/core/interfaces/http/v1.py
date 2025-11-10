from __future__ import annotations

import json
from typing import Callable, List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    InferenceRequestImage,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.requests.workflows import (
    WorkflowSpecificationInferenceRequest,
)


def _extract_api_key(request: Request, api_key: Optional[str] = None) -> Optional[str]:
    # Priority: Authorization: Bearer <key> -> X-API-Key -> api_key query
    auth_header = request.headers.get("authorization") or request.headers.get(
        "Authorization"
    )
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
    x_api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if x_api_key:
        return x_api_key
    return api_key


def _files_to_image_payloads(
    files: Optional[List[UploadFile]],
) -> Union[InferenceRequestImage, List[InferenceRequestImage]]:
    if not files:
        raise ValueError("No image files provided.")
    images = [
        InferenceRequestImage(type="multipart", value=upload.file) for upload in files
    ]
    return images[0] if len(images) == 1 else images


def create_v1_router(
    process_inference_request: Callable,
    process_workflow_inference_request: Callable,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/infer/object_detection",
        summary="Object detection inference (multipart)",
    )
    def v1_infer_object_detection(
        request: Request,
        model_id: str = Form(...),
        image: Optional[List[UploadFile]] = File(None),
        # Common OD params (optional)
        confidence: Optional[float] = Form(None),
        iou_threshold: Optional[float] = Form(None),
        max_detections: Optional[int] = Form(None),
        class_agnostic_nms: Optional[bool] = Form(None),
        class_filter: Optional[str] = Form(
            None
        ),  # comma-separated list of classes to include
        fix_batch_size: Optional[bool] = Form(None),
        visualize_predictions: Optional[bool] = Form(None),
        visualization_labels: Optional[bool] = Form(None),
        visualization_stroke_width: Optional[int] = Form(None),
        disable_preproc_auto_orient: Optional[bool] = Form(None),
        disable_preproc_contrast: Optional[bool] = Form(None),
        disable_preproc_grayscale: Optional[bool] = Form(None),
        disable_preproc_static_crop: Optional[bool] = Form(None),
        api_key: Optional[str] = Depends(_extract_api_key),
    ):
        image_payload = _files_to_image_payloads(image)
        class_filter_list = (
            [c.strip() for c in class_filter.split(",")] if class_filter else None
        )
        req = ObjectDetectionInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=image_payload,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter_list,
            fix_batch_size=fix_batch_size,
            visualize_predictions=visualize_predictions,
            visualization_labels=visualization_labels,
            visualization_stroke_width=visualization_stroke_width,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        return process_inference_request(req)

    @router.post(
        "/infer/classification",
        summary="Classification inference (multipart)",
    )
    def v1_infer_classification(
        request: Request,
        model_id: str = Form(...),
        image: Optional[List[UploadFile]] = File(None),
        confidence: Optional[float] = Form(None),
        visualize_predictions: Optional[bool] = Form(None),
        visualization_stroke_width: Optional[int] = Form(None),
        disable_preproc_auto_orient: Optional[bool] = Form(None),
        disable_preproc_contrast: Optional[bool] = Form(None),
        disable_preproc_grayscale: Optional[bool] = Form(None),
        disable_preproc_static_crop: Optional[bool] = Form(None),
        api_key: Optional[str] = Depends(_extract_api_key),
    ):
        image_payload = _files_to_image_payloads(image)
        req = ClassificationInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=image_payload,
            confidence=confidence,
            visualize_predictions=visualize_predictions,
            visualization_stroke_width=visualization_stroke_width,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        return process_inference_request(req)

    @router.post(
        "/workflows/run",
        summary="Run Workflow from specification (multipart image + JSON)",
        description="Accepts multipart/form-data with `spec` JSON and `image` file, plus optional `inputs` JSON for additional parameters.",
    )
    def v1_workflows_run(
        request: Request,
        spec: str = Form(...),
        image: Optional[List[UploadFile]] = File(None),
        inputs: Optional[str] = Form(None),
        api_key: Optional[str] = Depends(_extract_api_key),
    ):
        try:
            specification = json.loads(spec)
        except Exception:
            raise ValueError("Field `spec` must be a valid JSON string.")
        resolved_inputs = {}
        if inputs:
            try:
                resolved_inputs = json.loads(inputs)
            except Exception:
                raise ValueError("Field `inputs` must be a valid JSON string when set.")
        if image:
            # Convention: map to 'image' workflow input name
            image_payload = _files_to_image_payloads(image)
            resolved_inputs["image"] = (
                image_payload
                if isinstance(image_payload, list)
                else {"type": "multipart", "value": image[0].file}
            )
        req = WorkflowSpecificationInferenceRequest(
            api_key=api_key, specification=specification, inputs=resolved_inputs
        )
        # process_workflow_inference_request(workflow_request, workflow_specification, background_tasks, profiler)
        # We don't pass background tasks/profiler in MVP
        return process_workflow_inference_request(
            workflow_request=req,
            workflow_specification=specification,
            background_tasks=None,
            profiler=None,  # type: ignore
        )

    return router


