"""Core model HTTP routes (CLIP, PE, Gaze, Grounding DINO, YOLO World, SAM, etc.)."""

from functools import partial
from typing import List, Optional, Union

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from inference.core import logger
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderCompareRequest,
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.core.entities.requests.sam import SamEmbeddingRequest, SamSegmentationRequest
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2SegmentationRequest,
)
from inference.core.entities.requests.sam3 import Sam3SegmentationRequest
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.entities.requests.trocr import TrOCRInferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.entities.requests.clip import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
)
from inference.core.entities.requests.gaze import GazeDetectionInferenceRequest
from inference.core.entities.responses.clip import ClipCompareResponse, ClipEmbeddingResponse
from inference.core.entities.responses.gaze import GazeDetectionInferenceResponse
from inference.core.entities.responses.inference import ObjectDetectionInferenceResponse
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.entities.responses.perception_encoder import (
    PerceptionEncoderCompareResponse,
    PerceptionEncoderEmbeddingResponse,
)
from inference.core.entities.responses.sam import (
    SamEmbeddingResponse,
    SamSegmentationResponse,
)
from inference.core.entities.responses.sam2 import (
    Sam2EmbeddingResponse,
    Sam2SegmentationResponse,
)
from inference.core.entities.responses.sam3 import (
    Sam3EmbeddingResponse,
    Sam3SegmentationResponse,
)
from inference.core.env import (
    API_BASE_URL,
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_DOCTR_ENABLED,
    CORE_MODEL_EASYOCR_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_GROUNDINGDINO_ENABLED,
    CORE_MODEL_OWLV2_ENABLED,
    CORE_MODEL_PE_ENABLED,
    CORE_MODEL_SAM2_ENABLED,
    CORE_MODEL_SAM3_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODEL_TROCR_ENABLED,
    CORE_MODEL_YOLO_WORLD_ENABLED,
    GCP_SERVERLESS,
    LAMBDA,
    SAM3_EXEC_MODE,
    SAM3_FINE_TUNED_MODELS_ENABLED,
)
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.interfaces.http.orjson_utils import (
    orjson_response,
    orjson_response_keeping_parent_id,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import ModelEndpointType
from inference.usage_tracking.collector import usage_collector

if LAMBDA:
    from inference.core.usage import trackUsage


def create_core_models_router(model_manager: ModelManager) -> APIRouter:
    router = APIRouter()

    def load_core_model(
        inference_request: InferenceRequest,
        api_key: Optional[str] = None,
        core_model: str = None,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> str:
        if api_key:
            inference_request.api_key = api_key
        version_id_field = f"{core_model}_version_id"
        core_model_id = (
            f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
        )
        model_manager.add_model(
            core_model_id,
            inference_request.api_key,
            endpoint_type=ModelEndpointType.CORE_MODEL,
            countinference=countinference,
            service_secret=service_secret,
        )
        return core_model_id

    load_clip_model = partial(load_core_model, core_model="clip")
    load_pe_model = partial(load_core_model, core_model="perception_encoder")
    load_sam_model = partial(load_core_model, core_model="sam")
    load_sam2_model = partial(load_core_model, core_model="sam2")
    load_gaze_model = partial(load_core_model, core_model="gaze")
    load_doctr_model = partial(load_core_model, core_model="doctr")
    load_easy_ocr_model = partial(load_core_model, core_model="easy_ocr")
    load_grounding_dino_model = partial(load_core_model, core_model="grounding_dino")
    load_yolo_world_model = partial(load_core_model, core_model="yolo_world")
    load_owlv2_model = partial(load_core_model, core_model="owlv2")
    load_trocr_model = partial(load_core_model, core_model="trocr")
    load_paligemma_model = partial(load_core_model, core_model="paligemma")

    if CORE_MODEL_CLIP_ENABLED:

        @router.post(
            "/clip/embed_image",
            response_model=ClipEmbeddingResponse,
            summary="CLIP Image Embeddings",
            description="Run the Open AI CLIP model to embed image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def clip_embed_image(
            inference_request: ClipImageEmbeddingRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds image data using the OpenAI CLIP model.

            Args:
                inference_request (ClipImageEmbeddingRequest): The request containing the image to be embedded.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                ClipEmbeddingResponse: The response containing the embedded image.
            """
            logger.debug(f"Reached /clip/embed_image")
            clip_model_id = load_clip_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                clip_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(clip_model_id, actor)
            return response

        @router.post(
            "/clip/embed_text",
            response_model=ClipEmbeddingResponse,
            summary="CLIP Text Embeddings",
            description="Run the Open AI CLIP model to embed text data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def clip_embed_text(
            inference_request: ClipTextEmbeddingRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds text data using the OpenAI CLIP model.

            Args:
                inference_request (ClipTextEmbeddingRequest): The request containing the text to be embedded.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                ClipEmbeddingResponse: The response containing the embedded text.
            """
            logger.debug(f"Reached /clip/embed_text")
            clip_model_id = load_clip_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                clip_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(clip_model_id, actor)
            return response

        @router.post(
            "/clip/compare",
            response_model=ClipCompareResponse,
            summary="CLIP Compare",
            description="Run the Open AI CLIP model to compute similarity scores.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def clip_compare(
            inference_request: ClipCompareRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Computes similarity scores using the OpenAI CLIP model.

            Args:
                inference_request (ClipCompareRequest): The request containing the data to be compared.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                ClipCompareResponse: The response containing the similarity scores.
            """
            logger.debug(f"Reached /clip/compare")
            clip_model_id = load_clip_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                clip_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(clip_model_id, actor, n=2)
            return response

    if CORE_MODEL_PE_ENABLED:

        @router.post(
            "/perception_encoder/embed_image",
            response_model=PerceptionEncoderEmbeddingResponse,
            summary="PE Image Embeddings",
            description="Run the Meta Perception Encoder model to embed image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def pe_embed_image(
            inference_request: PerceptionEncoderImageEmbeddingRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds image data using the Perception Encoder PE model.

            Args:
                inference_request (PerceptionEncoderImageEmbeddingRequest): The request containing the image to be embedded.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                PerceptionEncoderEmbeddingResponse: The response containing the embedded image.
            """
            logger.debug(f"Reached /perception_encoder/embed_image")
            pe_model_id = load_pe_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                pe_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(pe_model_id, actor)
            return response

        @router.post(
            "/perception_encoder/embed_text",
            response_model=PerceptionEncoderEmbeddingResponse,
            summary="Perception Encoder Text Embeddings",
            description="Run the Meta Perception Encoder model to embed text data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def pe_embed_text(
            inference_request: PerceptionEncoderTextEmbeddingRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds text data using the Meta Perception Encoder model.

            Args:
                inference_request (PerceptionEncoderTextEmbeddingRequest): The request containing the text to be embedded.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                PerceptionEncoderEmbeddingResponse: The response containing the embedded text.
            """
            logger.debug(f"Reached /perception_encoder/embed_text")
            pe_model_id = load_pe_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                pe_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(pe_model_id, actor)
            return response

        @router.post(
            "/perception_encoder/compare",
            response_model=PerceptionEncoderCompareResponse,
            summary="Perception Encoder Compare",
            description="Run the Meta Perception Encoder model to compute similarity scores.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def pe_compare(
            inference_request: PerceptionEncoderCompareRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Computes similarity scores using the Meta Perception Encoder model.

            Args:
                inference_request (PerceptionEncoderCompareRequest): The request containing the data to be compared.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                PerceptionEncoderCompareResponse: The response containing the similarity scores.
            """
            logger.debug(f"Reached /perception_encoder/compare")
            pe_model_id = load_pe_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                pe_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(pe_model_id, actor, n=2)
            return response

    if CORE_MODEL_GROUNDINGDINO_ENABLED:

        @router.post(
            "/grounding_dino/infer",
            response_model=ObjectDetectionInferenceResponse,
            summary="Grounding DINO inference.",
            description="Run the Grounding DINO zero-shot object detection model.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def grounding_dino_infer(
            inference_request: GroundingDINOInferenceRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds image data using the Grounding DINO model.

            Args:
                inference_request GroundingDINOInferenceRequest): The request containing the image on which to run object detection.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                ObjectDetectionInferenceResponse: The object detection response.
            """
            logger.debug(f"Reached /grounding_dino/infer")
            grounding_dino_model_id = load_grounding_dino_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                grounding_dino_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(grounding_dino_model_id, actor)
            return response

    if CORE_MODEL_YOLO_WORLD_ENABLED:

        @router.post(
            "/yolo_world/infer",
            response_model=ObjectDetectionInferenceResponse,
            summary="YOLO-World inference.",
            description="Run the YOLO-World zero-shot object detection model.",
            response_model_exclude_none=True,
        )
        @with_route_exceptions
        @usage_collector("request")
        def yolo_world_infer(
            inference_request: YOLOWorldInferenceRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Runs the YOLO-World zero-shot object detection model.

            Args:
                inference_request (YOLOWorldInferenceRequest): The request containing the image on which to run object detection.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                ObjectDetectionInferenceResponse: The object detection response.
            """
            logger.debug(f"Reached /yolo_world/infer. Loading model")
            yolo_world_model_id = load_yolo_world_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            logger.debug("YOLOWorld model loaded. Staring the inference.")
            response = model_manager.infer_from_request_sync(
                yolo_world_model_id, inference_request
            )
            logger.debug("YOLOWorld prediction available.")
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(yolo_world_model_id, actor)
                logger.debug("Usage of YOLOWorld denoted.")
            return response

    if CORE_MODEL_DOCTR_ENABLED:

        @router.post(
            "/doctr/ocr",
            response_model=Union[
                OCRInferenceResponse, List[OCRInferenceResponse]
            ],
            summary="DocTR OCR response",
            description="Run the DocTR OCR model to retrieve text in an image.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def doctr_retrieve_text(
            inference_request: DoctrOCRInferenceRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds image data using the DocTR model.

            Args:
                inference_request (M.DoctrOCRInferenceRequest): The request containing the image from which to retrieve text.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                OCRInferenceResponse: The response containing the embedded image.
            """
            logger.debug(f"Reached /doctr/ocr")
            doctr_model_id = load_doctr_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                doctr_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(doctr_model_id, actor)
            return orjson_response_keeping_parent_id(response)

    if CORE_MODEL_EASYOCR_ENABLED:

        @router.post(
            "/easy_ocr/ocr",
            response_model=Union[
                OCRInferenceResponse, List[OCRInferenceResponse]
            ],
            summary="EasyOCR OCR response",
            description="Run the EasyOCR model to retrieve text in an image.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def easy_ocr_retrieve_text(
            inference_request: EasyOCRInferenceRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds image data using the EasyOCR model.

            Args:
                inference_request (EasyOCRInferenceRequest): The request containing the image from which to retrieve text.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                OCRInferenceResponse: The response containing the embedded image.
            """
            logger.debug(f"Reached /easy_ocr/ocr")
            easy_ocr_model_id = load_easy_ocr_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                easy_ocr_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(easy_ocr_model_id, actor)
            return orjson_response_keeping_parent_id(response)

    if CORE_MODEL_SAM_ENABLED:

        @router.post(
            "/sam/embed_image",
            response_model=SamEmbeddingResponse,
            summary="SAM Image Embeddings",
            description="Run the Meta AI Segmant Anything Model to embed image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def sam_embed_image(
            inference_request: SamEmbeddingRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds image data using the Meta AI Segmant Anything Model (SAM).

            Args:
                inference_request (SamEmbeddingRequest): The request containing the image to be embedded.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                M.SamEmbeddingResponse or Response: The response containing the embedded image.
            """
            logger.debug(f"Reached /sam/embed_image")
            sam_model_id = load_sam_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            model_response = model_manager.infer_from_request_sync(
                sam_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(sam_model_id, actor)
            if inference_request.format == "binary":
                return Response(
                    content=model_response.embeddings,
                    headers={"Content-Type": "application/octet-stream"},
                )
            return model_response

        @router.post(
            "/sam/segment_image",
            response_model=SamSegmentationResponse,
            summary="SAM Image Segmentation",
            description="Run the Meta AI Segmant Anything Model to generate segmenations for image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def sam_segment_image(
            inference_request: SamSegmentationRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Generates segmentations for image data using the Meta AI Segmant Anything Model (SAM).

            Args:
                inference_request (SamSegmentationRequest): The request containing the image to be segmented.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                M.SamSegmentationResponse or Response: The response containing the segmented image.
            """
            logger.debug(f"Reached /sam/segment_image")
            sam_model_id = load_sam_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            model_response = model_manager.infer_from_request_sync(
                sam_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(sam_model_id, actor)
            if inference_request.format == "binary":
                return Response(
                    content=model_response,
                    headers={"Content-Type": "application/octet-stream"},
                )
            return model_response

    if CORE_MODEL_SAM2_ENABLED:

        @router.post(
            "/sam2/embed_image",
            response_model=Sam2EmbeddingResponse,
            summary="SAM2 Image Embeddings",
            description="Run the Meta AI Segment Anything 2 Model to embed image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def sam2_embed_image(
            inference_request: Sam2EmbeddingRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds image data using the Meta AI Segment Anything Model (SAM).

            Args:
                inference_request (SamEmbeddingRequest): The request containing the image to be embedded.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                M.Sam2EmbeddingResponse or Response: The response affirming the image has been embedded
            """
            logger.debug(f"Reached /sam2/embed_image")
            sam2_model_id = load_sam2_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            model_response = model_manager.infer_from_request_sync(
                sam2_model_id, inference_request
            )
            return model_response

        @router.post(
            "/sam2/segment_image",
            response_model=Sam2SegmentationResponse,
            summary="SAM2 Image Segmentation",
            description="Run the Meta AI Segment Anything 2 Model to generate segmenations for image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def sam2_segment_image(
            inference_request: Sam2SegmentationRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Generates segmentations for image data using the Meta AI Segment Anything Model (SAM).

            Args:
                inference_request (Sam2SegmentationRequest): The request containing the image to be segmented.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                M.SamSegmentationResponse or Response: The response containing the segmented image.
            """
            logger.debug(f"Reached /sam2/segment_image")
            sam2_model_id = load_sam2_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            model_response = model_manager.infer_from_request_sync(
                sam2_model_id, inference_request
            )
            if inference_request.format == "binary":
                return Response(
                    content=model_response,
                    headers={"Content-Type": "application/octet-stream"},
                )
            return model_response

    if CORE_MODEL_SAM3_ENABLED and not GCP_SERVERLESS:

        @router.post(
            "/sam3/embed_image",
            response_model=Sam3EmbeddingResponse,
            summary="Seg preview Image Embeddings",
            description="Run the  Model to embed image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def sam3_embed_image(
            inference_request: Sam2EmbeddingRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            logger.debug(f"Reached /sam3/embed_image")

            if SAM3_EXEC_MODE == "remote":
                raise HTTPException(
                    status_code=501,
                    detail="SAM3 embedding is not supported in remote execution mode.",
                )

            model_manager.add_model(
                "sam3/sam3_interactive",
                api_key=api_key,
                endpoint_type=ModelEndpointType.CORE_MODEL,
                countinference=countinference,
                service_secret=service_secret,
            )

            model_response = model_manager.infer_from_request_sync(
                "sam3/sam3_interactive", inference_request
            )
            return model_response

    if CORE_MODEL_SAM3_ENABLED:

        @router.post(
            "/sam3/concept_segment",
            response_model=Sam3SegmentationResponse,
            summary="SAM3 PCS (promptable concept segmentation)",
            description="Run the SAM3 PCS (promptable concept segmentation) to generate segmentations for image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def sam3_segment_image(
            inference_request: Sam3SegmentationRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
        if not SAM3_FINE_TUNED_MODELS_ENABLED:
                    if not inference_request.model_id.startswith("sam3/"):
                        raise HTTPException(
                            status_code=501,
                            detail="Fine-tuned SAM3 models are not supported on this deployment. Please use a workflow or self-host the server.",
                        )

            if SAM3_EXEC_MODE == "remote":
                if not inference_request.model_id.startswith("sam3/"):
                    raise HTTPException(
                        status_code=501,
                        detail="Fine-tuned SAM3 models are not supported in remote execution mode yet. Please use a workflow or self-host the server.",
                    )
                endpoint = f"{API_BASE_URL}/inferenceproxy/seg-preview"

                # Construct payload for remote API
                # The remote API expects:
                # {
                #     "image": {"type": "base64", "value": ...},
                #     "prompts": [{"type": "text", "text": ...}, ...],
                #     "output_prob_thresh": ...
                # }

                # Extract prompts from request
                http_prompts = []
                for prompt in inference_request.prompts:
                    p_dict = prompt.dict(exclude_none=True)
                    # Ensure type is set if missing (default to text if text is present)
                    if "type" not in p_dict:
                        if "text" in p_dict:
                            p_dict["type"] = "text"
                    http_prompts.append(p_dict)

                # Prepare image
                # inference_request.image is InferenceRequestImage
                if inference_request.image.type == "base64":
                    http_image = {
                        "type": "base64",
                        "value": inference_request.image.value,
                    }
                elif inference_request.image.type == "url":
                    http_image = {
                        "type": "url",
                        "value": inference_request.image.value,
                    }
                elif inference_request.image.type == "numpy":
                    # Numpy not supported for remote proxy easily without serialization,
                    # but InferenceRequestImage usually comes as base64/url in HTTP API.
                    # If it is numpy, we might need to handle it, but for now assume base64/url.
                    # If it's numpy, it's likely from internal call, but this is HTTP API.
                    http_image = {
                        "type": "numpy",
                        "value": inference_request.image.value,
                    }
                else:
                    http_image = {
                        "type": inference_request.image.type,
                        "value": inference_request.image.value,
                    }

                payload = {
                    "image": http_image,
                    "prompts": http_prompts,
                    "output_prob_thresh": inference_request.output_prob_thresh,
                }

                try:
                    headers = {"Content-Type": "application/json"}
                    if ROBOFLOW_INTERNAL_SERVICE_NAME:
                        headers["X-Roboflow-Internal-Service-Name"] = (
                            ROBOFLOW_INTERNAL_SERVICE_NAME
                        )
                    if ROBOFLOW_INTERNAL_SERVICE_SECRET:
                        headers["X-Roboflow-Internal-Service-Secret"] = (
                            ROBOFLOW_INTERNAL_SERVICE_SECRET
                        )

                    headers = build_roboflow_api_headers(
                        explicit_headers=headers
                    )

                    response = requests.post(
                        f"{endpoint}?api_key={api_key}",
                        json=payload,
                        headers=headers,
                        timeout=60,
                    )
                    response.raise_for_status()
                    resp_json = response.json()

                    # The remote API returns the same structure as Sam3SegmentationResponse
                    return Sam3SegmentationResponse(**resp_json)

                except Exception as e:
                    logger.error(f"SAM3 remote request failed: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"SAM3 remote request failed: {str(e)}",
                    )

            if inference_request.model_id.startswith("sam3/"):
                model_manager.add_model(
                    inference_request.model_id,
                    api_key=api_key,
                    endpoint_type=ModelEndpointType.CORE_MODEL,
                    countinference=countinference,
                    service_secret=service_secret,
                )
            else:
                model_manager.add_model(
                    inference_request.model_id,
                    api_key=api_key,
                    endpoint_type=ModelEndpointType.ORT,
                    countinference=countinference,
                    service_secret=service_secret,
                )

            model_response = model_manager.infer_from_request_sync(
                inference_request.model_id, inference_request
            )
            if inference_request.format == "binary":
                return Response(
                    content=model_response,
                    headers={"Content-Type": "application/octet-stream"},
                )
            return model_response

        @router.post(
            "/sam3/visual_segment",
            response_model=Sam2SegmentationResponse,
            summary="SAM3 PVS (promptable visual segmentation)",
            description="Run the SAM3 PVS (promptable visual segmentation) to generate segmentations for image data.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def sam3_visual_segment(
            inference_request: Sam2SegmentationRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            logger.debug(f"Reached /sam3/visual_segment")

            if SAM3_EXEC_MODE == "remote":
                endpoint = f"{API_BASE_URL}/inferenceproxy/sam3-pvs"

                http_image = {
                    "type": inference_request.image.type,
                    "value": inference_request.image.value,
                }

                prompts_data = (
                    inference_request.prompts.dict(exclude_none=True)
                    if inference_request.prompts
                    else None
                )

                payload = {
                    "image": http_image,
                    "prompts": prompts_data,
                    "multimask_output": inference_request.multimask_output,
                }

                try:
                    headers = {"Content-Type": "application/json"}
                    if ROBOFLOW_INTERNAL_SERVICE_NAME:
                        headers["X-Roboflow-Internal-Service-Name"] = (
                            ROBOFLOW_INTERNAL_SERVICE_NAME
                        )
                    if ROBOFLOW_INTERNAL_SERVICE_SECRET:
                        headers["X-Roboflow-Internal-Service-Secret"] = (
                            ROBOFLOW_INTERNAL_SERVICE_SECRET
                        )

                    headers = build_roboflow_api_headers(
                        explicit_headers=headers
                    )

                    response = requests.post(
                        f"{endpoint}?api_key={api_key}",
                        json=payload,
                        headers=headers,
                        timeout=60,
                    )
                    response.raise_for_status()
                    resp_json = response.json()

                    return Sam2SegmentationResponse(**resp_json)

                except Exception as e:
                    logger.error(
                        f"SAM3 visual_segment remote request failed: {e}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"SAM3 visual_segment remote request failed: {str(e)}",
                    )

            model_manager.add_model(
                "sam3/sam3_interactive",
                api_key=api_key,
                endpoint_type=ModelEndpointType.CORE_MODEL,
                countinference=countinference,
                service_secret=service_secret,
            )

            model_response = model_manager.infer_from_request_sync(
                "sam3/sam3_interactive", inference_request
            )
            return model_response

    if CORE_MODEL_SAM3_ENABLED and not GCP_SERVERLESS:

        @router.post(
            "/sam3_3d/infer",
            summary="SAM3 3D Object Generation",
            description="Generate 3D meshes and Gaussian splatting from 2D images with mask prompts.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def sam3_3d_infer(
            inference_request: Sam3_3D_Objects_InferenceRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """Generate 3D meshes and Gaussian splatting from 2D images with mask prompts.

            Args:
                inference_request (Sam3_3D_Objects_InferenceRequest): The request containing
                    the image and mask input for 3D generation.
                api_key (Optional[str]): Roboflow API Key for artifact retrieval.

            Returns:
                dict: Response containing base64-encoded 3D outputs:
                    - mesh_glb: Scene mesh in GLB format (base64)
                    - gaussian_ply: Combined Gaussian splatting in PLY format (base64)
                    - objects: List of individual objects with their 3D data
                    - time: Inference time in seconds
            """
            logger.debug("Reached /sam3_3d/infer")
            model_id = inference_request.model_id or "sam3-3d-objects"

            model_manager.add_model(
                model_id,
                api_key=api_key,
                endpoint_type=ModelEndpointType.CORE_MODEL,
                countinference=countinference,
                service_secret=service_secret,
            )

            model_response = model_manager.infer_from_request_sync(
                model_id, inference_request
            )

            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(model_id, actor)

            # Convert bytes to base64 for JSON serialization
            def encode_bytes(data):
                if data is None:
                    return None
                return base64.b64encode(data).decode("utf-8")

            objects_list = []
            for obj in model_response.objects:
                objects_list.append(
                    {
                        "mesh_glb": encode_bytes(obj.mesh_glb),
                        "gaussian_ply": encode_bytes(obj.gaussian_ply),
                        "metadata": {
                            "rotation": obj.metadata.rotation,
                            "translation": obj.metadata.translation,
                            "scale": obj.metadata.scale,
                        },
                    }
                )

            return {
                "mesh_glb": encode_bytes(model_response.mesh_glb),
                "gaussian_ply": encode_bytes(model_response.gaussian_ply),
                "objects": objects_list,
                "time": model_response.time,
            }

    if CORE_MODEL_OWLV2_ENABLED:

        @router.post(
            "/owlv2/infer",
            response_model=ObjectDetectionInferenceResponse,
            summary="Owlv2 image prompting",
            description="Run the google owlv2 model to few-shot object detect",
        )
        @with_route_exceptions
        @usage_collector("request")
        def owlv2_infer(
            inference_request: OwlV2InferenceRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Embeds image data using the Meta AI Segmant Anything Model (SAM).

            Args:
                inference_request (SamEmbeddingRequest): The request containing the image to be embedded.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                M.Sam2EmbeddingResponse or Response: The response affirming the image has been embedded
            """
            logger.debug(f"Reached /owlv2/infer")
            owl2_model_id = load_owlv2_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            model_response = model_manager.infer_from_request_sync(
                owl2_model_id, inference_request
            )
            return model_response

    if CORE_MODEL_GAZE_ENABLED:

        @router.post(
            "/gaze/gaze_detection",
            response_model=List[GazeDetectionInferenceResponse],
            summary="Gaze Detection",
            description="Run the gaze detection model to detect gaze.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def gaze_detection(
            inference_request: GazeDetectionInferenceRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Detect gaze using the gaze detection model.

            Args:
                inference_request (M.GazeDetectionRequest): The request containing the image to be detected.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                M.GazeDetectionResponse: The response containing all the detected faces and the corresponding gazes.
            """
            logger.debug(f"Reached /gaze/gaze_detection")
            gaze_model_id = load_gaze_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                gaze_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(gaze_model_id, actor)
            return response

    if DEPTH_ESTIMATION_ENABLED:

        @router.post(
            "/core/depth-estimation",
            response_model=DepthEstimationResponse,
            summary="Depth Estimation",
            description="Run the depth estimation model to generate a depth map.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def depth_estimation(
            inference_request: DepthEstimationRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Generate a depth map using the depth estimation model.

            Args:
                inference_request (DepthEstimationRequest): The request containing the image to estimate depth for.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                DepthEstimationResponse: The response containing the normalized depth map and optional visualization.
            """
            logger.debug(f"Reached /infer/depth-estimation")
            depth_model_id = inference_request.model_id
            model_manager.add_model(
                depth_model_id,
                inference_request.api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                depth_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(depth_model_id, actor)

            # Extract data from nested response structure
            depth_data = response.response
            depth_response = DepthEstimationResponse(
                normalized_depth=depth_data["normalized_depth"].tolist(),
                image=depth_data["image"].base64_image,
            )
            return depth_response

    if CORE_MODEL_TROCR_ENABLED:

        @router.post(
            "/ocr/trocr",
            response_model=OCRInferenceResponse,
            summary="TrOCR OCR response",
            description="Run the TrOCR model to retrieve text in an image.",
        )
        @with_route_exceptions
        @usage_collector("request")
        def trocr_retrieve_text(
            inference_request: TrOCRInferenceRequest,
            request: Request,
            api_key: Optional[str] = Query(
                None,
                description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
            ),
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """
            Retrieves text from image data using the TrOCR model.

            Args:
                inference_request (TrOCRInferenceRequest): The request containing the image from which to retrieve text.
                api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                request (Request, default Body()): The HTTP request.

            Returns:
                OCRInferenceResponse: The response containing the retrieved text.
            """
            logger.debug(f"Reached /trocr/ocr")
            trocr_model_id = load_trocr_model(
                inference_request,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                trocr_model_id, inference_request
            )
            if LAMBDA:
                actor = request.scope["aws.event"]["requestContext"][
                    "authorizer"
                ]["lambda"]["actor"]
                trackUsage(trocr_model_id, actor)
            return orjson_response_keeping_parent_id(response)

    return router
