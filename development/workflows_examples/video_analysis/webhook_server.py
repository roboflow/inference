import logging
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, Response
from starlette.responses import JSONResponse

from inference.core import logger
from inference.core.utils.image_utils import load_image_bgr

logging.getLogger().setLevel(logging.INFO)

app = FastAPI()


@app.get("/data-sink/json-payload")
async def handle_json_payload_with_get(payload: Request) -> Response:
    data = await payload.json()
    logging.info(f"Received the request with the following payload: {data}")
    logging.info(f"Query params: {list(payload.query_params.items())}")
    return JSONResponse(content={"status": "ok"})


@app.post("/data-sink/json-payload")
async def handle_json_payload_with_post(payload: Request) -> Response:
    data = await payload.json()
    logging.info(f"Received the request with the following payload: {data}")
    logging.info(f"Query params: {list(payload.query_params.items())}")
    return JSONResponse(content={"status": "ok"})


@app.put("/data-sink/json-payload")
async def handle_json_payload_with_put(payload: Request) -> Response:
    data = await payload.json()
    logging.info(f"Received the request with the following payload: {data}")
    logging.info(f"Query params: {list(payload.query_params.items())}")
    return JSONResponse(content={"status": "ok"})


@app.get("/data-sink/multi-part-data")
async def handle_multipart_request_with_get(
    image: Annotated[bytes, File()],
    form_field: Annotated[str, Form()],
) -> Response:
    try:
        decoded_image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        logger.warning("Failed to decode image using np.frombuffer, please update numpy version")
        decoded_image = cv2.imdecode(np.fromstring(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    logging.info(f"Received image of size: {decoded_image.shape}")
    logging.info(f"Form data: {form_field}")
    return JSONResponse(content={"status": "ok"})


@app.post("/data-sink/multi-part-data")
async def handle_multipart_request_with_post(
    image: Annotated[bytes, File()],
    form_field: Annotated[str, Form()],
) -> Response:
    try:
        decoded_image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        logger.warning("Failed to decode image using np.frombuffer, please update numpy version")
        decoded_image = cv2.imdecode(np.fromstring(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    logging.info(f"Received image of size: {decoded_image.shape}")
    logging.info(f"Form data: {form_field}")
    return JSONResponse(content={"status": "ok"})


@app.put("/data-sink/multi-part-data")
async def handle_multipart_request_with_put(
    image: Annotated[bytes, File()],
    form_field: Annotated[str, Form()],
) -> Response:
    try:
        decoded_image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        logger.warning("Failed to decode image using np.frombuffer, please update numpy version")
        decoded_image = cv2.imdecode(np.fromstring(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    logging.info(f"Received image of size: {decoded_image.shape}")
    logging.info(f"Form data: {form_field}")
    return JSONResponse(content={"status": "ok"})


@app.post("/data-sink/active-learning")
async def handle_image_and_predictions(payload: Request) -> Response:
    data = await payload.json()
    image = {"type": "base64", "value": data["image"]}
    decoded_image = load_image_bgr(image)
    logging.info(f"Received image of size: {decoded_image.shape}")
    logging.info(f"Predictions: {len(data['predictions']['predictions'])} bounding boxes")
    return JSONResponse(content={"status": "ok"})

