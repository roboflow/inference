# `inference_sdk` API Reference

## Top-level

Top-level SDK configuration: API URLs, timeouts, environment variable loading, and remote execution settings.

::: inference_sdk.config

## `http`

Core HTTP client for making inference requests. `InferenceHTTPClient` supports object detection, classification, segmentation, keypoint detection, OCR, CLIP embeddings, and workflow execution.

::: inference_sdk.http.client

::: inference_sdk.http.entities

::: inference_sdk.http.errors

## `http/utils`

Internal utilities for request building, image encoding/decoding, response post-processing, retries, and API key handling.

::: inference_sdk.http.utils.aliases

::: inference_sdk.http.utils.encoding

::: inference_sdk.http.utils.executors

::: inference_sdk.http.utils.iterables

::: inference_sdk.http.utils.loaders

::: inference_sdk.http.utils.post_processing

::: inference_sdk.http.utils.pre_processing

::: inference_sdk.http.utils.profilling

::: inference_sdk.http.utils.request_building

::: inference_sdk.http.utils.requests

## `utils`

General-purpose helpers: lifecycle decorators (`@deprecated`, `@experimental`), environment variable parsing, and SDK logging.

::: inference_sdk.utils.decorators

::: inference_sdk.utils.environment

::: inference_sdk.utils.logging

## `webrtc`

WebRTC streaming client for real-time video inference over peer connections. Supports webcam, RTSP, MJPEG, and video file sources with configurable output routing.

::: inference_sdk.webrtc.client

::: inference_sdk.webrtc.config

::: inference_sdk.webrtc.datachannel

::: inference_sdk.webrtc.session

::: inference_sdk.webrtc.sources

