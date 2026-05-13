# `inference` API Reference

## `core/active_learning`

Active learning loop: sampling strategies, data collection middleware, and configuration.

::: inference.core.active_learning.accounting

::: inference.core.active_learning.configuration

## `core/cache`

Caching backends (in-memory, Redis) used for model artefacts and inference results.

::: inference.core.cache.air_gapped

::: inference.core.cache.base

::: inference.core.cache.memory

::: inference.core.cache.model_artifacts

::: inference.core.cache.redis

## `core/devices`

Hardware device detection and selection helpers.

::: inference.core.devices.utils

## `core/entities/requests`

::: inference.core.entities.requests.clip

::: inference.core.entities.requests.doctr

::: inference.core.entities.requests.dynamic_class_base

::: inference.core.entities.requests.easy_ocr

::: inference.core.entities.requests.gaze

::: inference.core.entities.requests.groundingdino

::: inference.core.entities.requests.inference

::: inference.core.entities.requests.moondream2

::: inference.core.entities.requests.owlv2

::: inference.core.entities.requests.perception_encoder

::: inference.core.entities.requests.sam

::: inference.core.entities.requests.sam2

::: inference.core.entities.requests.sam3

::: inference.core.entities.requests.sam3_3d

::: inference.core.entities.requests.server_state

::: inference.core.entities.requests.trocr

::: inference.core.entities.requests.yolo_world

## `core/entities/responses`

::: inference.core.entities.responses.clip

::: inference.core.entities.responses.gaze

::: inference.core.entities.responses.inference

::: inference.core.entities.responses.notebooks

::: inference.core.entities.responses.ocr

::: inference.core.entities.responses.perception_encoder

::: inference.core.entities.responses.sam

::: inference.core.entities.responses.sam2

::: inference.core.entities.responses.sam3_3d

::: inference.core.entities.responses.server_state

## `core`

Core framework internals: environment config, data entities, and shared utilities.

::: inference.core.exceptions

::: inference.core.nms

::: inference.core.roboflow_api

::: inference.core.telemetry

::: inference.core.usage

## `core/interfaces`

High-level inference interfaces: camera, HTTP, and stream processing.

::: inference.core.interfaces.base

## `core/interfaces/camera`

::: inference.core.interfaces.camera.camera

::: inference.core.interfaces.camera.entities

::: inference.core.interfaces.camera.utils

::: inference.core.interfaces.camera.video_source

## `core/interfaces/http/builder`

::: inference.core.interfaces.http.builder.routes

## `core/interfaces/http`

::: inference.core.interfaces.http.error_handlers

::: inference.core.interfaces.http.http_api

## `core/interfaces/http/handlers`

::: inference.core.interfaces.http.handlers.workflows

## `core/interfaces/http/middlewares`

::: inference.core.interfaces.http.middlewares.cors

## `core/interfaces/stream`

::: inference.core.interfaces.stream.sinks

::: inference.core.interfaces.stream.stream

::: inference.core.interfaces.stream.watchdog

## `core/interfaces/udp`

::: inference.core.interfaces.udp.udp_stream

## `core/interfaces/webrtc_worker`

::: inference.core.interfaces.webrtc_worker.entities

::: inference.core.interfaces.webrtc_worker.serializers

::: inference.core.interfaces.webrtc_worker.utils

::: inference.core.interfaces.webrtc_worker.webrtc

## `core/interfaces/webrtc_worker/sources`

::: inference.core.interfaces.webrtc_worker.sources.file

## `core/logging`

::: inference.core.logging.memory_handler

## `core/managers`

Model lifecycle managers: loading, unloading, registry, and resolution.

::: inference.core.managers.base

::: inference.core.managers.metrics

::: inference.core.managers.model_load_collector

::: inference.core.managers.pingback

::: inference.core.managers.prometheus

## `core/managers/decorators`

::: inference.core.managers.decorators.base

::: inference.core.managers.decorators.locked_load

::: inference.core.managers.decorators.logger

## `core/models`

Base model classes and common prediction logic shared across model types.

::: inference.core.models.base

::: inference.core.models.classification_base

::: inference.core.models.inference_models_adapters

::: inference.core.models.instance_segmentation_base

::: inference.core.models.keypoints_detection_base

::: inference.core.models.object_detection_base

::: inference.core.models.roboflow

## `core/models/utils`

::: inference.core.models.utils.keypoints

## `core/registries`

Model and block registries for dynamic lookup and plugin discovery.

::: inference.core.registries.base

::: inference.core.registries.roboflow

## `core/utils`

General-purpose utilities: image encoding, file I/O, hashing, URL handling, and more.

::: inference.core.utils.container

::: inference.core.utils.cuda_health

::: inference.core.utils.environment

::: inference.core.utils.file_system

::: inference.core.utils.image_utils

::: inference.core.utils.onnx

::: inference.core.utils.postprocess

::: inference.core.utils.preprocess

::: inference.core.utils.torchscript_guard

## `core/workflows/core_steps/analytics/detection_event_log`

::: inference.core.workflows.core_steps.analytics.detection_event_log.v1

## `core/workflows/core_steps/classical_cv/camera_focus`

::: inference.core.workflows.core_steps.classical_cv.camera_focus.v1

::: inference.core.workflows.core_steps.classical_cv.camera_focus.v2

## `core/workflows/core_steps/classical_cv/contours`

::: inference.core.workflows.core_steps.classical_cv.contours.v1

## `core/workflows/core_steps/classical_cv/contrast_enhancement`

::: inference.core.workflows.core_steps.classical_cv.contrast_enhancement.v1

## `core/workflows/core_steps/classical_cv/distance_measurement`

::: inference.core.workflows.core_steps.classical_cv.distance_measurement.v1

## `core/workflows/core_steps/classical_cv/image_blur`

::: inference.core.workflows.core_steps.classical_cv.image_blur.v1

## `core/workflows/core_steps/classical_cv/mask_area_measurement`

::: inference.core.workflows.core_steps.classical_cv.mask_area_measurement.v1

## `core/workflows/core_steps/classical_cv/mask_edge_snap`

::: inference.core.workflows.core_steps.classical_cv.mask_edge_snap.v1

## `core/workflows/core_steps/classical_cv/morphological_transformation`

::: inference.core.workflows.core_steps.classical_cv.morphological_transformation.v2

## `core/workflows/core_steps/classical_cv/motion_detection`

::: inference.core.workflows.core_steps.classical_cv.motion_detection.v1

## `core/workflows/core_steps/classical_cv/pixel_color_count`

::: inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1

## `core/workflows/core_steps/classical_cv/sift`

::: inference.core.workflows.core_steps.classical_cv.sift.v1

## `core/workflows/core_steps/classical_cv/sift_comparison`

::: inference.core.workflows.core_steps.classical_cv.sift_comparison.v2

## `core/workflows/core_steps/classical_cv/size_measurement`

::: inference.core.workflows.core_steps.classical_cv.size_measurement.v1

## `core/workflows/core_steps/classical_cv/threshold`

::: inference.core.workflows.core_steps.classical_cv.threshold.v1

## `core/workflows/core_steps/common/query_language/introspection`

::: inference.core.workflows.core_steps.common.query_language.introspection.core

## `core/workflows/core_steps/common`

::: inference.core.workflows.core_steps.common.utils

## `core/workflows/core_steps/flow_control/inner_workflow`

::: inference.core.workflows.core_steps.flow_control.inner_workflow.v1

## `core/workflows/core_steps/fusion/detections_list_rollup`

::: inference.core.workflows.core_steps.fusion.detections_list_rollup.v1

## `core/workflows/core_steps/fusion/detections_stitch`

::: inference.core.workflows.core_steps.fusion.detections_stitch.v1

## `core/workflows/core_steps`

::: inference.core.workflows.core_steps.loader

## `core/workflows/core_steps/models/foundation`

::: inference.core.workflows.core_steps.models.foundation._streaming_video_common

## `core/workflows/core_steps/models/foundation/anthropic_claude`

::: inference.core.workflows.core_steps.models.foundation.anthropic_claude.v3

## `core/workflows/core_steps/models/foundation/gaze`

::: inference.core.workflows.core_steps.models.foundation.gaze.v1

## `core/workflows/core_steps/models/foundation/google_gemini`

::: inference.core.workflows.core_steps.models.foundation.google_gemini.v3

## `core/workflows/core_steps/models/foundation/openai`

::: inference.core.workflows.core_steps.models.foundation.openai.v3

::: inference.core.workflows.core_steps.models.foundation.openai.v4

## `core/workflows/core_steps/models/foundation/segment_anything2_video`

::: inference.core.workflows.core_steps.models.foundation.segment_anything2_video.v1

## `core/workflows/core_steps/models/foundation/segment_anything3_3d`

::: inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1

## `core/workflows/core_steps/models/foundation/stability_ai/inpainting`

::: inference.core.workflows.core_steps.models.foundation.stability_ai.inpainting.v1

## `core/workflows/core_steps/models/foundation/stability_ai/outpainting`

::: inference.core.workflows.core_steps.models.foundation.stability_ai.outpainting.v1

## `core/workflows/core_steps/sinks/email_notification`

::: inference.core.workflows.core_steps.sinks.email_notification.v2

## `core/workflows/core_steps/sinks/roboflow/dataset_upload`

::: inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1

## `core/workflows/core_steps/sinks/roboflow/vision_events`

::: inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1

## `core/workflows/core_steps/sinks/twilio/sms`

::: inference.core.workflows.core_steps.sinks.twilio.sms.v2

## `core/workflows/core_steps/trackers`

::: inference.core.workflows.core_steps.trackers._base

## `core/workflows/core_steps/transformations/detections_merge`

::: inference.core.workflows.core_steps.transformations.detections_merge.v1

## `core/workflows/core_steps/transformations/image_slicer`

::: inference.core.workflows.core_steps.transformations.image_slicer.v1

::: inference.core.workflows.core_steps.transformations.image_slicer.v2

## `core/workflows/core_steps/transformations/qr_code_generator`

::: inference.core.workflows.core_steps.transformations.qr_code_generator.v1

## `core/workflows/core_steps/transformations/stitch_ocr_detections`

::: inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v1

::: inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v2

## `core/workflows/core_steps/visualizations/classification_label`

::: inference.core.workflows.core_steps.visualizations.classification_label.v1

## `core/workflows/core_steps/visualizations/common/annotators`

::: inference.core.workflows.core_steps.visualizations.common.annotators.background_color

::: inference.core.workflows.core_steps.visualizations.common.annotators.halo

::: inference.core.workflows.core_steps.visualizations.common.annotators.model_comparison

::: inference.core.workflows.core_steps.visualizations.common.annotators.polygon

## `core/workflows/core_steps/visualizations/text_display`

::: inference.core.workflows.core_steps.visualizations.text_display.utils

::: inference.core.workflows.core_steps.visualizations.text_display.v1

## `core/workflows`

Workflow execution engine entry points and helpers.

::: inference.core.workflows.errors

## `core/workflows/execution_engine/introspection`

::: inference.core.workflows.execution_engine.introspection.blocks_loader

::: inference.core.workflows.execution_engine.introspection.schema_parser

## `core/workflows/execution_engine/v1/compiler`

::: inference.core.workflows.execution_engine.v1.compiler.cache

::: inference.core.workflows.execution_engine.v1.compiler.graph_constructor

::: inference.core.workflows.execution_engine.v1.compiler.graph_traversal

::: inference.core.workflows.execution_engine.v1.compiler.syntactic_parser

## `core/workflows/execution_engine/v1/dynamic_blocks`

::: inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler

::: inference.core.workflows.execution_engine.v1.dynamic_blocks.error_utils

::: inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor

## `core/workflows/execution_engine/v1/executor/execution_data_manager`

::: inference.core.workflows.execution_engine.v1.executor.execution_data_manager.step_input_assembler

## `core/workflows/execution_engine/v1/inner_workflow`

::: inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge

::: inference.core.workflows.execution_engine.v1.inner_workflow.composition

::: inference.core.workflows.execution_engine.v1.inner_workflow.errors

::: inference.core.workflows.execution_engine.v1.inner_workflow.inline

::: inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution

## `core/workflows/prototypes`

::: inference.core.workflows.prototypes.block

## `enterprise/parallel`

Parallel HTTP inference via Celery workers for high-throughput deployments.

::: inference.enterprise.parallel.dispatch_manager

::: inference.enterprise.parallel.infer

::: inference.enterprise.parallel.utils

## `enterprise/workflows/enterprise_blocks/sinks/PLC_modbus`

::: inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1

## `enterprise/workflows/enterprise_blocks/sinks/PLCethernetIP`

::: inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1

## `enterprise/workflows/enterprise_blocks/sinks/event_writer`

::: inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1

## `enterprise/workflows/enterprise_blocks/sinks/microsoft_sql_server`

::: inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v1

## `enterprise/workflows/enterprise_blocks/sinks/opc_writer`

::: inference.enterprise.workflows.enterprise_blocks.sinks.opc_writer.v1

## `models/clip`

::: inference.models.clip.clip_inference_models

::: inference.models.clip.clip_model

## `models/deep_lab_v3_plus`

::: inference.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation

## `models/depth_anything_v3/architecture`

::: inference.models.depth_anything_v3.architecture.da3

::: inference.models.depth_anything_v3.architecture.dpt

::: inference.models.depth_anything_v3.architecture.dualdpt

::: inference.models.depth_anything_v3.architecture.head_utils

## `models/depth_anything_v3/architecture/layers`

::: inference.models.depth_anything_v3.architecture.layers.drop_path

::: inference.models.depth_anything_v3.architecture.layers.patch_embed

::: inference.models.depth_anything_v3.architecture.layers.rope

## `models/depth_anything_v3`

::: inference.models.depth_anything_v3.depth_anything_v3

## `models/dinov3`

::: inference.models.dinov3.dinov3_classification

## `models/doctr`

::: inference.models.doctr.doctr_model

## `models/easy_ocr`

::: inference.models.easy_ocr.easy_ocr

::: inference.models.easy_ocr.easy_ocr_inference_models

## `models/florence2`

::: inference.models.florence2.utils

## `models/gaze`

::: inference.models.gaze.gaze

::: inference.models.gaze.gaze_inference_models

::: inference.models.gaze.l2cs

## `models/grounding_dino`

::: inference.models.grounding_dino.grounding_dino

::: inference.models.grounding_dino.grounding_dino_inference_models

## `models/owlv2`

::: inference.models.owlv2.owlv2

## `models/paligemma`

::: inference.models.paligemma.paligemma

## `models/perception_encoder`

::: inference.models.perception_encoder.perception_encoder

::: inference.models.perception_encoder.perception_encoder_inference_models

## `models/perception_encoder/vision_encoder`

::: inference.models.perception_encoder.vision_encoder.config

::: inference.models.perception_encoder.vision_encoder.pe

::: inference.models.perception_encoder.vision_encoder.rope

::: inference.models.perception_encoder.vision_encoder.tokenizer

## `models/qwen25vl`

::: inference.models.qwen25vl.qwen25vl

## `models/resnet`

::: inference.models.resnet.resnet_classification

## `models/rfdetr`

::: inference.models.rfdetr.rfdetr

## `models/sam`

::: inference.models.sam.segment_anything

## `models/sam2`

::: inference.models.sam2.segment_anything2

::: inference.models.sam2.segment_anything2_inference_models

## `models/sam3`

::: inference.models.sam3.segment_anything3

::: inference.models.sam3.visual_segmentation

## `models/sam3_3d`

::: inference.models.sam3_3d.segment_anything_3d

## `models/vit`

::: inference.models.vit.vit_classification

## `models/yolact`

::: inference.models.yolact.yolact_instance_segmentation

## `models/yolo26`

::: inference.models.yolo26.yolo26_instance_segmentation

::: inference.models.yolo26.yolo26_keypoints_detection

## `models/yolo_world`

::: inference.models.yolo_world.yolo_world

## `models/yolov10`

::: inference.models.yolov10.yolov10_object_detection

## `models/yolov5`

::: inference.models.yolov5.yolov5_instance_segmentation

::: inference.models.yolov5.yolov5_object_detection

## `models/yolov7`

::: inference.models.yolov7.yolov7_instance_segmentation

## `models/yolov8`

::: inference.models.yolov8.yolov8_instance_segmentation

::: inference.models.yolov8.yolov8_keypoints_detection

::: inference.models.yolov8.yolov8_object_detection

## `models/yolov9`

::: inference.models.yolov9.yolov9_object_detection

## `usage_tracking`

Anonymous usage and telemetry reporting.

::: inference.usage_tracking.redis_queue

