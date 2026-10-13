"""Exercise selective tail discovery and representative compiled offline journeys."""

import builtins
import importlib
import json
import threading

import pytest
import torch
from roboflow_workflows.v2 import blocks
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine import Catalogue, compile_workflow
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import UnknownBlockError

from inference_models.models.base.object_detection import Detections
from tests.v2.integration.blocks import test_industrial_sql_compiled as sql
from tests.v2.integration.blocks.analytics import test_motion_workflows as motion
from tests.v2.integration.blocks.language import (
    test_compatible_vendor_workflows as compatible,
)
from tests.v2.integration.blocks.language import (
    test_openrouter_vendor_workflows as router,
)
from tests.v2.integration.blocks.models import test_vlm_workflows as q3
from tests.v2.integration.blocks.visualization.annotations import (
    test_counter_drawing_composition as counters,
)
from tests.v2.integration.blocks.visualization.annotations import (
    test_decorative_workflows as decorative,
)

RECORDS = [
    {
        "owner": "G03",
        "class": "Qwen35VLBlock",
        "type": "v2/qwen3_5vl@v2",
        "module": "roboflow_workflows.v2.blocks.models.vlm",
        "package": "models",
    },
    {
        "owner": "G04",
        "class": "OpenRouterBlock",
        "type": "v2/openrouter@v3",
        "module": "roboflow_workflows.v2.blocks.language.openrouter",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "KimiOpenRouterBlock",
        "type": "v2/kimi_openrouter@v3",
        "module": "roboflow_workflows.v2.blocks.language.openrouter_vendors",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "MetaVlmBlock",
        "type": "v2/meta_vlm@v3",
        "module": "roboflow_workflows.v2.blocks.language.openrouter_vendors",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "ZaiVlmBlock",
        "type": "v2/zai_vlm@v2",
        "module": "roboflow_workflows.v2.blocks.language.openrouter_vendors",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "GoogleGemmaBlock",
        "type": "v2/google_gemma@v4",
        "module": "roboflow_workflows.v2.blocks.language.compatible_vendors",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "SpaceXAIBlock",
        "type": "v2/spacexai@v3",
        "module": "roboflow_workflows.v2.blocks.language.compatible_vendors",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "OpenAIBlock",
        "type": "v2/open_ai@v7",
        "module": "roboflow_workflows.v2.blocks.language.openai",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "OpenAICompatibleBlock",
        "type": "v2/openai_compatible@v1",
        "module": "roboflow_workflows.v2.blocks.language.openai",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "AnthropicClaudeBlock",
        "type": "v2/anthropic_claude@v5",
        "module": "roboflow_workflows.v2.blocks.language.anthropic",
        "package": "language",
    },
    {
        "owner": "G04",
        "class": "GoogleGeminiBlock",
        "type": "v2/google_gemini@v6",
        "module": "roboflow_workflows.v2.blocks.language.gemini",
        "package": "language",
    },
    {
        "owner": "G05",
        "class": "LineCounterBlock",
        "type": "v2/line_counter@v2",
        "module": "roboflow_workflows.v2.blocks.analytics.line_counter",
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "TimeInZoneBlock",
        "type": "v2/time_in_zone@v3",
        "module": "roboflow_workflows.v2.blocks.analytics.time_in_zone",
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "PathDeviationBlock",
        "type": "v2/path_deviation_analytics@v2",
        "module": "roboflow_workflows.v2.blocks.analytics.path_deviation",
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "DetectionEventLogBlock",
        "type": "v2/detection_event_log@v1",
        "module": "roboflow_workflows.v2.blocks.analytics.event_log",
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "OverlapFilterBlock",
        "type": "v2/overlap_filter@v1",
        "module": "roboflow_workflows.v2.blocks.analytics.overlap_filter",
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "BackgroundSubtractionBlock",
        "type": "v2/background_subtraction@v1",
        "module": "roboflow_workflows.v2.blocks.analytics.background_subtraction",
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "MotionDetectionBlock",
        "type": "v2/motion_detection@v1",
        "module": "roboflow_workflows.v2.blocks.analytics.motion_detection",
        "package": "analytics",
    },
    {
        "owner": "G12",
        "class": "PlcReaderBlock",
        "type": "v2/plc_reader@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.plc",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "PlcWriterBlock",
        "type": "v2/plc_writer@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.plc",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "PostgreSQLSinkBlock",
        "type": "v2/postgresql_sink@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.sql",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "SQLServerSinkBlock",
        "type": "v2/microsoft_sql_server_sink@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.sql",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "MQTTReaderBlock",
        "type": "v2/mqtt_reader@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.mqtt",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "MQTTWriterBlock",
        "type": "v2/mqtt_writer_sink@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.mqtt",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "KafkaConsumerBlock",
        "type": "v2/kafka_consumer@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.kafka",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "KafkaProducerSinkBlock",
        "type": "v2/kafka_producer_sink@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.kafka",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "OPCWriterSinkBlock",
        "type": "v2/opc_writer_sink@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.opc",
        "package": "industrial",
    },
    {
        "owner": "G12",
        "class": "OnvifTrackingBlock",
        "type": "v2/onvif_sink@v1",
        "module": "roboflow_workflows.v2.blocks.industrial.onvif",
        "package": "industrial",
    },
    {
        "owner": "G08",
        "class": "BackgroundColorVisualizationBlock",
        "type": "v2/background_color_visualization@v1",
        "module": "roboflow_workflows.v2.blocks.visualization.annotations.background_color",
        "package": "visualization.annotations",
    },
    {
        "owner": "G08",
        "class": "ColorVisualizationBlock",
        "type": "v2/color_visualization@v1",
        "module": "roboflow_workflows.v2.blocks.visualization.annotations.color",
        "package": "visualization.annotations",
    },
    {
        "owner": "G08",
        "class": "GridVisualizationBlock",
        "type": "v2/grid_visualization@v1",
        "module": "roboflow_workflows.v2.blocks.visualization.annotations.grid",
        "package": "visualization.annotations",
    },
]
BASELINE = {
    "block_types": [
        "v2/crop",
        "v2/static_crop",
        "v2/resize",
        "v2/invert",
        "v2/mosaic",
        "v2/has_brightness",
        "v2/absolute_static_crop@v1",
        "v2/relative_static_crop@v1",
        "v2/dynamic_crop@v1",
        "v2/image_preprocessing@v1",
        "v2/image_slicer@v1",
        "v2/image_copy@v1",
        "v2/convert_grayscale@v1",
        "v2/image_blur@v1",
        "v2/threshold@v1",
        "v2/mask_edge_snap@v1",
        "v2/morphological_transformation@v2",
        "v2/contrast_enhancement@v1",
        "v2/contrast_equalization@v1",
        "v2/delta_filter@v1",
        "v2/rate_limiter@v1",
        "v2/switch_case@v1",
        "v2/continue_if",
        "v2/continue_if@v1",
        "v2/best_frame",
        "v2/top_k_brightest",
        "v2/frame_delay@v1",
        "v2/image_stack@v1",
        "v2/cache_get@v1",
        "v2/cache_set@v1",
        "v2/state_get",
        "v2/state_set",
        "v2/state_increment",
        "v2/state_compare_and_set",
        "v2/state_machine_set",
        "v2/cosmos3_edge@v1",
        "v2/florence_2@v2",
        "v2/google_vision_ocr@v1",
        "v2/moondream2@v1",
        "v2/qwen_vlm@v4",
        "v2/object_detection@v1",
        "v2/classification@v1",
        "v2/multi_label_classification@v1",
        "v2/keypoint_detection@v1",
        "v2/instance_segmentation@v1",
        "v2/semantic_segmentation@v1",
        "v2/action_recognition@v1",
        "v2/segment_anything@v1",
        "v2/segment_anything_2_video@v1",
        "v2/sam3@v3",
        "v2/sam3_interactive@v1",
        "v2/sam3_video@v1",
        "v2/trackers_sort@v1",
        "v2/trackers_bytetrack@v1",
        "v2/trackers_ocsort@v1",
        "v2/trackers_botsort@v1",
        "v2/track_class_lock@v1",
        "v2/stabilize_detections@v1",
        "v2/velocity@v1",
        "v2/detections_stitch@v1",
        "v2/detection_offset@v1",
        "v2/bounding_rect@v1",
        "v2/detections_classes_replacement@v1",
        "v2/dimension_collapse@v1",
        "v2/detections_filter@v1",
        "v2/detections_transformation@v1",
        "v2/per_class_confidence_filter@v1",
        "v2/detections_combine@v1",
        "v2/detections_merge@v1",
        "v2/detections_consensus@v1",
        "v2/detections_difference@v1",
        "v2/overlap_analysis@v1",
        "v2/detections_list_rollup@v1",
        "v2/identify_changes@v1",
        "v2/identify_outliers@v1",
        "v2/property_definition@v1",
        "v2/expression@v1",
        "v2/string_template@v1",
        "v2/csv_formatter@v1",
        "v2/first_non_empty_or_default@v1",
        "v2/current_time@v1",
        "v2/json_parser@v1",
        "v2/detection_drawing@v1",
        "v2/box_visualization@v1",
        "v2/label_visualization@v1",
        "v2/detection_visualization@v1",
        "v2/label_visualization_in_place@v1",
        "v2/bounding_box_visualization@v1",
        "v2/label_visualization@v2",
        "v2/polygon_visualization@v2",
        "v2/mask_visualization@v1",
        "v2/keypoint_visualization@v1",
        "v2/classification_label_visualization@v1",
        "v2/crop_visualization@v1",
        "v2/blur_visualization@v1",
        "v2/pixelate_visualization@v1",
        "v2/semantic_segmentation_visualization@v1",
        "v2/circle_visualization@v1",
        "v2/corner_visualization@v1",
        "v2/dot_visualization@v1",
        "v2/ellipse_visualization@v1",
        "v2/halo_visualization@v2",
        "v2/triangle_visualization@v1",
        "v2/heatmap_visualization@v1",
        "v2/trace_visualization@v1",
        "v2/polygon_zone_visualization@v1",
        "v2/line_counter_visualization@v1",
        "v2/reference_path_visualization@v1",
        "v2/asset_library_attributes@v1",
        "v2/roboflow_dataset_upload@v2",
        "v2/model_monitoring_inference_aggregator@v1",
        "v2/roboflow_custom_metadata@v1",
        "v2/visual_search@v1",
        "v2/visual_search_classifier@v1",
        "v2/roboflow_vision_events@v1",
        "v2/vision_event_bundle@v1",
        "v2/distance_measurement@v1",
        "v2/size_measurement@v1",
        "v2/mask_area_measurement@v1",
        "v2/pixel_color_count@v1",
    ],
    "operator_types": ["v2/align@v1", "v2/window@v1"],
    "kinds": [
        "*",
        "bar_code_detection",
        "boolean",
        "classification_prediction",
        "crop_summary",
        "detection",
        "detection_drawing",
        "detections_overlaps",
        "dictionary",
        "embedding",
        "float",
        "image",
        "instance_segmentation_prediction",
        "instance_segmentation_view",
        "integer",
        "keypoint_detection_prediction",
        "language_model_output",
        "list_of_values",
        "native_semantic_segmentation",
        "numpy_array",
        "object_detection_prediction",
        "prediction_rows",
        "qr_code_detection",
        "rle_instance_segmentation_prediction",
        "secret",
        "semantic_segmentation_prediction",
        "string",
        "tensor",
        "timestamp",
    ],
}


@pytest.mark.parametrize("route", ["create", "plugin", "factory"])
def test_exact_31_additions_preserve_current_registry_exports_and_schemas(route):
    catalogue = (
        blocks.create_catalogue()
        if route == "create"
        else Catalogue.from_modules(
            ["roboflow_workflows.v2.blocks"],
            **({"attribute": "create_catalogue"} if route == "factory" else {})
        )
    )
    records = RECORDS
    baseline = BASELINE
    added = {row["type"] for row in records}
    successor = set(
        (
            "v2/icon_visualization@v1",
            "v2/rich_label_visualization@v1",
            "v2/model_comparison_visualization@v1",
            "v2/clip_comparison@v1",
            "v2/clip@v1",
            "v2/perception_encoder@v1",
            "v2/depth_estimation@v1",
            "v2/cosine_similarity@v1",
            "v2/glm_ocr@v1",
            "v2/ocr_model@v1",
            "v2/easy_ocr@v1",
            "v2/pp_ocr@v1",
            "v2/smolvlm2@v1",
            "v2/camera_focus@v2",
            "v2/stitch_images@v1",
            "v2/auto_rotate_on_edges@v1",
            "v2/stitch_ocr_detections@v2",
            "v2/sift@v1",
            "v2/perspective_correction@v1",
            "v2/template_matching@v1",
            "v2/geotag_detection@v1",
            "v2/detections_nearest_neighbor@v1",
            "v2/sift_comparison@v2",
            "v2/camera_calibration@v1",
            "v2/dominant_color@v1",
            "v2/qr_code_generator@v1",
            "v2/barcode_detector@v1",
            "v2/qr_code_detector@v1",
            "v2/contours_detection@v1",
            "v2/dynamic_zone@v1",
            "v2/stability_ai_image_gen@v1",
            "v2/stability_ai_inpainting@v1",
            "v2/stability_ai_outpainting@v1",
            "v2/data_aggregator@v1",
            "v2/email_notification@v2",
            "v2/environment_secrets_store@v1",
            "v2/local_file_sink@v1",
            "v2/s3_sink@v1",
            "v2/slack_notification@v1",
            "v2/text_display@v1",
            "v2/twilio_sms_notification@v2",
            "v2/webhook_sink@v1",
            "v2/event_writer_sink@v1",
        )
    )
    assert len(added) == 31
    assert (
        set(catalogue.block_types) - set(baseline["block_types"]) == added | successor
    )
    assert tuple(
        identity
        for identity in catalogue.block_types
        if identity not in added | successor
    ) == tuple(baseline["block_types"])
    assert set(baseline["block_types"]) - set(catalogue.block_types) == set()
    description = catalogue.describe()
    json.dumps(description)
    schemas = {row["type"]: row["params_schema"] for row in description["blocks"]}
    for row in records:
        block = getattr(importlib.import_module(row["module"]), row["class"])
        package = importlib.import_module(
            "roboflow_workflows.v2.blocks." + row["package"]
        )
        assert catalogue.block_types.count(row["type"]) == 1
        assert getattr(blocks, row["class"]) is getattr(package, row["class"]) is block
        assert (
            blocks.__all__.count(row["class"])
            == package.__all__.count(row["class"])
            == 1
        )
        assert package.BLOCKS.count(block) == 1
        for identity in spec_of(block).identities:
            assert catalogue.entry(identity).spec is spec_of(block)
        assert schemas[row["type"]] == spec_of(block).params_model.model_json_schema()
    assert catalogue.kind("secret") is SECRET_KIND
    assert catalogue.kind("language_model_output") is LANGUAGE_MODEL_OUTPUT_KIND
    assert (
        blocks.__all__.count("SECRET_KIND")
        == blocks.__all__.count("LANGUAGE_MODEL_OUTPUT_KIND")
        == 1
    )
    assert "v2/window@v1" in catalogue.operator_types
    assert not hasattr(blocks, "BufferBlock")
    assert {name for values in catalogue.providers.values() for name in values} == {
        "object_cache"
    }


def test_retired_identities_stay_rejected_and_active_q3_is_selective():
    catalogue = blocks.create_catalogue()
    retired = [
        "v2/qwen25vl@v1",
        "v2/qwen3vl@v1",
        "v2/byte_tracker@v1",
        "v2/llama_vision@v2",
        "v2/qwen3_5_openrouter@v1",
        "v2/qwen3_6_openrouter@v1",
        "v2/vlm_as_classifier@v2",
        "v2/vlm_as_detector@v2",
        "v2/lmm@v1",
    ]
    for identity in retired:
        assert catalogue.find(identity) is None
        with pytest.raises(UnknownBlockError):
            compile_workflow(
                {
                    "version": "2.0",
                    "inputs": [],
                    "steps": [{"type": identity, "name": "retired"}],
                    "outputs": [],
                },
                catalogue=catalogue,
            )
    assert catalogue.find("v2/smolvlm2@v1") is not None
    assert catalogue.block_types.count("v2/qwen3_5vl@v2") == 1


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_active_q3_fake_prompts_keep_rows_through_default_catalogue(mode, monkeypatch):
    monkeypatch.setattr(q3, "_catalogue", blocks.create_catalogue)
    q3.test_per_image_models_load_once_and_answers_keep_row_order(
        "v2/qwen3_5vl@v2", mode
    )


@pytest.mark.parametrize("provider", ["router", "compatible"])
def test_active_provider_secret_fakewire_produces_native_geometry(
    provider, monkeypatch
):
    fixture = router if provider == "router" else compatible
    monkeypatch.setattr(fixture, "CATALOGUE", blocks.create_catalogue())
    if provider == "router":
        fixture.test_meta_detections_from_a_secret_keyed_request_feed_a_geometry_step()
    else:
        fixture.test_gemma_detections_from_a_secret_key_feed_a_geometry_step()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_current_counter_and_drawing_need_no_injected_leaf(mode, monkeypatch):
    original = counters.compile_workflow
    expected = blocks.create_catalogue().block_types

    def compile_default(definition, *, catalogue, **kwargs):
        assert catalogue.block_types == expected
        return original(definition, catalogue=catalogue, **kwargs)

    monkeypatch.setattr(counters, "compile_workflow", compile_default)
    counters.test_real_counter_outputs_draw_correct_labels_and_shared_zone_across_sources(
        mode
    )


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_default_motion_models_keep_two_sources_separate(mode, monkeypatch):
    monkeypatch.setattr(motion, "CATALOGUE", blocks.create_catalogue())
    motion.test_interleaved_cameras_keep_separate_models(mode)


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_three_default_decorative_drawings_form_exact_composite_tiles(
    mode, monkeypatch
):
    monkeypatch.setattr(decorative, "annotation_catalogue", blocks.create_catalogue)
    monkeypatch.setattr(decorative, "DECORATIVE_BLOCKS", ())
    decorative.test_grid_shows_an_image_and_its_drawings_on_one_canvas_per_image(mode)


@pytest.mark.parametrize("case", ["parameterized-write", "unavailable-secret"])
def test_default_industrial_sql_borrows_offline_client_and_preserves_secret_contract(
    case, monkeypatch
):
    monkeypatch.setattr(sql, "CATALOGUE", blocks.create_catalogue())
    if case == "parameterized-write":
        sql.test_a_compiled_postgresql_step_writes_one_parameterized_batch()
    else:
        sql.test_an_unavailable_secret_fails_the_step_and_writes_nothing()


def test_industrial_selection_session_and_disabled_paths_acquire_no_optional_resources(
    monkeypatch,
):
    acquired = []
    original_import = builtins.__import__
    optional = {
        "paho",
        "asyncua",
        "zeep",
        "confluent_kafka",
        "psycopg2",
        "pyodbc",
        "pycomm3",
        "pymodbus",
        "kafka",
    }

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in optional:
            acquired.append(name)
            raise AssertionError("Optional SDK acquired: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: pytest.fail("Thread acquired")
    )
    monkeypatch.setattr(
        Secret, "reveal", lambda self: pytest.fail("Secret revealed on disabled path")
    )
    catalogue = blocks.create_catalogue()
    samples = {
        "ip_address": "127.0.0.1",
        "tag": "x",
        "value": 1,
        "host": "offline",
        "server": "offline",
        "database": "d",
        "username": "u",
        "user_name": "u",
        "table_name": "records",
        "data": {"x": 1},
        "message": "payload",
        "bootstrap_servers": "offline:9092",
        "topic": "t",
        "url": "opc.tcp://offline",
        "namespace": "2",
        "object_name": "x",
        "variable_name": "x",
        "predictions": "$inputs.predictions",
        "camera_ip": "offline",
        "camera_port": 80,
        "camera_username": "u",
        "camera_password": "$inputs.secret",
        "password": "$inputs.secret",
    }
    records = RECORDS
    for row in (row for row in records if row["owner"] == "G12"):
        block = getattr(blocks, row["class"])
        spec = spec_of(block)
        step = {"type": block.type, "name": "industrial"}
        for name, field in spec.params_model.model_fields.items():
            if field.is_required():
                step[name] = samples[name]
        disabled = False
        for flag in ("disable_sink", "disable_reader"):
            if flag in spec.params_model.model_fields:
                step[flag] = True
                disabled = True
        if row["class"] in ("PostgreSQLSinkBlock", "SQLServerSinkBlock"):
            disabled = True
        definition = {
            "version": "2.0",
            "inputs": [
                {"type": "WorkflowParameter", "name": "secret", "kind": ["secret"]},
                {
                    "type": "WorkflowBatchInput",
                    "name": "predictions",
                    "kind": ["object_detection_prediction"],
                },
            ],
            "steps": [step],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": name,
                    "selector": "$steps.industrial." + name,
                }
                for name in spec.outputs
            ],
        }
        plan = compile_workflow(definition, catalogue=catalogue)
        with plan.create_session({"disable_sinks": True}) as session:
            if disabled:
                predictions = Detections(
                    xyxy=torch.zeros((0, 4)),
                    class_id=torch.zeros(0, dtype=torch.int64),
                    confidence=torch.zeros(0),
                )
                result = session.run(
                    {
                        "secret": Secret.unavailable("offline"),
                        "predictions": [predictions],
                    }
                )
                assert (
                    "offline" not in json.dumps(result.rows(serialize=True))
                    or row["class"] == "OnvifTrackingBlock"
                )
    assert acquired == []
