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
from tests.v2.integration.blocks.analytics import test_motion_workflows as motion
from tests.v2.integration.blocks.models.vision_language_models import (
    test_compatible_vendor_workflows as compatible,
)
from tests.v2.integration.blocks.models.vision_language_models import (
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
        "type": "roboflow_core/qwen3_5vl@v2",
        "module": 'roboflow_workflows.v2.blocks.models.qwen3_5vl.v2',
        "package": "models",
    },
    {
        "owner": "G04",
        "class": "OpenRouterBlock",
        "type": "roboflow_core/openrouter@v3",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.openrouter.v3',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "KimiOpenRouterBlock",
        "type": "roboflow_core/kimi_openrouter@v3",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.kimi_openrouter.v3',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "MetaVlmBlock",
        "type": "roboflow_core/meta_vlm@v3",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.meta_vlm.v3',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "ZaiVlmBlock",
        "type": "roboflow_core/zai_vlm@v2",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.zai_vlm.v2',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "GoogleGemmaBlock",
        "type": "roboflow_core/google_gemma@v4",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.google_gemma.v4',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "SpaceXAIBlock",
        "type": "roboflow_core/spacexai@v3",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.spacexai.v3',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "OpenAIBlock",
        "type": "roboflow_core/open_ai@v7",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.open_ai.v7',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "OpenAICompatibleBlock",
        "type": "roboflow_core/openai_compatible@v1",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.openai_compatible.v1',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "AnthropicClaudeBlock",
        "type": "roboflow_core/anthropic_claude@v5",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.anthropic_claude.v5',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G04",
        "class": "GoogleGeminiBlock",
        "type": "roboflow_core/google_gemini@v6",
        "module": 'roboflow_workflows.v2.blocks.models.vision_language_models.google_gemini.v6',
        "package": "models.vision_language_models",
    },
    {
        "owner": "G05",
        "class": "LineCounterBlock",
        "type": "roboflow_core/line_counter@v2",
        "module": 'roboflow_workflows.v2.blocks.analytics.line_counter.v2',
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "TimeInZoneBlock",
        "type": "roboflow_core/time_in_zone@v3",
        "module": 'roboflow_workflows.v2.blocks.analytics.time_in_zone.v3',
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "PathDeviationBlock",
        "type": "roboflow_core/path_deviation_analytics@v2",
        "module": 'roboflow_workflows.v2.blocks.analytics.path_deviation_analytics.v2',
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "DetectionEventLogBlock",
        "type": "roboflow_core/detection_event_log@v1",
        "module": 'roboflow_workflows.v2.blocks.analytics.detection_event_log.v1',
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "OverlapFilterBlock",
        "type": "roboflow_core/overlap@v1",
        "module": 'roboflow_workflows.v2.blocks.analytics.overlap.v1',
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "BackgroundSubtractionBlock",
        "type": "roboflow_core/background_subtraction@v1",
        "module": 'roboflow_workflows.v2.blocks.analytics.background_subtraction.v1',
        "package": "analytics",
    },
    {
        "owner": "G05",
        "class": "MotionDetectionBlock",
        "type": "roboflow_core/motion_detection@v1",
        "module": 'roboflow_workflows.v2.blocks.analytics.motion_detection.v1',
        "package": "analytics",
    },
    {
        "owner": "G12",
        "class": "OnvifTrackingBlock",
        "type": "roboflow_core/onvif_sink@v1",
        "module": 'roboflow_workflows.v2.blocks.industrial.onvif_sink.v1',
        "package": "industrial",
    },
    {
        "owner": "G08",
        "class": "BackgroundColorVisualizationBlock",
        "type": "roboflow_core/background_color_visualization@v1",
        "module": 'roboflow_workflows.v2.blocks.visualization.annotations.background_color_visualization.v1',
        "package": "visualization.annotations",
    },
    {
        "owner": "G08",
        "class": "ColorVisualizationBlock",
        "type": "roboflow_core/color_visualization@v1",
        "module": 'roboflow_workflows.v2.blocks.visualization.annotations.color_visualization.v1',
        "package": "visualization.annotations",
    },
    {
        "owner": "G08",
        "class": "GridVisualizationBlock",
        "type": "roboflow_core/grid_visualization@v1",
        "module": 'roboflow_workflows.v2.blocks.visualization.annotations.grid_visualization.v1',
        "package": "visualization.annotations",
    },
]
BASELINE = {
    "block_types": [
        "roboflow_core/crop@v1",
        "roboflow_core/static_crop@v1",
        "roboflow_core/resize@v1",
        "roboflow_core/invert@v1",
        "roboflow_core/mosaic@v1",
        "roboflow_core/has_brightness@v1",
        "roboflow_core/absolute_static_crop@v1",
        "roboflow_core/relative_statoic_crop@v1",
        "roboflow_core/dynamic_crop@v1",
        "roboflow_core/image_preprocessing@v1",
        "roboflow_core/image_slicer@v2",
        "roboflow_core/image_copy@v1",
        "roboflow_core/convert_grayscale@v1",
        "roboflow_core/image_blur@v1",
        "roboflow_core/threshold@v1",
        "roboflow_core/mask_edge_snap@v1",
        "roboflow_core/morphological_transformation@v2",
        "roboflow_core/contrast_enhancement@v1",
        "roboflow_core/contrast_equalization@v1",
        "roboflow_core/delta_filter@v1",
        "roboflow_core/rate_limiter@v1",
        "roboflow_core/switch_case@v1",
        "roboflow_core/continue_if_boolean@v1",
        "roboflow_core/continue_if@v1",
        "roboflow_core/best_frame@v1",
        "roboflow_core/top_k_brightest@v1",
        "roboflow_core/frame_delay@v1",
        "roboflow_core/image_stack@v1",
        "roboflow_core/cache_get@v1",
        "roboflow_core/cache_set@v1",
        "roboflow_core/state_get@v1",
        "roboflow_core/state_set@v1",
        "roboflow_core/state_increment@v1",
        "roboflow_core/state_compare_and_set@v1",
        "roboflow_core/state_machine_set@v1",
        "roboflow_core/cosmos3_edge@v1",
        "roboflow_core/florence_2@v2",
        "roboflow_core/google_vision_ocr@v1",
        "roboflow_core/moondream2@v1",
        "roboflow_core/qwen_vlm@v4",
        "roboflow_core/roboflow_object_detection_model@v3",
        "roboflow_core/roboflow_classification_model@v3",
        "roboflow_core/roboflow_multi_label_classification_model@v3",
        "roboflow_core/roboflow_keypoint_detection_model@v3",
        "roboflow_core/roboflow_instance_segmentation_model@v4",
        "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "roboflow_core/roboflow_action_recognition_model@v1",
        "roboflow_core/segment_anything@v1",
        "roboflow_core/segment_anything_2_video@v1",
        "roboflow_core/sam3@v3",
        "roboflow_core/sam3_interactive@v1",
        "roboflow_core/sam3_video@v1",
        "roboflow_core/trackers_sort@v1",
        "roboflow_core/trackers_bytetrack@v1",
        "roboflow_core/trackers_ocsort@v1",
        "roboflow_core/trackers_botsort@v1",
        "roboflow_core/track_class_lock@v1",
        "roboflow_core/stabilize_detections@v1",
        "roboflow_core/velocity@v1",
        "roboflow_core/detections_stitch@v1",
        "roboflow_core/detection_offset@v1",
        "roboflow_core/bounding_rect@v1",
        "roboflow_core/detections_classes_replacement@v1",
        "roboflow_core/dimension_collapse@v1",
        "roboflow_core/detections_filter@v1",
        "roboflow_core/detections_transformation@v1",
        "roboflow_core/per_class_confidence_filter@v1",
        "roboflow_core/detections_combine@v1",
        "roboflow_core/detections_merge@v1",
        "roboflow_core/detections_consensus@v1",
        "roboflow_core/detections_difference@v1",
        "roboflow_core/overlap_analysis@v1",
        "roboflow_core/detections_list_rollup@v1",
        "roboflow_core/identify_changes@v1",
        "roboflow_core/identify_outliers@v1",
        "roboflow_core/property_definition@v1",
        "roboflow_core/expression@v1",
        "roboflow_core/string_template@v1",
        "roboflow_core/csv_formatter@v1",
        "roboflow_core/first_non_empty_or_default@v1",
        "roboflow_core/current_time@v1",
        "roboflow_core/json_parser@v1",
        "roboflow_core/detection_drawing@v1",
        "roboflow_core/simple_box_visualization@v1",
        "roboflow_core/simple_label_visualization@v1",
        "roboflow_core/detection_visualization@v1",
        "roboflow_core/simple_label_visualization_in_place@v1",
        "roboflow_core/bounding_box_visualization@v1",
        "roboflow_core/label_visualization@v2",
        "roboflow_core/polygon_visualization@v2",
        "roboflow_core/mask_visualization@v1",
        "roboflow_core/keypoint_visualization@v1",
        "roboflow_core/classification_label_visualization@v1",
        "roboflow_core/crop_visualization@v1",
        "roboflow_core/blur_visualization@v1",
        "roboflow_core/pixelate_visualization@v1",
        "roboflow_core/semantic_segmentation_visualization@v1",
        "roboflow_core/circle_visualization@v1",
        "roboflow_core/corner_visualization@v1",
        "roboflow_core/dot_visualization@v1",
        "roboflow_core/ellipse_visualization@v1",
        "roboflow_core/halo_visualization@v2",
        "roboflow_core/triangle_visualization@v1",
        "roboflow_core/heatmap_visualization@v1",
        "roboflow_core/trace_visualization@v1",
        "roboflow_core/polygon_zone_visualization@v1",
        "roboflow_core/line_counter_visualization@v1",
        "roboflow_core/reference_path_visualization@v1",
        "roboflow_core/asset_library_attributes@v1",
        "roboflow_core/roboflow_dataset_upload@v2",
        "roboflow_core/model_monitoring_inference_aggregator@v1",
        "roboflow_core/roboflow_custom_metadata@v1",
        "roboflow_core/visual_search@v1",
        "roboflow_core/visual_search_classifier@v1",
        "roboflow_core/roboflow_vision_events@v1",
        "roboflow_core/vision_event_bundle@v1",
        "roboflow_core/distance_measurement@v1",
        "roboflow_core/size_measurement@v1",
        "roboflow_core/mask_area_measurement@v1",
        "roboflow_core/pixel_color_count@v1",
    ],
    "operator_types": ["roboflow_core/align@v1", "roboflow_core/window@v1"],
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
def test_exact_22_additions_preserve_current_registry_exports_and_schemas(route):
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
            "roboflow_core/icon_visualization@v1",
            "roboflow_core/rich_label_visualization@v1",
            "roboflow_core/model_comparison_visualization@v1",
            "roboflow_core/clip_comparison@v2",
            "roboflow_core/clip@v1",
            "roboflow_core/perception_encoder@v1",
            "roboflow_core/depth_estimation@v1",
            "roboflow_core/cosine_similarity@v1",
            "roboflow_core/glm_ocr@v1",
            "roboflow_core/ocr_model@v1",
            "roboflow_core/easy_ocr@v1",
            "roboflow_core/pp_ocr@v1",
            "roboflow_core/smolvlm2@v1",
            "roboflow_core/camera_focus@v2",
            "roboflow_core/stitch_images@v1",
            "roboflow_core/auto_rotate_on_edges@v1",
            "roboflow_core/stitch_ocr_detections@v2",
            "roboflow_core/sift@v1",
            "roboflow_core/perspective_correction@v1",
            "roboflow_core/template_matching@v1",
            "roboflow_core/geotag_detection@v1",
            "roboflow_core/detections_nearest_neighbor@v1",
            "roboflow_core/sift_comparison@v2",
            "roboflow_core/camera-calibration@v1",
            "roboflow_core/dominant_color@v1",
            "roboflow_core/qr_code_generator@v1",
            "roboflow_core/barcode_detector@v1",
            "roboflow_core/qr_code_detector@v1",
            "roboflow_core/contours_detection@v1",
            "roboflow_core/dynamic_zone@v1",
            "roboflow_core/stability_ai_image_gen@v1",
            "roboflow_core/stability_ai_inpainting@v1",
            "roboflow_core/stability_ai_outpainting@v1",
            "roboflow_core/data_aggregator@v1",
            "roboflow_core/email_notification@v2",
            "roboflow_core/environment_secrets_store@v1",
            "roboflow_core/local_file_sink@v1",
            "roboflow_core/s3_sink@v1",
            "roboflow_core/slack_notification@v1",
            "roboflow_core/text_display@v1",
            "roboflow_core/twilio_sms_notification@v2",
            "roboflow_core/webhook_sink@v1",
        )
    )
    assert len(added) == 22
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
    assert "roboflow_core/window@v1" in catalogue.operator_types
    assert not hasattr(blocks, "BufferBlock")
    assert {name for values in catalogue.providers.values() for name in values} == {
        "object_cache"
    }


def test_retired_identities_stay_rejected_and_active_q3_is_selective():
    catalogue = blocks.create_catalogue()
    retired = [
        "roboflow_core/qwen25vl@v1",
        "roboflow_core/qwen3vl@v1",
        "roboflow_core/byte_tracker@v3",
        "roboflow_core/llama_vision@v2",
        "roboflow_core/qwen3_5_openrouter@v1",
        "roboflow_core/qwen3_6_openrouter@v1",
        "roboflow_core/vlm_as_classifier@v2",
        "roboflow_core/vlm_as_detector@v2",
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
    assert catalogue.find("roboflow_core/smolvlm2@v1") is not None
    assert catalogue.block_types.count("roboflow_core/qwen3_5vl@v2") == 1


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_active_q3_fake_prompts_keep_rows_through_default_catalogue(mode, monkeypatch):
    monkeypatch.setattr(q3, "_catalogue", blocks.create_catalogue)
    q3.test_selected_models_load_once_and_answers_keep_row_order(
        "roboflow_core/qwen3_5vl@v2", mode
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
