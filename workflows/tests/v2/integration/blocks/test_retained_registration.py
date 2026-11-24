"""Public retained discovery and representative compiled offline compositions."""

import importlib
import json
import pytest
from roboflow_workflows.v2 import blocks
from roboflow_workflows.v2.engine import Catalogue
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.data.secrets import SECRET_KIND

RECORDS = [
    {
        "type": "roboflow_core/icon_visualization@v1",
        "class_name": "IconVisualizationBlock",
        "module": 'roboflow_workflows.v2.blocks.visualization.annotations.icon_visualization.v1',
        "package": "visualization.annotations",
    },
    {
        "type": "roboflow_core/rich_label_visualization@v1",
        "class_name": "RichLabelVisualizationBlock",
        "module": 'roboflow_workflows.v2.blocks.visualization.annotations.rich_label_visualization.v1',
        "package": "visualization.annotations",
    },
    {
        "type": "roboflow_core/model_comparison_visualization@v1",
        "class_name": "ModelComparisonVisualizationBlock",
        "module": 'roboflow_workflows.v2.blocks.visualization.annotations.model_comparison_visualization.v1',
        "package": "visualization.annotations",
    },
    {
        "type": "roboflow_core/clip_comparison@v2",
        "class_name": "ClipComparisonBlock",
        "module": 'roboflow_workflows.v2.blocks.models.clip_comparison.v2',
        "package": "models",
    },
    {
        "type": "roboflow_core/clip@v1",
        "class_name": "ClipEmbeddingBlock",
        "module": 'roboflow_workflows.v2.blocks.models.clip.v1',
        "package": "models",
    },
    {
        "type": "roboflow_core/perception_encoder@v1",
        "class_name": "PerceptionEncoderEmbeddingBlock",
        "module": 'roboflow_workflows.v2.blocks.models.perception_encoder.v1',
        "package": "models",
    },
    {
        "type": "roboflow_core/depth_estimation@v1",
        "class_name": "DepthEstimationBlock",
        "module": 'roboflow_workflows.v2.blocks.models.depth_estimation.v1',
        "package": "models",
    },
    {
        "type": "roboflow_core/cosine_similarity@v1",
        "class_name": "CosineSimilarityBlock",
        "module": 'roboflow_workflows.v2.blocks.transforms.cosine_similarity.v1',
        "package": "transforms",
    },
    {
        "type": "roboflow_core/glm_ocr@v1",
        "class_name": "GlmOcrBlock",
        "module": 'roboflow_workflows.v2.blocks.models.glm_ocr.v1',
        "package": "models",
    },
    {
        "type": "roboflow_core/ocr_model@v1",
        "class_name": "OCRModelBlock",
        "module": 'roboflow_workflows.v2.blocks.models.ocr_model.v1',
        "package": "models",
    },
    {
        "type": "roboflow_core/easy_ocr@v1",
        "class_name": "EasyOCRBlock",
        "module": 'roboflow_workflows.v2.blocks.models.easy_ocr.v1',
        "package": "models",
    },
    {
        "type": "roboflow_core/pp_ocr@v1",
        "class_name": "PPOCRBlock",
        "module": 'roboflow_workflows.v2.blocks.models.pp_ocr.v1',
        "package": "models",
    },
    {
        "type": "roboflow_core/smolvlm2@v1",
        "class_name": "SmolVLM2Block",
        "module": 'roboflow_workflows.v2.blocks.models.smolvlm2.v1',
        "package": "models",
    },
    {
        "type": "roboflow_core/camera_focus@v2",
        "class_name": "CameraFocusBlock",
        "module": 'roboflow_workflows.v2.blocks.image.camera_focus.v2',
        "package": "image",
    },
    {
        "type": "roboflow_core/stitch_images@v1",
        "class_name": "StitchImagesBlock",
        "module": 'roboflow_workflows.v2.blocks.image.stitch_images.v1',
        "package": "image",
    },
    {
        "type": "roboflow_core/auto_rotate_on_edges@v1",
        "class_name": "AutoRotateOnEdgesBlock",
        "module": 'roboflow_workflows.v2.blocks.image.auto_rotate_on_edges.v1',
        "package": "image",
    },
    {
        "type": "roboflow_core/stitch_ocr_detections@v2",
        "class_name": "StitchOCRDetectionsBlock",
        "module": 'roboflow_workflows.v2.blocks.codes.stitch_ocr_detections.v2',
        "package": "codes",
    },
    {
        "type": "roboflow_core/sift@v1",
        "class_name": "SiftBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.sift.v1',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/perspective_correction@v1",
        "class_name": "PerspectiveCorrectionBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.perspective_correction.v1',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/template_matching@v1",
        "class_name": "TemplateMatchingBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.template_matching.v1',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/geotag_detection@v1",
        "class_name": "GeotagDetectionBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.geotag_detection.v1',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/detections_nearest_neighbor@v1",
        "class_name": "DetectionsNearestNeighborBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.detections_nearest_neighbor.v1',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/sift_comparison@v2",
        "class_name": "SiftComparisonBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.sift_comparison.v2',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/camera-calibration@v1",
        "class_name": "CameraCalibrationBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.camera_calibration.v1',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/dominant_color@v1",
        "class_name": "DominantColorBlock",
        "module": 'roboflow_workflows.v2.blocks.image.dominant_color.v1',
        "package": "image",
    },
    {
        "type": "roboflow_core/qr_code_generator@v1",
        "class_name": "QRCodeGeneratorBlock",
        "module": 'roboflow_workflows.v2.blocks.codes.qr_code_generator.v1',
        "package": "codes",
    },
    {
        "type": "roboflow_core/barcode_detector@v1",
        "class_name": "BarcodeDetectorBlock",
        "module": 'roboflow_workflows.v2.blocks.codes.barcode_detector.v1',
        "package": "codes",
    },
    {
        "type": "roboflow_core/qr_code_detector@v1",
        "class_name": "QRCodeDetectorBlock",
        "module": 'roboflow_workflows.v2.blocks.codes.qr_code_detector.v1',
        "package": "codes",
    },
    {
        "type": "roboflow_core/contours_detection@v1",
        "class_name": "ContoursDetectionBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.contours_detection.v1',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/dynamic_zone@v1",
        "class_name": "DynamicZoneBlock",
        "module": 'roboflow_workflows.v2.blocks.geometry.dynamic_zone.v1',
        "package": "geometry",
    },
    {
        "type": "roboflow_core/stability_ai_image_gen@v1",
        "class_name": "StabilityAIImageGenerationBlock",
        "module": 'roboflow_workflows.v2.blocks.generation.stability_ai_image_gen.v1',
        "package": "generation",
    },
    {
        "type": "roboflow_core/stability_ai_inpainting@v1",
        "class_name": "StabilityAIInpaintingBlock",
        "module": 'roboflow_workflows.v2.blocks.generation.stability_ai_inpainting.v1',
        "package": "generation",
    },
    {
        "type": "roboflow_core/stability_ai_outpainting@v1",
        "class_name": "StabilityAIOutpaintingBlock",
        "module": 'roboflow_workflows.v2.blocks.generation.stability_ai_outpainting.v1',
        "package": "generation",
    },
    {
        "type": "roboflow_core/data_aggregator@v1",
        "class_name": "DataAggregatorBlock",
        "module": 'roboflow_workflows.v2.blocks.analytics.data_aggregator.v1',
        "package": "analytics",
    },
    {
        "type": "roboflow_core/email_notification@v2",
        "class_name": "EmailNotificationBlock",
        "module": 'roboflow_workflows.v2.blocks.sinks.email_notification.v2',
        "package": "sinks",
    },
    {
        "type": "roboflow_core/environment_secrets_store@v1",
        "class_name": "EnvironmentSecretsStoreBlock",
        "module": 'roboflow_workflows.v2.blocks.secrets.environment_secrets_store.v1',
        "package": "secrets",
    },
    {
        "type": "roboflow_core/local_file_sink@v1",
        "class_name": "LocalFileSinkBlock",
        "module": 'roboflow_workflows.v2.blocks.sinks.local_file_sink.v1',
        "package": "sinks",
    },
    {
        "type": "roboflow_core/s3_sink@v1",
        "class_name": "S3SinkBlock",
        "module": 'roboflow_workflows.v2.blocks.sinks.s3_sink.v1',
        "package": "sinks",
    },
    {
        "type": "roboflow_core/slack_notification@v1",
        "class_name": "SlackNotificationBlock",
        "module": 'roboflow_workflows.v2.blocks.sinks.slack_notification.v1',
        "package": "sinks",
    },
    {
        "type": "roboflow_core/text_display@v1",
        "class_name": "TextDisplayBlock",
        "module": 'roboflow_workflows.v2.blocks.visualization.annotations.text_display.v1',
        "package": "visualization.annotations",
    },
    {
        "type": "roboflow_core/twilio_sms_notification@v2",
        "class_name": "TwilioSmsNotificationBlock",
        "module": 'roboflow_workflows.v2.blocks.sinks.twilio_sms_notification.v2',
        "package": "sinks",
    },
    {
        "type": "roboflow_core/webhook_sink@v1",
        "class_name": "WebhookSinkBlock",
        "module": 'roboflow_workflows.v2.blocks.sinks.webhook_sink.v1',
        "package": "sinks",
    },
]


@pytest.mark.parametrize("route", ["create", "plugin", "factory"])
def test_default_identity_export_schema_and_operator_contract(route):
    catalogue = (
        blocks.create_catalogue()
        if route == "create"
        else Catalogue.from_modules(
            ["roboflow_workflows.v2.blocks"],
            **({"attribute": "create_catalogue"} if route == "factory" else {})
        )
    )
    description = catalogue.describe()
    json.dumps(description)
    schemas = {row["type"]: row["params_schema"] for row in description["blocks"]}
    for row in RECORDS:
        cls = getattr(importlib.import_module(row["module"]), row["class_name"])
        package = importlib.import_module(
            "roboflow_workflows.v2.blocks." + row["package"]
        )
        assert (
            getattr(blocks, row["class_name"])
            is getattr(package, row["class_name"])
            is cls
        )
        if row["package"] != "image":
            assert package.BLOCKS.count(cls) == 1
        assert (
            blocks.__all__.count(row["class_name"])
            == package.__all__.count(row["class_name"])
            == 1
        )
        assert catalogue.block_types.count(row["type"]) == 1
        assert catalogue.entry(row["type"]).spec is spec_of(cls)
        assert schemas[row["type"]] == spec_of(cls).params_model.model_json_schema()
    assert catalogue.kind("secret") is SECRET_KIND
    assert "roboflow_core/window@v1" in catalogue.operator_types
    assert not hasattr(blocks, "BufferBlock")


@pytest.mark.parametrize(
    "journey", ["embeddings", "ocr", "features", "camera", "ocr_stitch"]
)
def test_default_compiled_cpu_journeys(journey, monkeypatch):
    catalogue = blocks.create_catalogue()
    if journey == "embeddings":
        from tests.v2.integration.blocks.models import test_embedding_workflows as owner

        monkeypatch.setattr(owner, "CATALOGUE", catalogue)
        owner.test_image_and_literal_text_embeddings_compare_per_image()
    elif journey == "ocr":
        from tests.v2.integration.blocks.models import test_ocr_workflows as owner

        monkeypatch.setattr(owner, "_catalogue", blocks.create_catalogue)
        owner.test_three_ocr_blocks_share_one_cache_and_load_each_model_once("run")
    elif journey == "features":
        from tests.v2.integration.blocks.geometry import (
            test_feature_matching_workflows as owner,
        )

        original = owner.compile_workflow

        def compile_default(definition, **kwargs):
            kwargs["catalogue"] = catalogue
            return original(definition, **kwargs)

        monkeypatch.setattr(owner, "compile_workflow", compile_default)
        owner.test_matches_pair_up_and_geotag_per_image_of_a_batch()
    elif journey == "camera":
        from tests.v2.integration.blocks.geometry import (
            test_camera_geometry_workflows as owner,
        )
        from tests.v2.integration.blocks.geometry.deterministic import (
            MarkerDetectorBlock,
        )

        monkeypatch.setattr(
            owner,
            "CATALOGUE",
            catalogue.with_blocks(
                [MarkerDetectorBlock, owner.MarkerCornerKeyPointsBlock]
            ),
        )
        owner.test_undistort_warp_and_score_focus_per_box("run")
    else:
        from tests.v2.integration.blocks.image import test_stitching_workflows as owner

        monkeypatch.setattr(owner, "_catalogue", lambda base=None: catalogue)
        owner.test_ocr_text_per_image_with_auto_direction_and_empty_image()


def test_secret_provider_and_geometry_use_final_public_catalogue(monkeypatch):
    from tests.v2.integration.blocks.models.vision_language_models import (
        test_openrouter_vendor_workflows as owner,
    )

    monkeypatch.setattr(owner, "CATALOGUE", blocks.create_catalogue())
    owner.test_meta_detections_from_a_secret_keyed_request_feed_a_geometry_step()


def test_default_secret_store_output_feeds_real_provider_and_geometry():
    from roboflow_workflows.v2.engine import compile_workflow
    from roboflow_workflows.v2.blocks.secrets import StaticSecrets
    from tests.v2.integration.blocks.models.vision_language_models import (
        test_openrouter_vendor_workflows as owner,
    )

    definition = {
        "version": "2.0",
        "inputs": [owner.IMAGE_INPUT],
        "steps": [
            {
                "type": "roboflow_core/environment_secrets_store@v1",
                "name": "secrets",
                "variables_storing_secrets": ["OPENROUTER_KEY"],
            },
            {
                "type": "roboflow_core/meta_vlm@v3",
                "name": "muse",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["car"],
                "api_key": "$steps.secrets.openrouter_key",
            },
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "offset",
                "predictions": "$steps.muse.predictions",
                "offset_width": 10,
                "offset_height": 10,
            },
        ],
        "outputs": [
            owner._output("boxes", "$steps.offset.predictions"),
            owner._output("secret", "$steps.secrets.openrouter_key"),
        ],
    }
    plan = compile_workflow(definition, catalogue=blocks.create_catalogue())
    server = owner._answering(
        '[{"label":"car","x_min":0,"y_min":0,"x_max":500,"y_max":500}]'
    )
    with plan.create_session(
        resources={
            "secrets": StaticSecrets({"OPENROUTER_KEY": owner.OPENROUTER_KEY}),
            "openrouter_http_client": server.http_client(),
        }
    ) as session:
        result = session.run({"image": [owner._image(200, 100)]})
    assert result.rows()[0]["boxes"].xyxy.tolist() == [[0.0, 0.0, 105.0, 55.0]]
    assert (
        server.requests[0].headers["authorization"] == "Bearer " + owner.OPENROUTER_KEY
    )
    assert owner.OPENROUTER_KEY not in json.dumps(result.rows(serialize=True))


def test_default_notification_borrows_recording_client():
    from roboflow_workflows.v2.engine import compile_workflow
    from roboflow_workflows.v2.blocks.secrets import StaticSecrets

    calls = []

    class RecordingSmtp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def login(self, sender, password):
            calls.append(("login", sender, password))

        def sendmail(self, sender, envelope, mime):
            calls.append(("send", sender, envelope, mime))

    definition = {
        "version": "2.0",
        "inputs": [],
        "steps": [
            {
                "type": "roboflow_core/environment_secrets_store@v1",
                "name": "secrets",
                "variables_storing_secrets": ["SMTP_PASSWORD"],
            },
            {
                "type": "roboflow_core/email_notification@v2",
                "name": "email",
                "email_provider": "Custom SMTP",
                "subject": "Gate alert",
                "receiver_email": "ops@example.test",
                "message": "Three trucks",
                "sender_email": "camera@example.test",
                "smtp_server": "unused",
                "sender_email_password": "$steps.secrets.smtp_password",
                "fire_and_forget": False,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "message", "selector": "$steps.email.message"}
        ],
    }
    plan = compile_workflow(definition, catalogue=blocks.create_catalogue())
    with plan.create_session(
        resources={
            "secrets": StaticSecrets({"SMTP_PASSWORD": "retained-sentinel"}),
            "smtp_connect": lambda host, port: RecordingSmtp(),
        }
    ) as session:
        result = session.run({})
    assert result.rows() == [{"message": "Notification sent successfully"}]
    assert calls[0] == ("login", "camera@example.test", "retained-sentinel")
    assert calls[1][0] == "send" and "Three trucks" in calls[1][3]
    assert "retained-sentinel" not in json.dumps(result.rows(serialize=True))


def test_selection_constructs_no_retained_block(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Catalogue selection constructed a retained block")

    for row in RECORDS:
        cls = getattr(importlib.import_module(row["module"]), row["class_name"])
        monkeypatch.setattr(cls, "__init__", forbidden)
    catalogue = blocks.create_catalogue()
    assert all(catalogue.find(row["type"]) is not None for row in RECORDS)
