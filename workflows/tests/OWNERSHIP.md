# Workflows test ownership inventory

> Historical inventory captured before Phase E moved tests physically. Rows
> retain their ORIGINAL source paths so a reviewer can grep the plan and see
> where each file used to live; the *Executed state* section below records
> what actually happened. The one-off migration script has been removed;
> the inventory is the durable artifact. No commit was created by this task.

Generated for MOVE_WORKFLOWS_PLAN.MD Phase E. Source root: `tests/workflows/unit_tests`.

Total: **456** unit test files.

| Action | Count | Destination |
| --- | --- | --- |
| MOVE | 369 | `workflows/tests/unit_tests/` |
| MOVE_ENV | 19 | `workflows/tests/unit_tests/` (rewrite tensor selection to explicit config) |
| SPLIT | 25 | Split: pure part -> `workflows/tests/unit_tests/`; server parity retained in `tests/workflows/unit_tests/` |
| RETAIN | 43 | Kept in `tests/workflows/unit_tests/` (server adapters/parity/platform) |

### Executed state

* Final manifest: 385 moved (`MOVE` + `MOVE_ENV`), 70 retained, and one split.
  Physical trees contain 387 package test files (385 moved, split configuration,
  new import-boundary guard) and 71 retained files (70 retained, split configuration).
* SPLIT executed in this pass:
  * `test_configuration.py` — pure config assertions moved (field count 75,
    4 new field asserts). Retained shim keeps the differential AST harness
    and subprocess parity test that both reach into `inference/core/env.py`.
  * `test_configuration_injection.py`, `test_step_error_handler_default.py`,
    `test_plugin_block_source.py` — moved wholesale (only aliased workflows
    imports; the residual `"inference.core.exceptions"` / `"inference.core.env"`
    strings are decontamination assertions checking those names are *absent*
    from sys.modules or plugin source, not load targets).
* SPLIT rows reclassified as RETAIN (20 files) — audited per-file; each
  reaches into server response models (`inference.core.entities.responses.*`),
  server request models (`inference.core.entities.requests.*`), the platform
  plugin (`inference.roboflow_workflows_plugin.*`), server-side utils
  (`inference.core.utils.*`), server-side environment vars, server-side
  exception types (`inference.core.exceptions`), or `inference.core.roboflow_api`.
  These are server-parity/adapter suites; the correct home is the retained
  server pytest job. Rationale kept per file in §Split reclassifications.
* Modal server-source tests reclassified as RETAIN — moved back to
  `tests/workflows/unit_tests/execution_engine/dynamic_blocs/`. They load
  `<repo>/modal/modal_app.py` by path (deployment code that is NOT part of
  the roboflow-workflows package) and belong with the server suite.
* Helper stubs copied under `workflows/tests/unit_tests/` (rewritten to
  `roboflow_workflows` where they referenced aliased workflows modules):
  `prototypes/platform_client_double.py`,
  `core_steps/_vlm_prediction_readers.py`,
  `core_steps/models/roboflow/_hosted_api_resolution.py`,
  `core_steps/trackers/conftest.py`,
  `execution_engine/compiler/plugin_with_test_blocks/**`,
  `execution_engine/introspection/plugin_*` (4 plugin dirs),
  `execution_engine/executor/execution_data_manager/common.py`.
  `execution_engine/dynamic_blocs/_workspace_resolver_stub.py` is a NEW file
  that exposes just `StubResolver` for `test_block_scaffolding.py`; the
  RETAINed `test_workspace_resolver.py` (server-parity) is not duplicated.
* Package-owned assets under `workflows/tests/assets/` and
  `workflows/tests/unit_tests/core_steps/models/third_party/assets/`. The
  `dogs_image`, `barcode_image`, `qr_codes_image` fixtures resolve from the
  package tree only; no fixture reaches into `tests/workflows/...`.
* Package-owned `workflows/tests/stub_plugins/scalar_only_block_plugin/` -
  used by `execution_engine/inner_workflow/test_inline*.py` via a
  `get_plugin_modules` monkeypatch to `tests.stub_plugins.scalar_only_block_plugin`.
* Isolation probe (`workflows/scripts/workflows_isolation_probe.py`):
  installs `<wheel>[enterprise]` so the enterprise plugin's declared
  dependencies participate in pip's resolver, sets
  `WORKFLOWS_PLUGINS=roboflow_workflows.enterprise_blocks.loader` in the
  child env, and asserts every declared enterprise block is present in
  `load_workflow_blocks()` with `block_source == workflows_core`. Merges
  `WORKFLOWS_ISOLATION_FIND_LINKS` (pathsep-separated) into `--find-links`
  for CI-local wheel resolution. The child regression tests
  (`workflows/tests/isolation/test_isolation_probe.py`) cover `_child_env`,
  `_validate_results` against empty/malformed child output, and the
  `PYTHONOPTIMIZE` guard.

### Split reclassifications

Server-parity tests kept RETAIN with explicit reasons:

* `test_classification_results_operations.py`, `test_property_extraction.py`,
  `test_detections_classes_replacement.py`, `test_representation_boundary.py`,
  `test_segment_anything2.py` — assert against
  `inference.core.entities.responses.inference.*` shapes returned by server
  adapters; the block itself is exercised standalone in the moved tests.
* `test_segmentation_entities.py`, `test_sam_prompts.py`,
  `test_inner_workflow_dispatch.py` — pin `inference.core.entities.*` request/
  response wire formats.
* `test_openrouter.py`, `test_cog_vlm_deprecated.py`, `test_gaze_deprecated.py`,
  `test_twilio_sms_v2.py` — exercise server exception types
  (`inference.core.exceptions.*`) that the server adapter raises through
  the block.
* `test_platform_client_headers.py` — checks the block never imports
  `inference.core.roboflow_api` / `inference.core.utils.url_utils` (server
  decontamination assertion; must run alongside the server modules to detect
  regressions).
* `test_load_core_model.py` — verifies bridging into
  `inference.core.roboflow_api.ModelEndpointType`.
* `test_depth_estimation.py` — parity against
  `inference.core.utils.depth_encoding`.
* `test_dependent_resources.py` — coverage of the
  `inference.roboflow_workflows_plugin` platform plugin.
* `test_image_codec.py`, `test_image_encoding.py`, `test_images.py`,
  `test_images_delegates.py`, `test_text.py` — server-utility parity
  (`inference.core.utils.*` / `inference.core.warnings`).

## MOVE (369 files)

- `tests/workflows/unit_tests/core_steps/analytics/test_coords_overlap_v1.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_data_aggregator_v1.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_detection_event_log_v1.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_line_counter_v1.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_line_counter_v2.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_line_counter_v2_tensor.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_path_deviation_v1.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_path_deviation_v2.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_tensor_native_empty_selection.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_time_in_zone_v1.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_time_in_zone_v2.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_time_in_zone_v3.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_time_in_zone_v3_tensor.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_velocity.py`
- `tests/workflows/unit_tests/core_steps/analytics/test_zone_geometry_parity.py`
- `tests/workflows/unit_tests/core_steps/cache/test_cache.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_auto_rotate_on_edges.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_blur.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_camera_focus.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_camera_focus_v2.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_contours_detection.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_contrast_enhancement.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_contrast_equalization.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_distance_measurement.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_dominant_color.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_grayscale.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_image_preprocessing.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_mask_area_measurement.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_mask_edge_snap.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_morphological_transform.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_morphological_transformation.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_morphological_transformation_v2.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_motion_detection.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_pixel_color_count.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_sift.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_sift_comparison.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_sift_comparison_v2.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_size_measurement.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_template_matching.py`
- `tests/workflows/unit_tests/core_steps/classical_cv/test_threshold.py`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/test_dictionaries_operations.py`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/test_image_operations.py`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/test_numbers_operations.py`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/test_sequences_operations.py`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/test_strings_operations.py`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/test_video_frame_operations.py`
- `tests/workflows/unit_tests/core_steps/common/query_language/test_introspection_operations.py`
- `tests/workflows/unit_tests/core_steps/common/test_deserializers.py`
- `tests/workflows/unit_tests/core_steps/common/test_detections_mismatch.py`
- `tests/workflows/unit_tests/core_steps/common/test_keypoints.py`
- `tests/workflows/unit_tests/core_steps/common/test_platform_client_injection.py`
- `tests/workflows/unit_tests/core_steps/common/test_reasoning_contract.py`
- `tests/workflows/unit_tests/core_steps/common/test_rle_compact.py`
- `tests/workflows/unit_tests/core_steps/common/test_rle_embed.py`
- `tests/workflows/unit_tests/core_steps/common/test_serializers.py`
- `tests/workflows/unit_tests/core_steps/common/test_tensor_native_host_mirror.py`
- `tests/workflows/unit_tests/core_steps/common/test_tensor_native_root_coordinates.py`
- `tests/workflows/unit_tests/core_steps/common/test_tensor_native_selection.py`
- `tests/workflows/unit_tests/core_steps/common/test_token_usage.py`
- `tests/workflows/unit_tests/core_steps/common/test_utils.py`
- `tests/workflows/unit_tests/core_steps/common/test_vlm_decoding_legacy_parity.py`
- `tests/workflows/unit_tests/core_steps/common/test_vlm_json.py`
- `tests/workflows/unit_tests/core_steps/control_flow/test_control_flow.py`
- `tests/workflows/unit_tests/core_steps/control_flow/test_rate_limiter.py`
- `tests/workflows/unit_tests/core_steps/control_flow/test_switch_case.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_anthropic_claude.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_clip.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_cosmos3.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_depth_estimation.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_easy_ocr.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_florence2.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_glm_ocr.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_google_gemini.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_google_gemma.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_kimi_openrouter.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_llama_vision.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_lmm.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_lmm_classifier.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_moondream2.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_ocr_doctr.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_openai.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_openai_compatible.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_openrouter.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_perception_encoder.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_pp_ocr.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_qwen25vl.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_qwen3_5_openrouter.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_qwen3_5vl.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_qwen3_6_openrouter.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_qwen3vl.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_qwen_vlm.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_roboflow_instance_segmentation.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_roboflow_keypoint_detection.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_roboflow_multi_class_classification.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_roboflow_multi_label_classification.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_roboflow_object_detection.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_roboflow_semantic_segmentation.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_segment_anything2.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_segment_anything2_video.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_segment_anything3.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_segment_anything3_3d.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_segment_anything3_interactive.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_segment_anything3_video.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_smolvlm.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_stability_ai_image_gen.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_yolo_world.py`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_zai_vlm.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_anthropic_detection_parsing.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_csv.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_current_time.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_empty_vlm_root_coordinates.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_expression.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_first_non_empty_or_default.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_gemini_detection_parsing.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_json_parser.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_muse_detection_parsing.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_openai_detection_parsing.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_qwen_detection_parsing.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_spacexai_detection_parsing.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_string_template.py`
- `tests/workflows/unit_tests/core_steps/formatters/test_tensor_formatters_no_materialization.py`
- `tests/workflows/unit_tests/core_steps/formatters/vlm_as_classifier/test_v1.py`
- `tests/workflows/unit_tests/core_steps/formatters/vlm_as_classifier/test_v2.py`
- `tests/workflows/unit_tests/core_steps/formatters/vlm_as_detector/test_v1.py`
- `tests/workflows/unit_tests/core_steps/formatters/vlm_as_detector/test_v1_tensor.py`
- `tests/workflows/unit_tests/core_steps/formatters/vlm_as_detector/test_v2.py`
- `tests/workflows/unit_tests/core_steps/formatters/vlm_as_detector/test_v2_tensor.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_buffer.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_detections_consensus.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_detections_consensus_v1_tensor.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_detections_difference_v1_tensor.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_detections_stitch.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_detections_stitch_v1_tensor.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_dimension_rollup.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_domension_collapse.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_frame_delay.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_image_stack.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_overlap_analysis.py`
- `tests/workflows/unit_tests/core_steps/fusion/test_overlap_analysis_v1_tensor.py`
- `tests/workflows/unit_tests/core_steps/math/test_cosine_similarity.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_anthropic_claude.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_anthropic_claude_model_capabilities.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_anthropic_claude_v4.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_anthropic_claude_v4_detection.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_anthropic_claude_v5.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_clip.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_clip_comparison.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_cosmos3.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_easy_ocr.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_florence2.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_glm_ocr.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_google_gemini.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_google_gemini_v5.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_google_gemini_v6.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_google_gemma.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_google_gemma_v2.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_google_gemma_v3.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_google_gemma_v4.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_image_prep_before_registration.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_kimi_openrouter.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_kimi_openrouter_v2.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_kimi_openrouter_v3.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_llama_3_2_vision.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_llama_vision_v2.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_lmm.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_lmm_classifier.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_meta_vlm_v1.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_meta_vlm_v2.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_meta_vlm_v3.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_ocr.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_offline_remote_execution.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_openai_compatible_v1.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_openai_v1.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_openai_v4.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_openai_v5.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_openai_v6.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_openai_v7.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_openrouter_v2.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_openrouter_v3.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_perception_encoder.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_pp_ocr.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_qwen3_5_openrouter.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_qwen3_5vl.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_qwen3_6_openrouter.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_qwen_vlm.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_qwen_vlm_v2.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_qwen_vlm_v3.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_qwen_vlm_v4.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_segment_anything3_3d.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_segment_anything3_interactive.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_segment_anything3_v3.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_spacexai.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_spacexai_v2.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_spacexai_v3.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_streaming_video_common.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_streaming_video_common_tensor.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_vlm_blocks_in_block_decoding_smoke.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_vlm_remote_execution.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_yolo_world.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_zai_vlm_v1.py`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_zai_vlm_v2.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/instance_segmentation/test_hosted_api_resolution.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/instance_segmentation/test_v1.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/instance_segmentation/test_v2.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/instance_segmentation/test_v3.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/instance_segmentation/test_v4.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/keypoint_detection/test_hosted_api_resolution.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/keypoint_detection/test_v1.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/keypoint_detection/test_v2.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/keypoint_detection/test_v3.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/multi_class_classification/test_hosted_api_resolution.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/multi_class_classification/test_v1.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/multi_class_classification/test_v2.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/multi_class_classification/test_v3.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/multi_label_classification/test_hosted_api_resolution.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/multi_label_classification/test_v1.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/multi_label_classification/test_v2.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/multi_label_classification/test_v3.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/object_detection/test_hosted_api_resolution.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/object_detection/test_v1.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/object_detection/test_v1_local_execution.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/object_detection/test_v2.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/object_detection/test_v3.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/semantic_segmentation/test_hosted_api_resolution.py`
- `tests/workflows/unit_tests/core_steps/models/roboflow/semantic_segmentation/test_v2.py`
- `tests/workflows/unit_tests/core_steps/models/third_party/test_barcode_detection.py`
- `tests/workflows/unit_tests/core_steps/models/third_party/test_qr_code_detection.py`
- `tests/workflows/unit_tests/core_steps/sampling/test_identify_changes.py`
- `tests/workflows/unit_tests/core_steps/sampling/test_identify_outliers.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_email.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_email_v2.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_email_v2_inline_images.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_event_writer_v1.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_local_file.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_modbus_tcp.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_mqtt_writer.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_onvif_movement.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_opc_writer.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_plc.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_plc_ethernetip.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_postgresql.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_s3_sink.py`
- `tests/workflows/unit_tests/core_steps/sinks/test_webhook.py`
- `tests/workflows/unit_tests/core_steps/test_init_files.py`
- `tests/workflows/unit_tests/core_steps/test_loader_block_filtering.py`
- `tests/workflows/unit_tests/core_steps/test_loader_tensor_mode_parity.py`
- `tests/workflows/unit_tests/core_steps/trackers/botsort/test_botsort_v1.py`
- `tests/workflows/unit_tests/core_steps/trackers/bytetrack/test_byte_track_v1.py`
- `tests/workflows/unit_tests/core_steps/trackers/ocsort/test_ocsort_v1.py`
- `tests/workflows/unit_tests/core_steps/trackers/sort/test_sort_v1.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_absolute_static_crop.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_bounding_rect.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_byte_track_v1.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_byte_track_v2.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_byte_track_v3.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_byte_tracker_tensor.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_crop.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_detection_offset.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_detections_combine.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_detections_merge.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_dynamic_zones.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_geotag_detection.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_image_slicer_v1.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_image_slicer_v2.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_per_class_confidence_filter.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_perspective_correction.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_qr_code_generator.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_relative_static_crop.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_stabilize_detections.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_stitch.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_stitch_ocr_detections_v1.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_stitch_ocr_detections_v2.py`
- `tests/workflows/unit_tests/core_steps/transformations/test_track_class_lock.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_adaptive_text_size.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_blur.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_bounding_box.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_bounding_box_v1_tensor_gpu.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_circle.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_classification_label.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_color.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_corner.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_crop.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_dot.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_ellipse.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_fonts_download.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_fonts_registry.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_grid.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_halo.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_host_mirror_viz.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_icon.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_icon_alpha.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_keypoints.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_label.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_label_v1_tensor_gpu.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_line_counter_zone.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_mask.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_mask_v1_tensor_gpu.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_model_comparison.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_negative_safe_palette.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_pixelate.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_polygon.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_polygon_zone.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_str_to_color.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_tensor_overlap_owner_regression.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_text_display.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_triangle.py`
- `tests/workflows/unit_tests/core_steps/visualizations/test_viz_phase_transfer_audit.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_cache.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_core.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_core_dynamic_blocks_in_inner_workflow.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_duplicate_dynamic_block_warning.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_graph_constructor.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_graph_traversal.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_reference_type_checker.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_steps_initialiser.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_utils.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_validator.py`
- `tests/workflows/unit_tests/execution_engine/compiler/test_workflow_schema_detections_property.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_block_assembler.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_debug_logs.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_app_result_serialization.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_code_hash.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_endpoint_urls.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_execution_timing.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_executor.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_semantic_segmentation_transport.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_ws_chunking.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_ws_contract.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_ws_server_dedup.py`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_workflow_debug.py`
- `tests/workflows/unit_tests/execution_engine/executor/execution_data_manager/test_branching_manager.py`
- `tests/workflows/unit_tests/execution_engine/executor/execution_data_manager/test_dynamic_batches_manager.py`
- `tests/workflows/unit_tests/execution_engine/executor/execution_data_manager/test_execution_cache.py`
- `tests/workflows/unit_tests/execution_engine/executor/execution_data_manager/test_step_input_assembler.py`
- `tests/workflows/unit_tests/execution_engine/executor/execution_data_manager/test_step_output_future_resolution.py`
- `tests/workflows/unit_tests/execution_engine/executor/test_flow_coordinator.py`
- `tests/workflows/unit_tests/execution_engine/executor/test_future_resolution_utils.py`
- `tests/workflows/unit_tests/execution_engine/executor/test_output_constructor.py`
- `tests/workflows/unit_tests/execution_engine/executor/test_run_steps_in_parallel.py`
- `tests/workflows/unit_tests/execution_engine/executor/test_runtime_input_validator.py`
- `tests/workflows/unit_tests/execution_engine/executor/test_step_observer.py`
- `tests/workflows/unit_tests/execution_engine/executor/test_stream_pipeline_flush.py`
- `tests/workflows/unit_tests/execution_engine/inner_workflow/test_compiler_bridge.py`
- `tests/workflows/unit_tests/execution_engine/inner_workflow/test_composition.py`
- `tests/workflows/unit_tests/execution_engine/inner_workflow/test_dynamic_blocks_collection.py`
- `tests/workflows/unit_tests/execution_engine/inner_workflow/test_inline.py`
- `tests/workflows/unit_tests/execution_engine/inner_workflow/test_inline_nested_continue_if_crop_lineage_shape.py`
- `tests/workflows/unit_tests/execution_engine/inner_workflow/test_parameter_bindings_validation.py`
- `tests/workflows/unit_tests/execution_engine/inner_workflow/test_reference_resolution.py`
- `tests/workflows/unit_tests/execution_engine/introspection/test_blocks_loader.py`
- `tests/workflows/unit_tests/execution_engine/introspection/test_connections_discovery.py`
- `tests/workflows/unit_tests/execution_engine/introspection/test_inputs_discovery.py`
- `tests/workflows/unit_tests/execution_engine/introspection/test_schema_parser.py`
- `tests/workflows/unit_tests/execution_engine/introspection/test_selectors_parser.py`
- `tests/workflows/unit_tests/execution_engine/introspection/test_types_discovery.py`
- `tests/workflows/unit_tests/execution_engine/introspection/test_utils.py`
- `tests/workflows/unit_tests/execution_engine/profiling/test_base_profiler.py`
- `tests/workflows/unit_tests/execution_engine/profiling/test_null_profiler.py`
- `tests/workflows/unit_tests/execution_engine/test_core.py`
- `tests/workflows/unit_tests/execution_engine/test_dependencies_pre_loading.py`
- `tests/workflows/unit_tests/execution_engine/test_image_codec_injection.py`
- `tests/workflows/unit_tests/execution_engine/test_model_manager_bind_hook.py`
- `tests/workflows/unit_tests/execution_engine/test_observer_resolution.py`
- `tests/workflows/unit_tests/prototypes/test_background_tasks.py`
- `tests/workflows/unit_tests/prototypes/test_observer.py`
- `tests/workflows/unit_tests/test_decontamination_lint.py` — plan Phase E table: decontamination/env scans/isolation move
- `tests/workflows/unit_tests/test_no_server_env_imports.py` — non-workflows refs: `inference.core.env` — plan Phase E table: decontamination/env scans/isolation move
- `tests/workflows/unit_tests/test_observer_decontamination.py` — plan Phase E table: decontamination/env scans/isolation move
- `tests/workflows/unit_tests/utils/test_url_input.py`

## MOVE_ENV (19 files)

- `tests/workflows/unit_tests/core_steps/classical_cv/test_detections_nearest_neighbor.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/detection/test_base.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/detections/test_base.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/common/query_language/operations/test_detections_operations.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/common/test_vlm_decoding.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/common/test_vlm_decoding_tensor.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_clip_comparison.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/fusion/test_detections_difference.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/fusion/test_frame_delay_v1_tensor.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/models/roboflow/semantic_segmentation/test_v1.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/models/roboflow/test_remote_api_key_transport.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/models/test_remote_billing_forwarding.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/visualizations/test_label_v2_and_font_schema.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/core_steps/visualizations/test_rich_label.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_block_scaffolding.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_configuration_source.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_ws_protocol.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_tensor_native_imports_lines.py` — non-workflows refs: `inference.core.env`
- `tests/workflows/unit_tests/execution_engine/entities/test_base.py` — non-workflows refs: `inference.core.env`

## SPLIT (25 files)

- `tests/workflows/unit_tests/core_steps/common/query_language/operations/test_classification_results_operations.py` — non-workflows refs: `inference.core.entities.responses.inference`, `inference.core.env`
- `tests/workflows/unit_tests/core_steps/common/test_load_core_model.py` — non-workflows refs: `inference.core.entities`, `inference.core.roboflow_api`
- `tests/workflows/unit_tests/core_steps/common/test_openrouter.py` — non-workflows refs: `inference.core.exceptions`
- `tests/workflows/unit_tests/core_steps/common/test_segmentation_entities.py` — non-workflows refs: `inference.core.entities.responses`
- `tests/workflows/unit_tests/core_steps/control_flow/test_inner_workflow_dispatch.py` — non-workflows refs: `inference.core.entities.requests.workflows`, `inference.core.env`
- `tests/workflows/unit_tests/core_steps/formatters/test_property_extraction.py` — non-workflows refs: `inference.core.entities.responses.inference`, `inference.core.env`
- `tests/workflows/unit_tests/core_steps/fusion/test_detections_classes_replacement.py` — non-workflows refs: `inference.core.entities.responses.inference`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_cog_vlm_deprecated.py` — non-workflows refs: `inference.core.exceptions`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_depth_estimation.py` — non-workflows refs: `inference.core.env`, `inference.core.utils.depth_encoding`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_gaze_deprecated.py` — non-workflows refs: `inference.core.exceptions`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_platform_client_headers.py` — non-workflows refs: `inference.core.roboflow_api`, `inference.core.utils.url_utils`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_sam_prompts.py` — non-workflows refs: `inference.core.entities.requests`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_segment_anything2.py` — non-workflows refs: `inference.core.entities.responses.inference`, `inference.core.roboflow_api`
- `tests/workflows/unit_tests/core_steps/sinks/test_twilio_sms_v2.py` — non-workflows refs: `inference.core.exceptions`
- `tests/workflows/unit_tests/core_steps/test_dependent_resources.py` — non-workflows refs: `inference.roboflow_workflows_plugin.loader`, `inference.roboflow_workflows_plugin.sinks.dataset_upload.v2`, `inference.roboflow_workflows_plugin.sinks.model_monitoring_inference_aggregator.v1` — plan Phase E table: split package-only vs server parity
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_representation_boundary.py` — non-workflows refs: `inference.core.entities.responses.inference`
- `tests/workflows/unit_tests/execution_engine/introspection/test_plugin_block_source.py` — non-workflows refs: `inference.enterprise`
- `tests/workflows/unit_tests/execution_engine/test_configuration_injection.py` — plan Phase E table: split package-only vs server parity
- `tests/workflows/unit_tests/execution_engine/test_step_error_handler_default.py` — non-workflows refs: `inference.core.exceptions`
- `tests/workflows/unit_tests/prototypes/test_image_codec.py` — non-workflows refs: `inference.core.utils`
- `tests/workflows/unit_tests/test_configuration.py` — plan Phase E table: split package-only vs server parity
- `tests/workflows/unit_tests/utils/test_image_encoding.py` — non-workflows refs: `inference.core.utils`
- `tests/workflows/unit_tests/utils/test_images.py` — non-workflows refs: `inference.core.utils.preprocess`
- `tests/workflows/unit_tests/utils/test_images_delegates.py` — non-workflows refs: `inference.core.utils`
- `tests/workflows/unit_tests/utils/test_text.py` — non-workflows refs: `inference.core.utils.environment`, `inference.core.utils.file_system`, `inference.core.utils.function`, `inference.core.utils.postprocess`, `inference.core.warnings`

## RETAIN (43 files)

- `tests/workflows/unit_tests/core_steps/common/test_run_in_parallel_context.py` — non-workflows refs: `inference.core.managers.model_load_collector`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_dataset_upload.py` — non-workflows refs: `inference.roboflow_workflows_plugin.sinks.dataset_upload.v1`, `inference.roboflow_workflows_plugin.sinks.dataset_upload.v2`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_model_monitoring.py` — non-workflows refs: `inference.roboflow_workflows_plugin.sinks.model_monitoring_inference_aggregator.v1`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_visual_search.py` — non-workflows refs: `inference.roboflow_workflows_plugin.integrations.visual_search.v1`
- `tests/workflows/unit_tests/core_steps/dependent_resources/test_visual_search_classifier.py` — non-workflows refs: `inference.roboflow_workflows_plugin.integrations.visual_search_classifier.v1`
- `tests/workflows/unit_tests/core_steps/integrations/roboflow/visual_search/test_helpers.py` — non-workflows refs: `inference.roboflow_workflows_plugin.integrations.visual_search.helpers`
- `tests/workflows/unit_tests/core_steps/integrations/roboflow/visual_search/test_v1.py` — non-workflows refs: `inference.roboflow_workflows_plugin.integrations.visual_search`, `inference.roboflow_workflows_plugin.integrations.visual_search.v1`
- `tests/workflows/unit_tests/core_steps/integrations/roboflow/visual_search_classifier/test_classification_annotations.py` — non-workflows refs: `inference.roboflow_workflows_plugin.integrations.visual_search_classifier.classification_annotations`
- `tests/workflows/unit_tests/core_steps/integrations/roboflow/visual_search_classifier/test_v1.py` — non-workflows refs: `inference.core.env`, `inference.core.utils.image_utils`, `inference.roboflow_workflows_plugin.integrations.visual_search_classifier`, `inference.roboflow_workflows_plugin.integrations.visual_search_classifier.v1`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_port_requests_match_legacy_clip.py` — non-workflows refs: `inference.core.entities.requests.clip`, `inference.core.entities.requests.perception_encoder`, `inference.core.interfaces.workflows_models_provider`, `inference.core.roboflow_api` — plan Phase E: server response/request parity
- `tests/workflows/unit_tests/core_steps/models/foundation/test_port_requests_match_legacy_lmm.py` — non-workflows refs: `inference.core.entities.requests.inference`, `inference.core.entities.requests.moondream2`, `inference.core.interfaces.workflows_models_provider` — plan Phase E: server response/request parity
- `tests/workflows/unit_tests/core_steps/models/foundation/test_port_requests_match_legacy_ocr.py` — non-workflows refs: `inference.core.entities.requests.doctr`, `inference.core.entities.requests.easy_ocr`, `inference.core.entities.requests.pp_ocr`, `inference.core.entities.requests.yolo_world`, `inference.core.interfaces.workflows_models_provider`, `inference.core.roboflow_api` — plan Phase E: server response/request parity
- `tests/workflows/unit_tests/core_steps/models/foundation/test_port_requests_match_legacy_sam.py` — non-workflows refs: `inference.core.entities.requests.sam2`, `inference.core.entities.requests.sam3`, `inference.core.entities.requests.sam3_3d`, `inference.core.interfaces.workflows_models_provider` — plan Phase E: server response/request parity
- `tests/workflows/unit_tests/core_steps/models/foundation/test_segment_anything2_video.py` — non-workflows refs: `inference.core.interfaces.workflows_execution_observer`, `inference.usage_tracking`, `inference.usage_tracking.collector`
- `tests/workflows/unit_tests/core_steps/models/foundation/test_segment_anything3_video.py` — non-workflows refs: `inference.core.interfaces.workflows_execution_observer`, `inference.usage_tracking`, `inference.usage_tracking.collector`
- `tests/workflows/unit_tests/core_steps/models/roboflow/action_recognition/test_action_recognition_entities.py` — non-workflows refs: `inference.core.entities`, `inference.core.entities.responses.action_recognition`, `inference.core.models.action_recognition`
- `tests/workflows/unit_tests/core_steps/models/roboflow/action_recognition/test_v1.py` — non-workflows refs: `inference.core.managers.base`
- `tests/workflows/unit_tests/core_steps/models/roboflow/instance_segmentation/test_v3_stream_pipeline.py` — non-workflows refs: `inference.core.entities.requests.inference`, `inference.core.entities.responses.inference`, `inference.core.interfaces.workflows_models_provider`
- `tests/workflows/unit_tests/core_steps/models/roboflow/test_port_requests_match_legacy.py` — non-workflows refs: `inference.core.entities.requests.inference`, `inference.core.interfaces.workflows_models_provider` — plan Phase E: server response/request parity
- `tests/workflows/unit_tests/core_steps/sinks/roboflow/asset_library_attributes/test_v1.py` — non-workflows refs: `inference.core.cache`, `inference.roboflow_workflows_plugin.sinks.asset_library_attributes`, `inference.roboflow_workflows_plugin.sinks.asset_library_attributes.v1`
- `tests/workflows/unit_tests/core_steps/sinks/roboflow/roboflow_dataset_upload/test_v1.py` — non-workflows refs: `inference.core.cache`, `inference.core.env`, `inference.roboflow_workflows_plugin.sinks.dataset_upload`, `inference.roboflow_workflows_plugin.sinks.dataset_upload.v1`, `inference.roboflow_workflows_plugin.sinks.dataset_upload.v1_tensor`
- `tests/workflows/unit_tests/core_steps/sinks/roboflow/roboflow_dataset_upload/test_v2.py` — non-workflows refs: `inference.core.cache`, `inference.core.env`, `inference.roboflow_workflows_plugin.sinks.dataset_upload`, `inference.roboflow_workflows_plugin.sinks.dataset_upload.v2`, `inference.roboflow_workflows_plugin.sinks.dataset_upload.v2_tensor`
- `tests/workflows/unit_tests/core_steps/sinks/roboflow/test_model_monitoring_inference_aggregator.py` — non-workflows refs: `inference.core.cache`, `inference.core.roboflow_api.send_inference_results_to_model_monitoring`, `inference.roboflow_workflows_plugin.sinks.model_monitoring_inference_aggregator.v1`, `inference.roboflow_workflows_plugin.sinks.model_monitoring_inference_aggregator.v1.get_roboflow_workspace`, `inference.roboflow_workflows_plugin.sinks.model_monitoring_inference_aggregator.v1.send_inference_results_to_model_monitoring`
- `tests/workflows/unit_tests/core_steps/sinks/roboflow/test_roboflow_custom_metadata.py` — non-workflows refs: `inference.core.cache`, `inference.roboflow_workflows_plugin.sinks.custom_metadata.v1`, `inference.roboflow_workflows_plugin.sinks.custom_metadata.v1.add_custom_metadata`, `inference.roboflow_workflows_plugin.sinks.custom_metadata.v1.add_custom_metadata_request`, `inference.roboflow_workflows_plugin.sinks.custom_metadata.v1.get_roboflow_workspace`
- `tests/workflows/unit_tests/core_steps/sinks/roboflow/vision_events/test_v1.py` — non-workflows refs: `inference.roboflow_workflows_plugin.sinks.vision_events`, `inference.roboflow_workflows_plugin.sinks.vision_events.v1`, `inference.roboflow_workflows_plugin.sinks.vision_events.v1._execute_local_event`, `inference.roboflow_workflows_plugin.sinks.vision_events.v1._execute_vision_event`, `inference.roboflow_workflows_plugin.sinks.vision_events.v1._send_event`, `inference.roboflow_workflows_plugin.sinks.vision_events.v1._send_local_event`, `inference.roboflow_workflows_plugin.sinks.vision_events.v1.requests.post`
- `tests/workflows/unit_tests/core_steps/sinks/roboflow/vision_events_bundle/test_v1.py` — non-workflows refs: `inference.core.env`, `inference.roboflow_workflows_plugin.sinks`, `inference.roboflow_workflows_plugin.sinks.vision_events.v1`, `inference.roboflow_workflows_plugin.sinks.vision_events.v1_tensor`, `inference.roboflow_workflows_plugin.sinks.vision_events_bundle.v1`, `inference.roboflow_workflows_plugin.sinks.vision_events_bundle.v1_tensor`
- `tests/workflows/unit_tests/core_steps/sinks/test_execution_policy.py` — non-workflows refs: `inference.roboflow_workflows_plugin.loader`
- `tests/workflows/unit_tests/core_steps/sinks/test_slack_notification.py` — non-workflows refs: `inference.core.cache`
- `tests/workflows/unit_tests/core_steps/sinks/test_twilio_sms_notification.py` — non-workflows refs: `inference.core.cache`
- `tests/workflows/unit_tests/core_steps/visualizations/test_icon_local_file_gate.py` — non-workflows refs: `inference.core.exceptions`, `inference.core.interfaces.workflows_image_codec`, `inference.core.utils`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_block_duration.py` — non-workflows refs: `inference.usage_tracking`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_block_usage_metering.py` — non-workflows refs: `inference.core.interfaces.workflows_execution_observer`, `inference.usage_tracking`, `inference.usage_tracking.block_execution`, `inference.usage_tracking.collector`
- `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_workspace_resolver.py` — non-workflows refs: `inference.core.exceptions`, `inference.core.interfaces.roboflow_platform_client`, `inference.core.roboflow_api`
- `tests/workflows/unit_tests/execution_engine/executor/test_runtime_input_assembler.py` — non-workflows refs: `inference.core.interfaces.workflows_image_codec`
- `tests/workflows/unit_tests/execution_engine/inner_workflow/test_spec_resolver_default.py` — non-workflows refs: `inference.core.interfaces.roboflow_platform_client`, `inference.core.roboflow_api`
- `tests/workflows/unit_tests/execution_engine/introspection/test_describe_outputs.py` — non-workflows refs: `inference.core.entities.requests.workflows`, `inference.core.interfaces.http.handlers.workflows`
- `tests/workflows/unit_tests/prototypes/test_cache.py` — non-workflows refs: `inference.core.cache.base`
- `tests/workflows/unit_tests/prototypes/test_models_provider.py` — non-workflows refs: `inference.core.interfaces.workflows_models_provider`, `inference.core.managers.base`, `inference.core.managers.decorators.base`
- `tests/workflows/unit_tests/prototypes/test_platform_client.py` — non-workflows refs: `inference.core.interfaces.roboflow_platform_client`, `inference.core.roboflow_api`, `inference.core.utils.requests`, `inference.core.utils.url_utils`, `inference.core.version`
- `tests/workflows/unit_tests/prototypes/test_platform_errors.py` — non-workflows refs: `inference.core`, `inference.core.interfaces.workflows_step_error_handlers`
- `tests/workflows/unit_tests/utils/test_action_recognition_copy.py` — non-workflows refs: `inference.core.entities`, `inference.core.managers.base`, `inference.core.managers.decorators.base`, `inference.core.models.action_recognition`
- `tests/workflows/unit_tests/utils/test_in_memory_cache.py` — non-workflows refs: `inference.core.cache.memory`
- `tests/workflows/unit_tests/utils/test_lru_cache.py` — non-workflows refs: `inference.core.cache.lru_cache`

## Reconciliation rules

- `MOVE`: unconditional move; rewrite `inference.core.workflows` -> `roboflow_workflows` in imports, strings, `mock.patch` targets, `importlib` calls; rewrite `tests.workflows.` intra-test imports to `tests.` (package-local).
- `MOVE_ENV`: same rewrite plus replace any `from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION` with the explicit `configured_workflows(tensor={"representation_enabled": ...})` fixture invocation. See §Tensor selection.
- `SPLIT`: leave the server-facing assertions in `tests/workflows/unit_tests/`; peel off pure block/engine assertions into a sibling file under `workflows/tests/unit_tests/` with the same subpath. Do not duplicate coverage; each assertion has one owner.
- `RETAIN`: no move. These tests exercise server adapters (`inference.core.interfaces.*`), the platform plugin (`inference.roboflow_workflows_plugin.*`), model manager/registry, cache, or legacy request parity. They keep running in the retained server pytest job.

## Tensor selection

Standalone package tests must not rely on `ENABLE_TENSOR_DATA_REPRESENTATION`. Configure `TensorConfiguration` explicitly before any workflows import. The `configured_workflows` fixture in `workflows/tests/conftest.py` is the sanctioned entry point. CI selects mode via job matrix, not environment.

## Nonempty-source guards

The migration is done and its scaffolding (`workflows/tests/migrate_unit_tests.py`) has been removed - it was a one-shot, not a durable check. The nonempty-source contract is now enforced at test time:

- `workflows/tests/unit_tests/core_steps/test_init_files.py` walks the installed package trees (`roboflow_workflows.core_steps`, `roboflow_workflows.enterprise_blocks`) and fails if either walk yields zero modules or a subpackage is missing `__init__.py`.
- The isolation probe's `import_everything` check (`workflows/scripts/workflows_isolation_probe.py`) asserts `pkgutil.walk_packages(roboflow_workflows)` is nonempty before importing each module.
- `workflows/tests/unit_tests/test_no_server_imports_in_package_tests.py` asserts the walked test corpus is nonempty and every file parses.
