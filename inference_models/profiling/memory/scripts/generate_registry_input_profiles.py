"""Generate registry_input_profiles.json from REGISTERED_MODELS.

Run from inference_models/:

    uv run python profiling/memory/scripts/generate_registry_input_profiles.py
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import (
    CLASSIFICATION_TASK,
    DEPTH_ESTIMATION_TASK,
    EMBEDDING_TASK,
    GAZE_DETECTION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    INTERACTIVE_INSTANCE_SEGMENTATION_TASK,
    KEYPOINT_DETECTION_TASK,
    MULTI_LABEL_CLASSIFICATION_TASK,
    OBJECT_DETECTION_TASK,
    OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
    REGISTERED_MODELS,
    SEMANTIC_SEGMENTATION_TASK,
    STRUCTURED_OCR_TASK,
    TEXT_ONLY_OCR_TASK,
    VLM_TASK,
    RegistryEntry,
)
from inference_models.utils.imports import LazyClass

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "registry_input_profiles.json"

MEMORY_IMPACT_LEVELS = ("low", "medium", "high", "critical")

# How profiling resolves tensor shapes from on-disk model packages, per backend.
# Spatial preprocessing generally follows inference_config.json for Roboflow exports;
# TRT runtime batch bounds are additionally (or exclusively) defined in trt_config.json.
BACKEND_PACKAGE_INPUT_PROFILES: Dict[str, Dict[str, Any]] = {
    "onnx": {
        "description": (
            "Roboflow ONNX packages: preprocessing and batch policy in inference_config.json; "
            "exported graph input shape in weights.onnx (should match static export)."
        ),
        "primary_artifacts": ["weights.onnx", "inference_config.json"],
        "dimensions": {
            "batch_size": {
                "static": {
                    "file": "inference_config.json",
                    "json_path": "forward_pass.static_batch_size",
                },
                "dynamic": {
                    "file": "inference_config.json",
                    "json_path": "forward_pass.max_dynamic_batch_size",
                    "profiling_note": (
                        "Profile at max_dynamic_batch_size for worst-case ORT allocator peaks."
                    ),
                },
            },
            "height": {
                "static": {
                    "file": "inference_config.json",
                    "json_path": "network_input.training_input_size.height",
                    "when": "network_input.dynamic_spatial_size_supported is false",
                },
                "dynamic": {
                    "file": "inference_config.json",
                    "json_path": "network_input",
                    "when": "network_input.dynamic_spatial_size_supported is true",
                    "profiling_note": (
                        "Sweep upper bound from dynamic_spatial_size_mode or package docs; "
                        "divisible-padding uses divisor constraints."
                    ),
                },
            },
            "width": {
                "static": {
                    "file": "inference_config.json",
                    "json_path": "network_input.training_input_size.width",
                    "when": "network_input.dynamic_spatial_size_supported is false",
                },
                "dynamic": {
                    "file": "inference_config.json",
                    "json_path": "network_input",
                    "when": "network_input.dynamic_spatial_size_supported is true",
                },
            },
        },
    },
    "torch-script": {
        "description": (
            "Roboflow TorchScript packages: same inference_config.json as ONNX; "
            "graph fixed to forward_pass.static_batch_size and training_input_size at export."
        ),
        "primary_artifacts": [
            "weights.torchscript",
            "inference_config.json",
            "class_names.txt",
        ],
        "dimensions": {
            "batch_size": {
                "static": {
                    "file": "inference_config.json",
                    "json_path": "forward_pass.static_batch_size",
                    "profiling_note": "TorchScript export requires static_batch_size.",
                },
                "dynamic": None,
            },
            "height": {
                "static": {
                    "file": "inference_config.json",
                    "json_path": "network_input.training_input_size.height",
                },
                "dynamic": None,
            },
            "width": {
                "static": {
                    "file": "inference_config.json",
                    "json_path": "network_input.training_input_size.width",
                },
                "dynamic": None,
            },
        },
    },
    "trt": {
        "description": (
            "Roboflow TensorRT packages: spatial preprocessing from inference_config.json; "
            "engine batch profile from trt_config.json (static or min/opt/max dynamic). "
            "Runtime spatial limits are baked into engine.plan / optimization profile."
        ),
        "primary_artifacts": [
            "engine.plan",
            "inference_config.json",
            "trt_config.json",
        ],
        "dimensions": {
            "batch_size": {
                "static": {
                    "file": "trt_config.json",
                    "json_path": "static_batch_size",
                },
                "dynamic": {
                    "file": "trt_config.json",
                    "json_paths": [
                        "dynamic_batch_size_min",
                        "dynamic_batch_size_opt",
                        "dynamic_batch_size_max",
                    ],
                    "profiling_note": (
                        "Profile at dynamic_batch_size_max for admission; "
                        "opt is the builder tuning point."
                    ),
                },
            },
            "height": {
                "static": {
                    "file": "inference_config.json",
                    "json_path": "network_input.training_input_size.height",
                    "profiling_note": (
                        "Preprocess target; must match engine input binding shape when engine is static."
                    ),
                },
                "dynamic": {
                    "file": "engine.plan",
                    "json_path": None,
                    "when": "TensorRT optimization profile exposes variable H",
                    "profiling_note": (
                        "Read min/opt/max from engine optimization profile via TRT APIs, "
                        "not from inference_config alone."
                    ),
                },
            },
            "width": {
                "static": {
                    "file": "inference_config.json",
                    "json_path": "network_input.training_input_size.width",
                },
                "dynamic": {
                    "file": "engine.plan",
                    "json_path": None,
                    "when": "TensorRT optimization profile exposes variable W",
                },
            },
        },
    },
    "torch": {
        "description": (
            "Native PyTorch packages: Roboflow exports use inference_config.json when present; "
            "library checkpoints (SAM, RF-DETR local) may use package-specific JSON instead."
        ),
        "primary_artifacts": ["weights.pt or package-specific checkpoint names"],
        "config_variants": {
            "roboflow": {
                "files": ["inference_config.json"],
                "dimensions": {
                    "batch_size": {
                        "static": {
                            "file": "inference_config.json",
                            "json_path": "forward_pass.static_batch_size",
                        },
                        "dynamic": {
                            "file": "inference_config.json",
                            "json_path": "forward_pass.max_dynamic_batch_size",
                        },
                    },
                    "height": {
                        "static": {
                            "file": "inference_config.json",
                            "json_path": "network_input.training_input_size.height",
                            "when": "network_input.dynamic_spatial_size_supported is false",
                        },
                        "dynamic": {
                            "file": "inference_config.json",
                            "json_path": "network_input",
                            "when": "network_input.dynamic_spatial_size_supported is true",
                        },
                    },
                    "width": {
                        "static": {
                            "file": "inference_config.json",
                            "json_path": "network_input.training_input_size.width",
                            "when": "network_input.dynamic_spatial_size_supported is false",
                        },
                        "dynamic": {
                            "file": "inference_config.json",
                            "json_path": "network_input",
                            "when": "network_input.dynamic_spatial_size_supported is true",
                        },
                    },
                },
            },
            "sam": {
                "files": ["sam_configuration.json", "model.pth"],
                "dimensions": {
                    "batch_size": {
                        "static": {
                            "file": "sam_configuration.json",
                            "json_path": "max_batch_size",
                            "fallback": "model constructor max_batch_size default",
                        },
                        "dynamic": None,
                    },
                    "height": {
                        "static": {
                            "file": "model.pth",
                            "json_path": None,
                            "profiling_note": "Derived from SAM image_encoder.img_size after load.",
                        },
                        "dynamic": None,
                    },
                    "width": {
                        "static": {
                            "file": "model.pth",
                            "json_path": None,
                            "profiling_note": "Same as height for square SAM encoder input.",
                        },
                        "dynamic": None,
                    },
                },
            },
            "sam3": {
                "files": ["weights.pt", "sam_configuration.json"],
                "dimensions": {
                    "batch_size": {
                        "static": {
                            "file": "constructor",
                            "json_path": "max_batch_size",
                        },
                        "dynamic": None,
                    },
                    "height": {
                        "static": {
                            "file": "constructor",
                            "json_path": "image_size",
                        },
                        "dynamic": None,
                    },
                    "width": {
                        "static": {
                            "file": "constructor",
                            "json_path": "image_size",
                        },
                        "dynamic": None,
                    },
                },
            },
        },
    },
    "hugging-face": {
        "description": (
            "Hugging Face layout packages: shapes and generation limits from transformers "
            "config artifacts (not inference_config.json)."
        ),
        "primary_artifacts": [
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "generation_config.json",
        ],
        "dimensions": {
            "batch_size": {
                "static": {
                    "file": "preprocessor_config.json",
                    "json_path": "size",
                    "profiling_note": (
                        "Often 1 for VLMs; multi-image prompts batch in processor."
                    ),
                },
                "dynamic": {
                    "file": "runtime",
                    "json_path": None,
                    "profiling_note": "Batch driven by number of images passed to prompt().",
                },
            },
            "height": {
                "static": {
                    "file": "preprocessor_config.json",
                    "json_paths": [
                        "size.shortest_edge",
                        "size.height",
                        "crop_size.height",
                    ],
                    "profiling_note": "First matching key wins per model family.",
                },
                "dynamic": {
                    "file": "preprocessor_config.json",
                    "json_path": None,
                    "when": "Model accepts variable resolution in processor",
                },
            },
            "width": {
                "static": {
                    "file": "preprocessor_config.json",
                    "json_paths": [
                        "size.longest_edge",
                        "size.width",
                        "crop_size.width",
                    ],
                },
                "dynamic": {
                    "file": "preprocessor_config.json",
                    "json_path": None,
                    "when": "Model accepts variable resolution in processor",
                },
            },
            "prompt_token_length": {
                "static": None,
                "dynamic": {
                    "file": "runtime",
                    "json_path": "tokenizer.encode(prompt)",
                    "profiling_note": "Sweep representative prompt strings for admission.",
                },
            },
            "max_new_tokens": {
                "static": {
                    "file": "generation_config.json",
                    "json_path": "max_new_tokens",
                    "fallback": "model-specific default constant in code",
                },
                "dynamic": {
                    "file": "method_kwargs",
                    "json_path": "max_new_tokens",
                    "profiling_note": "KV-cache peak scales with decode length cap.",
                },
            },
        },
    },
}

# Task-level inference API (method + non-spatial inputs). Spatial axes come from
# backend_package_input_profiles via each registry entry's backend field.
TASK_INFERENCE_PROFILES: Dict[str, Dict[str, Any]] = {
    "vision_infer": {
        "description": "Standard CV: infer(images, **kwargs).",
        "profiling_method": "infer",
        "inputs": [
            {
                "name": "images",
                "kind": "image_batch",
                "required": True,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            }
        ],
        "method_kwargs_memory_relevant": [
            {
                "name": "image_size",
                "memory_impact": "medium",
                "notes": "Runtime resize override when package allows dynamic spatial size.",
            }
        ],
    },
    "vlm_prompt": {
        "description": "HF VLM: prompt(images, prompt=..., max_new_tokens=...).",
        "profiling_method": "prompt",
        "inputs": [
            {
                "name": "images",
                "kind": "image_batch",
                "required": True,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            },
            {
                "name": "prompt",
                "kind": "text",
                "required": True,
                "memory_impact": "high",
                "axes": [
                    {
                        "name": "prompt_token_length",
                        "memory_impact": "high",
                        "resolution": "dynamic",
                        "backend_overrides": {
                            "hugging-face": {
                                "file": "runtime",
                                "json_path": "tokenizer.encode(prompt)",
                            }
                        },
                    }
                ],
            },
        ],
        "method_kwargs_memory_relevant": [
            {
                "name": "max_new_tokens",
                "memory_impact": "critical",
                "resolution": {
                    "static": {
                        "file": "generation_config.json",
                        "json_path": "max_new_tokens",
                    },
                    "dynamic": {
                        "file": "method_kwargs",
                        "json_path": "max_new_tokens",
                    },
                },
            },
            {
                "name": "num_beams",
                "memory_impact": "medium",
            },
            {
                "name": "images_to_single_prompt",
                "memory_impact": "medium",
            },
        ],
    },
    "vlm_florence2": {
        "extends": "vlm_prompt",
        "description": "Florence-2 prompt() with task-specific decoder head.",
        "method_kwargs_memory_relevant": [
            {
                "name": "task",
                "memory_impact": "medium",
            }
        ],
    },
    "vlm_moondream2": {
        "description": "Moondream2: detect / caption / query (not prompt).",
        "profiling_method": "detect",
        "alternate_methods": ["caption", "query"],
        "inputs": [
            {
                "name": "images",
                "kind": "image_batch",
                "required": True,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            },
            {
                "name": "classes",
                "kind": "text_list",
                "required": True,
                "memory_impact": "high",
                "axes": [
                    {
                        "name": "num_classes",
                        "memory_impact": "high",
                        "resolution": "dynamic",
                        "profiling_note": (
                            "Package profiling supplies class list; detect() runs once per class."
                        ),
                    }
                ],
            },
        ],
        "method_kwargs_memory_relevant": [
            {
                "name": "max_new_tokens",
                "memory_impact": "critical",
                "resolution": {
                    "dynamic": {
                        "file": "method_kwargs",
                        "json_path": "settings.max_tokens",
                    }
                },
            },
            {
                "name": "length",
                "memory_impact": "medium",
                "notes": "caption() only.",
            },
        ],
    },
    "embedding": {
        "description": "embed_images / embed_text.",
        "profiling_method": "embed_images",
        "alternate_methods": ["embed_text", "compare_embeddings"],
        "inputs": [
            {
                "name": "images",
                "kind": "image_batch",
                "required": False,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            },
            {
                "name": "texts",
                "kind": "text_list",
                "required": False,
                "memory_impact": "high",
                "axes": [
                    {
                        "name": "batch_size",
                        "memory_impact": "high",
                        "resolution": "dynamic",
                    },
                    {
                        "name": "sequence_length",
                        "memory_impact": "high",
                        "resolution": "dynamic",
                        "backend_overrides": {
                            "hugging-face": {
                                "file": "runtime",
                                "json_path": "tokenizer.encode(texts)",
                            },
                            "torch": {
                                "file": "runtime",
                                "json_path": "tokenizer.encode(texts)",
                            },
                        },
                    },
                ],
            },
        ],
    },
    "open_vocabulary_detection": {
        "description": "infer(images, classes=...).",
        "profiling_method": "infer",
        "inputs": [
            {
                "name": "images",
                "kind": "image_batch",
                "required": True,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            },
            {
                "name": "classes",
                "kind": "text_list",
                "required": True,
                "memory_impact": "high",
                "axes": [
                    {
                        "name": "num_classes",
                        "memory_impact": "high",
                        "resolution": "dynamic",
                        "profiling_note": (
                            "Profiling package should include representative class queries."
                        ),
                    }
                ],
            },
        ],
        "method_kwargs_memory_relevant": [
            {
                "name": "max_detections",
                "memory_impact": "low",
            }
        ],
    },
    "interactive_sam": {
        "description": "embed_images then segment_images with geometry prompts.",
        "profiling_workflow": ["embed_images", "segment_images"],
        "profiling_method": "segment_images",
        "inputs": [
            {
                "name": "images",
                "kind": "image_batch",
                "required": True,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            },
            {
                "name": "point_coordinates",
                "kind": "prompt_geometry",
                "required": False,
                "memory_impact": "medium",
                "axes": [
                    {
                        "name": "num_points",
                        "memory_impact": "medium",
                        "resolution": "dynamic",
                    }
                ],
            },
            {
                "name": "boxes",
                "kind": "prompt_geometry",
                "required": False,
                "memory_impact": "medium",
                "axes": [
                    {
                        "name": "num_boxes",
                        "memory_impact": "medium",
                        "resolution": "dynamic",
                    }
                ],
            },
        ],
        "method_kwargs_memory_relevant": [
            {
                "name": "multi_mask_output",
                "memory_impact": "medium",
            }
        ],
    },
    "sam3_text_segmentation": {
        "description": "segment_with_text_prompts(images, prompts).",
        "profiling_method": "segment_with_text_prompts",
        "inputs": [
            {
                "name": "images",
                "kind": "image_batch",
                "required": True,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            },
            {
                "name": "prompts",
                "kind": "structured_prompt_list",
                "required": True,
                "memory_impact": "high",
                "axes": [
                    {
                        "name": "num_text_queries",
                        "memory_impact": "high",
                        "resolution": "dynamic",
                    }
                ],
            },
        ],
    },
    "video_tracking": {
        "description": "HF streaming video: prompt / track with session state.",
        "profiling_workflow": ["prompt", "track"],
        "profiling_method": "prompt",
        "alternate_methods": ["track"],
        "inputs": [
            {
                "name": "image",
                "kind": "single_image",
                "required": True,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            },
            {
                "name": "text",
                "kind": "text",
                "required": False,
                "memory_impact": "medium",
                "axes": [
                    {
                        "name": "prompt_token_length",
                        "memory_impact": "medium",
                        "resolution": "dynamic",
                    }
                ],
            },
            {
                "name": "state_dict",
                "kind": "session_state",
                "required": False,
                "memory_impact": "critical",
                "axes": [
                    {
                        "name": "session_frames",
                        "memory_impact": "critical",
                        "resolution": "dynamic",
                        "profiling_note": "Sweep track iterations for worst-case session memory.",
                    }
                ],
            },
        ],
        "method_kwargs_memory_relevant": [
            {
                "name": "bboxes",
                "memory_impact": "medium",
            },
            {
                "name": "clear_old_prompts",
                "memory_impact": "medium",
            },
        ],
    },
    "ocr_generative": {
        "description": "TrOCR infer: internal generate() on pixel batch.",
        "profiling_method": "infer",
        "inputs": [
            {
                "name": "images",
                "kind": "image_batch",
                "required": True,
                "memory_impact": "high",
                "shape_from_backend_package": True,
            }
        ],
        "method_kwargs_memory_relevant": [
            {
                "name": "max_new_tokens",
                "memory_impact": "critical",
                "resolution": {
                    "static": {
                        "file": "generation_config.json",
                        "json_path": "max_new_tokens",
                    },
                    "dynamic": {
                        "file": "generate_defaults",
                        "json_path": "model.generate",
                    },
                },
            }
        ],
    },
}

ARCHITECTURE_OVERRIDES: Dict[str, str] = {
    "florence-2": "vlm_florence2",
    "moondream2": "vlm_moondream2",
    "glm-ocr": "vlm_prompt",
    "sam2video": "video_tracking",
    "segment-anything-2-rt": "video_tracking",
    "sam": "interactive_sam",
    "sam2": "interactive_sam",
    "sam3": "interactive_sam",
}

TASK_TYPE_TO_PROFILE: Dict[Optional[str], str] = {
    OBJECT_DETECTION_TASK: "vision_infer",
    INSTANCE_SEGMENTATION_TASK: "vision_infer",
    SEMANTIC_SEGMENTATION_TASK: "vision_infer",
    KEYPOINT_DETECTION_TASK: "vision_infer",
    CLASSIFICATION_TASK: "vision_infer",
    MULTI_LABEL_CLASSIFICATION_TASK: "vision_infer",
    DEPTH_ESTIMATION_TASK: "vision_infer",
    STRUCTURED_OCR_TASK: "vision_infer",
    GAZE_DETECTION_TASK: "vision_infer",
    VLM_TASK: "vlm_prompt",
    EMBEDDING_TASK: "embedding",
    OPEN_VOCABULARY_OBJECT_DETECTION_TASK: "open_vocabulary_detection",
    INTERACTIVE_INSTANCE_SEGMENTATION_TASK: "interactive_sam",
    TEXT_ONLY_OCR_TASK: "ocr_generative",
}

TORCH_PACKAGE_VARIANT: Dict[str, str] = {
    "sam": "sam",
    "sam2": "sam",
    "sam3": "sam3",
}


def _drop_none_memory_impact(obj: Any) -> Any:
    """Recursively remove dict/list entries whose memory_impact is 'none'."""
    if isinstance(obj, dict):
        if obj.get("memory_impact") == "none":
            return None

        cleaned: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "method_kwargs_memory_relevant" and isinstance(value, list):
                filtered = []
                for item in value:
                    pruned = _drop_none_memory_impact(item)
                    if pruned is not None:
                        filtered.append(pruned)
                if filtered:
                    cleaned[key] = filtered
                continue

            if key == "infer_kwargs_memory_relevant" and isinstance(value, list):
                filtered = []
                for item in value:
                    pruned = _drop_none_memory_impact(item)
                    if pruned is not None:
                        filtered.append(pruned)
                if filtered:
                    cleaned[key] = filtered
                continue

            pruned = _drop_none_memory_impact(value)
            if pruned is None:
                continue
            if isinstance(pruned, list) and len(pruned) == 0:
                continue

            cleaned[key] = pruned

        return cleaned if cleaned else None

    if isinstance(obj, list):
        filtered = []
        for item in obj:
            pruned = _drop_none_memory_impact(item)
            if pruned is not None:
                filtered.append(pruned)

        return filtered

    return obj


def _resolve_lazy(entry: Union[LazyClass, RegistryEntry]) -> LazyClass:
    lazy = entry.model_class if isinstance(entry, RegistryEntry) else entry
    assert isinstance(lazy, LazyClass)

    return lazy


def _resolve_task_profile_id(
    architecture: str,
    task_type: Optional[str],
) -> str:
    if architecture in ARCHITECTURE_OVERRIDES:
        override = ARCHITECTURE_OVERRIDES[architecture]
        if architecture == "sam3" and task_type == INSTANCE_SEGMENTATION_TASK:
            return "sam3_text_segmentation"

        return override

    return TASK_TYPE_TO_PROFILE[task_type]


def _torch_package_variant(architecture: str) -> str:
    return TORCH_PACKAGE_VARIANT.get(architecture, "roboflow")


def _build_registry_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    for key, entry in sorted(REGISTERED_MODELS.items()):
        architecture, task_type, backend = key
        lazy = _resolve_lazy(entry)
        backend_value = backend.value

        registry_entry: Dict[str, Any] = {
            "architecture": architecture,
            "task_type": task_type,
            "backend": backend_value,
            "module_name": lazy._module_name,
            "class_name": lazy._class_name,
            "task_inference_profile": _resolve_task_profile_id(
                architecture=architecture,
                task_type=task_type,
            ),
            "package_input_backend": backend_value,
        }

        if backend == BackendType.TORCH:
            registry_entry["torch_package_variant"] = _torch_package_variant(
                architecture=architecture,
            )

        if isinstance(entry, RegistryEntry):
            if entry.required_model_features:
                registry_entry["required_model_features"] = sorted(
                    entry.required_model_features
                )
            if entry.supported_model_features:
                registry_entry["supported_model_features"] = sorted(
                    entry.supported_model_features
                )

        entries.append(registry_entry)

    return entries


def _expand_task_profiles() -> Dict[str, Dict[str, Any]]:
    expanded: Dict[str, Dict[str, Any]] = {}

    for profile_id, profile in TASK_INFERENCE_PROFILES.items():
        if "extends" in profile:
            base = deepcopy(expanded[profile["extends"]])
            overlay = {k: v for k, v in profile.items() if k != "extends"}
            if "method_kwargs_memory_relevant" in overlay:
                base_kwargs = base.get("method_kwargs_memory_relevant", [])
                overlay_kwargs = overlay.pop("method_kwargs_memory_relevant")
                base["method_kwargs_memory_relevant"] = base_kwargs + overlay_kwargs
            base.update(overlay)
            expanded[profile_id] = base
        else:
            expanded[profile_id] = deepcopy(profile)

    return expanded


def main() -> None:
    task_profiles = _drop_none_memory_impact(_expand_task_profiles())
    backend_profiles = _drop_none_memory_impact(
        deepcopy(BACKEND_PACKAGE_INPUT_PROFILES)
    )

    document: Dict[str, Any] = {
        "schema_version": "2.0",
        "description": (
            "Memory-relevant inference inputs for REGISTERED_MODELS. "
            "Spatial/batch shapes are resolved from model packages per backend "
            "(see backend_package_input_profiles). Task profiles define API methods "
            "and non-spatial inputs (prompts, classes, decode limits)."
        ),
        "memory_impact_levels": list(MEMORY_IMPACT_LEVELS),
        "resolution_modes": {
            "static": (
                "Single fixed value from package artifacts; use for profiling peaks "
                "when the runtime path is fixed-shape."
            ),
            "dynamic": (
                "Variable at inference time or bounded by min/opt/max; profile at "
                "the declared maximum for admission."
            ),
        },
        "input_kinds": {
            "image_batch": "RGB/BGR images; H×W×batch drives activations.",
            "single_image": "One video frame.",
            "text": "Prompt string.",
            "text_list": "Class names or text batch.",
            "structured_prompt_list": "SAM3 prompt dicts.",
            "prompt_geometry": "Points or boxes for SAM.",
            "session_state": "Video tracker state across frames.",
        },
        "backend_package_input_profiles": backend_profiles,
        "task_inference_profiles": task_profiles,
        "registry_entries": _build_registry_entries(),
    }

    serialized = json.dumps(document, indent=2)
    serialized = f"{serialized}\n"

    OUTPUT_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {len(document['registry_entries'])} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
