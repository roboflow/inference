from __future__ import annotations

import json
from pathlib import Path

from profiling.memory.profiling_inputs import (
    DEFAULT_VLM_PROMPT,
    build_profiling_infer_kwargs,
    infer_kwargs_defaults_from_task_profile,
    resolve_profiling_method,
)
from profiling.memory.registry_profiles import get_task_inference_profile


def test_resolve_profiling_method_from_vlm_profile() -> None:
    method = resolve_profiling_method(
        architecture="paligemma",
        task_type="vlm",
        backend="torch",
    )

    assert method == "prompt"


def test_vlm_prompt_defaults_include_prompt_and_max_new_tokens(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 256}),
        encoding="utf-8",
    )

    profile = get_task_inference_profile("vlm_prompt")
    assert profile is not None

    defaults = infer_kwargs_defaults_from_task_profile(
        profile,
        package_dir=package_dir,
        profile_name="vlm_prompt",
    )

    assert defaults["prompt"] == DEFAULT_VLM_PROMPT
    assert defaults["max_new_tokens"] == 256


def test_build_profiling_infer_kwargs_merges_user_override(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()

    merged = build_profiling_infer_kwargs(
        architecture="grounding-dino",
        task_type="open-vocabulary-object-detection",
        backend="torch",
        package_dir=package_dir,
        user={"classes": ["cat"]},
    )

    assert merged["classes"] == ["cat"]
