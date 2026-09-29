from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, List, Optional, Sequence, Tuple

DISABLED_STAGE = "none"


class InvalidPipelineIdError(ValueError):
    pass


class PPOCRv6StructuredOCR:
    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def infer(self, images: Any, **kwargs) -> Tuple[List[str], List[Any]]:
        results = self._pipeline.infer(images, **kwargs)
        texts = [result.text for result in results]
        detections = [_structured_ocr_detections(result) for result in results]
        return texts, detections


def _structured_ocr_detections(result: Any) -> Any:
    import torch

    from inference_models.models.base.object_detection import Detections

    detections = result.detections
    if detections is None:
        return Detections(
            xyxy=torch.empty((0, 4)),
            class_id=torch.empty((0,), dtype=torch.long),
            confidence=torch.empty((0,)),
        )
    metadata = detections.bboxes_metadata or [{}] * len(detections)
    line_texts = result.line_texts
    return Detections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        image_metadata=detections.image_metadata,
        bboxes_metadata=[
            dict(meta, text=line_texts[index] if index < len(line_texts) else "")
            for index, meta in enumerate(metadata)
        ],
    )


@dataclass(frozen=True)
class PipelineFamily:
    pipeline_id: str
    stage_prefixes: Tuple[str, ...]
    stage_tokens: FrozenSet[str]
    facade: Callable[[Any], Any]
    task_type: str
    default_action: str


@dataclass(frozen=True)
class PipelineRequest:
    family: PipelineFamily
    stage_model_ids: Tuple[Optional[str], ...]


PIPELINE_FAMILIES: dict[str, PipelineFamily] = {
    "pp_ocr": PipelineFamily(
        pipeline_id="pp-ocrv6",
        stage_prefixes=("pp-ocrv6-det", "pp-ocrv6-rec"),
        stage_tokens=frozenset({DISABLED_STAGE, "tiny", "small", "medium"}),
        facade=PPOCRv6StructuredOCR,
        task_type="structured-ocr",
        default_action="infer",
    ),
}


def stage_tokens(family_key: str) -> FrozenSet[str]:
    return PIPELINE_FAMILIES[family_key].stage_tokens


def default_stage_tokens(family_key: str) -> Tuple[str, ...]:
    family = PIPELINE_FAMILIES[family_key]
    return tuple(
        stage_model_id.partition("/")[2]
        for stage_model_id in _default_stage_model_ids(family)
    )


def pipeline_model_id(family_key: str, tokens: Sequence[str]) -> str:
    return f"{family_key}/{'-'.join(tokens)}"


def resolve_pipeline_request(model_id: str) -> Optional[PipelineRequest]:
    family_key, _, variants = model_id.partition("/")
    family = PIPELINE_FAMILIES.get(family_key)
    if family is None:
        return None
    if not variants:
        stage_model_ids = _default_stage_model_ids(family)
    else:
        tokens = variants.split("-")
        if len(tokens) == 1:
            tokens = tokens * len(family.stage_prefixes)
        if len(tokens) != len(family.stage_prefixes) or any(
            token not in family.stage_tokens for token in tokens
        ):
            raise InvalidPipelineIdError(f"Invalid pipeline model id: {model_id}")
        stage_model_ids = tuple(
            None if token == DISABLED_STAGE else f"{prefix}/{token}"
            for prefix, token in zip(family.stage_prefixes, tokens)
        )
    if not any(stage_model_ids):
        raise InvalidPipelineIdError(
            f"Pipeline model id disables every stage: {model_id}"
        )
    return PipelineRequest(family=family, stage_model_ids=stage_model_ids)


def _default_stage_model_ids(family: PipelineFamily) -> Tuple[str, ...]:
    from inference_models.model_pipelines.auto_loaders.pipelines_registry import (
        get_default_pipeline_parameters,
    )

    defaults = get_default_pipeline_parameters(pipline_id=family.pipeline_id)
    if defaults is None or len(defaults) != len(family.stage_prefixes):
        raise ValueError(
            f"Pipeline {family.pipeline_id} must register exactly one default "
            f"model id per stage"
        )
    for prefix, default in zip(family.stage_prefixes, defaults):
        head, _, token = (
            default.partition("/") if isinstance(default, str) else ("", "", "")
        )
        if (
            head != prefix
            or token == DISABLED_STAGE
            or token not in family.stage_tokens
        ):
            raise ValueError(
                f"Default stage model {default!r} of pipeline {family.pipeline_id} "
                f"is not an enabled {prefix}/<token> model id"
            )
    return tuple(defaults)


def is_pipeline_model_id(model_id: str) -> bool:
    return model_id.partition("/")[0] in PIPELINE_FAMILIES


def pipeline_stage_model_ids(model_id: str) -> List[str]:
    request = resolve_pipeline_request(model_id)
    if request is None:
        return []
    return [stage for stage in request.stage_model_ids if stage is not None]


def _load_stage(stage: str, api_key: str, **load_kwargs) -> Any:
    from inference_model_manager.backends.base import attach_model_caches
    from inference_models.models.auto_loaders.core import AutoModel

    model = AutoModel.from_pretrained(stage, api_key=api_key, **load_kwargs)
    attach_model_caches(model)
    return model


def load_pipeline(request: PipelineRequest, api_key: str, **load_kwargs) -> Any:
    from inference_models.model_pipelines.auto_loaders.pipelines_registry import (
        resolve_pipeline_class,
    )

    models: List[Any] = []
    try:
        for stage in request.stage_model_ids:
            models.append(
                None if stage is None else _load_stage(stage, api_key, **load_kwargs)
            )
        pipeline = resolve_pipeline_class(pipline_id=request.family.pipeline_id)
        return request.family.facade(pipeline.with_models(models))
    except Exception:
        models.clear()
        raise


def load_model(model_id: str, api_key: str, **load_kwargs) -> Any:
    request = resolve_pipeline_request(model_id)
    if request is None:
        from inference_models.models.auto_loaders.core import AutoModel

        return AutoModel.from_pretrained(model_id, api_key=api_key, **load_kwargs)
    return load_pipeline(request, api_key, **load_kwargs)
