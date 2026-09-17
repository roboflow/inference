"""Composed execution-plan integration for Torch and ONNX object detection."""

import threading
from dataclasses import replace

import torch

from inference_models.errors import ModelRuntimeError
from inference_models.logger import LOGGER
from inference_models.models.common.streams import get_cuda_stream
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    OptimizationStage,
)
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.optimization.fallback_warnings import (
    FallbackWarningTracker,
)
from inference_models.models.optimization.runtime_components import (
    get_runtime_components,
)
from inference_models.models.rfdetr.optimization.catalog import (
    build_rfdetr_implementation_registry,
)
from inference_models.models.rfdetr.optimization.contracts import PreprocessRequest
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.selection import (
    resolve_preprocessor_for_model,
    resolve_preprocessor_for_request,
    resolve_preprocessor_runtime_fallback,
)
from inference_models.models.rfdetr.pre_processing import (
    resolve_rfdetr_preprocessor_max_workers,
)


class RFDetrBackendPath:
    """Own selected stage objects and request-local observability for one model."""

    def __init__(
        self,
        *,
        device,
        inference_config,
        backend,
        execution_plan=None,
        max_workers=None
    ):
        self.device = device
        self.config = inference_config
        self._local = threading.local()
        self._warnings = FallbackWarningTracker()
        requested = RFDetrExecutionPlan.resolve(execution_plan=execution_plan)
        self.registry = build_rfdetr_implementation_registry(
            device=device,
            preprocessor_max_workers=resolve_rfdetr_preprocessor_max_workers(
                max_workers
            ),
            backend=backend,
        )
        context = self.context()
        selections = {
            "preprocessor": resolve_preprocessor_for_model(
                registry=self.registry,
                requested_id=requested.preprocessor_id,
                context=context,
                image_pre_processing=self.config.image_pre_processing,
                network_input=self.config.network_input,
                allow_fallback=requested.allow_compatibility_fallback,
            )
        }
        for name, stage in (
            ("buffer_strategy", OptimizationStage.BUFFER_STRATEGY),
            ("scheduler", OptimizationStage.SCHEDULER),
            ("postprocessor", OptimizationStage.POSTPROCESS),
            ("engine_plugin", OptimizationStage.ENGINE_PLUGIN),
        ):
            selections[name] = self.registry.resolve_selection(
                stage=stage,
                requested_id=getattr(requested, name + "_id"),
                context=context,
                allow_fallback=requested.allow_compatibility_fallback,
            )
        self.selections = selections
        self.plan = replace(
            requested,
            **{name + "_id": item.effective_id for name, item in selections.items()}
        )
        for name, selection in selections.items():
            setattr(self, name, selection.implementation)
            if selection.used_fallback:
                LOGGER.warning(
                    "RF-DETR %s fallback requested=%s effective=%s reason=%s",
                    name,
                    selection.requested_id,
                    selection.effective_id,
                    selection.fallback_reason,
                )

    def context(self, stream=None):
        return ExecutionContext(
            device_kind="gpu" if self.device.type != "cpu" else "cpu",
            device=str(self.device),
            current_stream=stream,
            compute_capability=(
                torch.cuda.get_device_capability(self.device)
                if self.device.type == "cuda"
                else None
            ),
            runtime_components=get_runtime_components(),
        )

    def record(self, name, selection=None):
        if not hasattr(self._local, "last_execution"):
            self._local.last_execution = {}
        self._local.last_execution[name] = (
            selection or self.selections[name]
        ).to_dict()

    @property
    def runtime_metadata(self):
        return {
            "execution_plan": self.plan.to_dict(),
            **{
                name: selection.implementation.metadata.to_dict()
                for name, selection in self.selections.items()
            },
            "model_selection": {
                name: selection.to_dict() for name, selection in self.selections.items()
            },
            "last_execution": dict(getattr(self._local, "last_execution", {})),
        }

    def preprocess(
        self,
        images,
        *,
        input_color_format=None,
        pre_processing_overrides=None,
        image_size=None,
        independent_stage_execution=True
    ):
        stream = get_cuda_stream(device=self.device, purpose="pre-processing")
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.device))
        context = self.context(stream)
        request = PreprocessRequest(
            images=images,
            input_color_format=input_color_format,
            image_pre_processing=self.config.image_pre_processing,
            network_input=self.config.network_input,
            pre_processing_overrides=pre_processing_overrides,
            image_size_wh=image_size,
        )
        selection = resolve_preprocessor_for_request(
            registry=self.registry,
            implementation=self.preprocessor,
            request=request,
            context=context,
            allow_fallback=self.plan.allow_compatibility_fallback,
        )
        runtime_fallback = (
            self.plan.allow_compatibility_fallback
            and self.plan.allow_runtime_failure_fallback
        )
        try:
            selection = resolve_preprocessor_runtime_fallback(
                registry=self.registry,
                selection=selection,
                request=request,
                context=context,
                allow_fallback=runtime_fallback,
            )
            self.record("preprocessor", selection)
            try:
                result = selection.implementation.preprocess(request, context)
            except RecoverableStageExecutionError:
                if not runtime_fallback:
                    raise
                fallback = resolve_preprocessor_runtime_fallback(
                    registry=self.registry,
                    selection=selection,
                    request=request,
                    context=context,
                    allow_fallback=True,
                )
                if fallback.implementation is selection.implementation:
                    raise
                selection = fallback
                self.record("preprocessor", selection)
                result = selection.implementation.preprocess(request, context)
        except RecoverableStageExecutionError as error:
            raise ModelRuntimeError(
                message=str(error),
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            ) from error
        if selection.used_fallback and self._warnings.claim(
            stage=OptimizationStage.PREPROCESS,
            requested_id=selection.requested_id,
            effective_id=selection.effective_id,
            reason=selection.fallback_reason,
        ):
            LOGGER.warning(
                "RF-DETR request preprocessor fallback requested=%s effective=%s reason=%s",
                selection.requested_id,
                selection.effective_id,
                selection.fallback_reason,
            )
        result = replace(result, fallback_reason=selection.fallback_reason)
        engine_input = self.buffer_strategy.prepare_engine_input(result, context)
        self.record("buffer_strategy")
        self.record("scheduler")
        tensor = self.scheduler.finalize_preprocess(
            engine_input,
            context=context,
            independent_stage_execution=independent_stage_execution,
        )
        return tensor, result.metadata

    def forward(self, tensor, *, stream, operation):
        self.record("scheduler")
        self.record("engine_plugin")
        return self.scheduler.execute_engine(
            tensor,
            stream=stream,
            operation=lambda: self.engine_plugin.execute(operation),
        )

    def postprocess(self, operation):
        self.record("postprocessor")
        return self.postprocessor.execute(operation)


class RFDetrBackendPlanMixin:
    """Expose the same plan and metadata API as RF-DETR TensorRT."""

    @property
    def rfdetr_execution_plan(self):
        return self._execution_path.plan

    @property
    def optimization_runtime_metadata(self):
        return self._execution_path.runtime_metadata

    @property
    def preprocessor_implementation_id(self):
        return self._execution_path.preprocessor.metadata.implementation_id

    @property
    def preprocessor_implementation_metadata(self):
        return self._execution_path.preprocessor.metadata

    @property
    def buffer_strategy_implementation_id(self):
        return self._execution_path.buffer_strategy.metadata.implementation_id

    @property
    def buffer_strategy_implementation_metadata(self):
        return self._execution_path.buffer_strategy.metadata

    @property
    def scheduler_implementation_id(self):
        return self._execution_path.scheduler.metadata.implementation_id

    @property
    def scheduler_implementation_metadata(self):
        return self._execution_path.scheduler.metadata

    @property
    def postprocessor_implementation_id(self):
        return self._execution_path.postprocessor.metadata.implementation_id

    @property
    def postprocessor_implementation_metadata(self):
        return self._execution_path.postprocessor.metadata

    @property
    def engine_plugin_implementation_id(self):
        return self._execution_path.engine_plugin.metadata.implementation_id

    @property
    def engine_plugin_implementation_metadata(self):
        return self._execution_path.engine_plugin.metadata

    def infer(self, images, **kwargs):
        kwargs.pop("independent_stage_execution", None)
        tensor, metadata = self.pre_process(
            images, independent_stage_execution=False, **kwargs
        )
        results = self.forward(tensor, **kwargs)
        return self.post_process(results, metadata, **kwargs)
