from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.core.managers.entities import ModelDescription


class ServerVersionInfo(BaseModel):
    """Server version information.

    Attributes:
        name (str): Server name.
        version (str): Server version.
        uuid (str): Server UUID.
    """

    name: str = Field(examples=["Roboflow Inference Server"])
    version: str = Field(examples=["0.0.1"])
    uuid: str = Field(examples=["9c18c6f4-2266-41fb-8a0f-c12ae28f6fbe"])


class ModelDescriptionEntity(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(
        description="Identifier of the model", examples=["some-project/3"]
    )
    task_type: str = Field(
        description="Type of the task that the model performs",
        examples=["classification"],
    )
    batch_size: Optional[int] = Field(
        None,
        description="Batch size accepted by the model (if registered).",
    )
    input_height: Optional[int] = Field(
        None,
        description="Image input height accepted by the model (if registered).",
    )
    input_width: Optional[int] = Field(
        None,
        description="Image input width accepted by the model (if registered).",
    )
    vram_bytes: Optional[int] = Field(
        None,
        description="Estimated GPU VRAM consumed by this model in bytes (measured during load).",
    )

    @classmethod
    def from_model_description(
        cls, model_description: ModelDescription
    ) -> "ModelDescriptionEntity":
        # Convert string batch_size (indicating dynamic batching) to None
        batch_size = model_description.batch_size
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            batch_size = None
        return cls(
            model_id=model_description.model_id,
            task_type=model_description.task_type,
            batch_size=batch_size,
            input_height=model_description.input_height,
            input_width=model_description.input_width,
            vram_bytes=model_description.vram_bytes,
        )


class ModelsDescriptions(BaseModel):
    models: List[ModelDescriptionEntity] = Field(
        description="List of models that are loaded by model manager.",
    )
    total_vram_bytes: Optional[int] = Field(
        None,
        description="Total estimated VRAM consumed by all loaded models in bytes.",
    )
    cuda_memory_allocated: Optional[int] = Field(
        None,
        description="Current total CUDA memory allocated (from torch.cuda.memory_allocated).",
    )
    cuda_memory_reserved: Optional[int] = Field(
        None,
        description="Current total CUDA memory reserved by the allocator (from torch.cuda.memory_reserved).",
    )

    @classmethod
    def from_models_descriptions(
        cls, models_descriptions: List[ModelDescription]
    ) -> "ModelsDescriptions":
        model_entities = [
            ModelDescriptionEntity.from_model_description(
                model_description=model_description
            )
            for model_description in models_descriptions
        ]
        vram_values = [m.vram_bytes for m in model_entities if m.vram_bytes is not None]
        total_vram = sum(vram_values) if vram_values else None
        cuda_allocated, cuda_reserved = _get_cuda_memory_stats()
        return cls(
            models=model_entities,
            total_vram_bytes=total_vram,
            cuda_memory_allocated=cuda_allocated,
            cuda_memory_reserved=cuda_reserved,
        )


def _get_cuda_memory_stats() -> tuple:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    except ImportError:
        pass
    except Exception:
        pass
    return None, None
