from typing import Optional, Tuple, Union

from inference.core.entities.types import DatasetID, ModelID, VersionID
from inference.core.exceptions import InvalidModelIDError


def get_model_id_chunks(
    model_id: str,
) -> Tuple[Union[DatasetID, ModelID], Optional[VersionID]]:
    model_id_chunks = model_id.split("/")
    if len(model_id_chunks) != 2:
        raise InvalidModelIDError(f"Model ID: `{model_id}` is invalid.")
    dataset_id, version_id = model_id_chunks[0], model_id_chunks[1]
    if dataset_id.lower() in {
        "clip",
        "doctr",
        "doctr_rec",
        "doctr_det",
        "gaze",
        "grounding_dino",
        "sam",
        "sam2",
        "owlv2",
        "trocr",
        "yolo_world",
    }:
        return dataset_id, version_id
    try:
        return dataset_id, str(int(version_id))
    except Exception:
        return model_id, None
