from typing import Optional, Tuple, Union

from inference.core.entities.types import DatasetID, ModelID, VersionID
from inference.core.exceptions import InvalidModelIDError


def get_model_id_chunks(model_id: str) -> Tuple[Union[DatasetID, ModelID], Optional[VersionID]]:
    model_id_chunks = model_id.split("/")
    if len(model_id_chunks) != 2:
        raise InvalidModelIDError(f"Model ID: `{model_id}` is invalid.")
    try:
        dataset_id, version_id = model_id_chunks[0], int(model_id_chunks[1])
        return dataset_id, str(version_id)
    except Exception:
        return model_id, None
