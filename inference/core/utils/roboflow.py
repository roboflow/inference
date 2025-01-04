from typing import Optional, Tuple, Union

from inference.core.entities.types import DatasetID, VersionID
from inference.core.exceptions import InvalidModelIDError


def get_model_id_chunks(
    model_id: str,
) -> Union[Tuple[DatasetID, VersionID], Tuple[str, None]]:
    """Parse a model ID into its components.

    Args:
        model_id (str): The model identifier, either in format "dataset/version"
                       or a plain string for the new model IDs

    Returns:
        Union[Tuple[DatasetID, VersionID], Tuple[str, None]]:
            For traditional IDs: (dataset_id, version_id)
            For new string IDs: (model_id, None)

    Raises:
        InvalidModelIDError: If traditional model ID format is invalid
    """
    if "/" not in model_id:
        # Handle new style model IDs that are just strings
        return model_id, None

    # Handle traditional dataset/version model IDs
    model_id_chunks = model_id.split("/")
    if len(model_id_chunks) != 2:
        raise InvalidModelIDError(
            f"Model ID: `{model_id}` is invalid. Expected format: 'dataset/version' or 'model_name'"
        )

    return model_id_chunks[0], model_id_chunks[1]
