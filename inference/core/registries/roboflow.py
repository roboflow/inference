import json
import os

import requests

from inference.core.env import API_BASE_URL, MODEL_CACHE_DIR
from inference.core.exceptions import DatasetLoadError, WorkspaceLoadError
from inference.core.logger import logger
from inference.core.models.base import Model
from inference.core.registries.base import ModelRegistry
from inference.core.utils.url_utils import ApiUrl

MODEL_TYPE_DEFAULTS = {
    "object-detection": "yolov5v2s",
    "instance-segmentation": "yolact",
    "classification": "vit",
}


class RoboflowModelRegistry(ModelRegistry):
    """A Roboflow-specific model registry which gets the model type using the model id,
    then returns a model class based on the model type.
    """

    def get_model(self, model_id: str, api_key: str) -> Model:
        """Returns the model class based on the given model id and API key.

        Args:
            model_id (str): The ID of the model to be retrieved.
            api_key (str): The API key used to authenticate.

        Returns:
            Model: The model class corresponding to the given model ID and type.

        Raises:
            DatasetLoadError: If the model type is not supported or found.
        """
        model_type = get_model_type(model_id, api_key)
        if model_type not in self.registry_dict:
            raise DatasetLoadError(f"Model type not supported: {model_type}")
        return self.registry_dict[model_type]


def get_model_type(model_id: str, api_key: str) -> str:
    """Retrieves the model type based on the given model ID and API key.

    Args:
        model_id (str): The ID of the model.
        api_key (str): The API key used to authenticate.

    Returns:
        tuple: The project task type and the model type.

    Raises:
        WorkspaceLoadError: If the workspace could not be loaded or if the API key is invalid.
        DatasetLoadError: If the dataset could not be loaded due to invalid ID, workspace ID or version ID.
    """
    dataset_id = model_id.split("/")[0]
    version_id = model_id.split("/")[1]

    if dataset_id == "clip":
        return "embed", "clip"
    elif dataset_id == "sam":
        return "embed", "sam"
    elif dataset_id == "gaze":
        return "gaze", "l2cs"

    cache_dir = os.path.join(MODEL_CACHE_DIR, dataset_id, version_id)
    model_type_cache_path = os.path.join(cache_dir, "model_type.json")
    if os.path.exists(model_type_cache_path):
        with open(model_type_cache_path) as f:
            cache_data = json.load(f)
            project_task_type = cache_data["project_task_type"]
            model_type = cache_data["model_type"]
        return project_task_type, model_type

    api_url = ApiUrl("/".join([API_BASE_URL, f"?api_key={api_key}"]))
    api_key_info = requests.get(api_url)
    try:
        api_key_info.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(e)
        raise WorkspaceLoadError("Could not load workspace, check your API key")

    workspace_id = api_key_info.json().get("workspace")

    if workspace_id is None:
        raise WorkspaceLoadError(f"Empty workspace, check your API key")

    api_url = ApiUrl(
        "/".join(
            [API_BASE_URL, workspace_id, dataset_id, f"?api_key={api_key}&nocache=true"]
        )
    )
    dataset_info = requests.get(api_url)
    api_url = ApiUrl(
        "/".join(
            [
                API_BASE_URL,
                workspace_id,
                dataset_id,
                version_id,
                f"?api_key={api_key}&nocache=true",
            ]
        )
    )
    version_info = requests.get(api_url)
    try:
        dataset_info.raise_for_status()
        version_info.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(e)
        raise DatasetLoadError(
            f"Could not load dataset with ID {dataset_id} and workspace ID {workspace_id} for version ID {version_id}, check your API key"
        )

    # For legacy support we default to object-detection since some older models don't have a type
    project_task_type = dataset_info.json()["project"].get("type", "object-detection")
    model_type = version_info.json()["version"].get(
        "modelType", MODEL_TYPE_DEFAULTS[project_task_type]
    )

    os.makedirs(cache_dir, exist_ok=True)
    with open(model_type_cache_path, "w") as f:
        json.dump({"project_task_type": project_task_type, "model_type": model_type}, f)

    return project_task_type, model_type
