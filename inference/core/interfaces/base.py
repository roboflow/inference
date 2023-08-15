from inference.core.managers.base import ModelManager


class BaseInterface:
    """Base interface class which accepts a model manager on initialization"""

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager
