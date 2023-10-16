from inference.core.managers.base import ModelManager


class StubLoaderManager(ModelManager):
    def add_model(self, model_id: str, api_key: str) -> None:
        """Adds a new model to the manager.

        Args:
            model_id (str): The identifier of the model.
            model (Model): The model instance.
        """
        if model_id in self._models:
            return
        model = self.model_registry.get_model(model_id, api_key)(
            model_id=model_id, api_key=api_key, load_weights=False
        )
        self._models[model_id] = model
