from abc import ABC, abstractmethod
from typing import Any, List, Optional


class Backend(ABC):
    """Handle to a single loaded model.

    One instance per model. Loading happens in __init__ (blocks until ready).
    ModelManager owns a dict of these and routes calls by model_id.
    """

    @property
    @abstractmethod
    def class_names(self) -> Optional[List[str]]:
        """Class names for the loaded model, if available."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all resources held by this model."""
        ...

    @abstractmethod
    async def infer_async(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    def infer_sync(self, *args, **kwargs) -> Any: ...
