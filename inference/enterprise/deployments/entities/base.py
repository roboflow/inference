from abc import ABC, abstractmethod


class GraphNone(ABC):
    @abstractmethod
    def get_type(self) -> str:
        pass
