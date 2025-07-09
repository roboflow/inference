from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AccessIdentifiers:
    model_id: str
    package_id: str
    api_key: Optional[str]


class ModelStorageManager(ABC):

    @abstractmethod
    def on_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> None:
        pass

    @abstractmethod
    def on_model_package_access_granted(
        self, dir_path: str, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    @abstractmethod
    def on_file_allocated(
        self, file_path: str, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    @abstractmethod
    def on_file_renamed(
        self, old_path: str, new_path: str, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    @abstractmethod
    def on_symlink_created(
        self, target_path: str, link_name: str, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    @abstractmethod
    def is_model_access_forbidden(
        self, model_name_or_path: str, api_key: Optional[str]
    ) -> bool:
        pass

    @abstractmethod
    def is_model_package_access_granted(
        self, model_id: str, package_id: str, api_key: Optional[str]
    ) -> bool:
        pass

    @abstractmethod
    def is_directory_access_granted(
        self, dir_path: str, access_identifiers: AccessIdentifiers
    ) -> bool:
        pass

    @abstractmethod
    def is_file_access_granted(
        self, file_path: str, access_identifiers: AccessIdentifiers
    ) -> bool:
        pass


class LiberalModelStorageManager(ModelStorageManager):

    def on_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> None:
        pass

    def on_model_package_access_granted(
        self, dir_path: str, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    def on_file_allocated(
        self, file_path: str, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    def on_file_renamed(
        self, old_path: str, new_path: str, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    def on_symlink_created(
        self, target_path: str, link_name: str, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    def is_model_access_forbidden(
        self, model_name_or_path: str, api_key: Optional[str]
    ) -> bool:
        return False

    def is_model_package_access_granted(
        self, model_id: str, package_id: str, api_key: Optional[str]
    ) -> bool:
        return True

    def is_directory_access_granted(
        self, dir_path: str, access_identifiers: AccessIdentifiers
    ) -> bool:
        return True

    def is_file_access_granted(
        self, file_path: str, access_identifiers: AccessIdentifiers
    ) -> bool:
        return True
