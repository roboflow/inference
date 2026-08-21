from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from inference_models.models.auto_loaders.entities import AnyModel


@dataclass(frozen=True)
class AccessIdentifiers:
    model_id: str
    package_id: str
    api_key: Optional[str]


class ModelAccessManager(ABC):

    @abstractmethod
    def on_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> None:
        pass

    @abstractmethod
    def on_model_package_access_granted(
        self, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    @abstractmethod
    def on_file_created(
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
    def on_symlink_deleted(self, link_name: str) -> None:
        pass

    @abstractmethod
    def on_file_deleted(self, file_path: str) -> None:
        pass

    @abstractmethod
    def on_directory_deleted(self, dir_path: str) -> None:
        pass

    @abstractmethod
    def is_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> bool:
        pass

    @abstractmethod
    def is_model_package_access_granted(
        self, model_id: str, package_id: str, api_key: Optional[str]
    ) -> bool:
        pass

    @abstractmethod
    def retrieve_model_instance(
        self,
        model_id: str,
        package_id: Optional[str],
        api_key: Optional[str],
        loading_parameter_digest: Optional[str],
    ) -> Optional[AnyModel]:
        pass

    @abstractmethod
    def on_model_loaded(
        self,
        model: AnyModel,
        access_identifiers: AccessIdentifiers,
        model_storage_path: str,
    ) -> None:
        pass

    @abstractmethod
    def on_model_alias_discovered(self, alias: str, model_id: str) -> None:
        pass

    @abstractmethod
    def on_model_dependency_discovered(
        self,
        base_model_id: str,
        base_model_package_id: Optional[str],
        dependent_model_id: str,
    ) -> None:
        pass


class LiberalModelAccessManager(ModelAccessManager):

    def on_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> None:
        pass

    def on_model_package_access_granted(
        self, access_identifiers: AccessIdentifiers
    ) -> None:
        pass

    def on_file_created(
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

    def on_symlink_deleted(self, link_name: str) -> None:
        pass

    def on_file_deleted(self, file_path: str) -> None:
        pass

    def on_directory_deleted(self, dir_path: str) -> None:
        pass

    def is_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> bool:
        return False

    def is_model_package_access_granted(
        self, model_id: str, package_id: str, api_key: Optional[str]
    ) -> bool:
        return True

    def retrieve_model_instance(
        self,
        model_id: str,
        package_id: Optional[str],
        api_key: Optional[str],
        loading_parameter_digest: Optional[str],
    ) -> Optional[AnyModel]:
        return None

    def on_model_loaded(
        self,
        model: AnyModel,
        access_identifiers: AccessIdentifiers,
        model_storage_path: str,
    ) -> None:
        pass

    def on_model_alias_discovered(self, alias: str, model_id: str) -> None:
        pass

    def on_model_dependency_discovered(
        self,
        base_model_id: str,
        base_model_package_id: Optional[str],
        dependent_model_id: str,
    ) -> None:
        pass
