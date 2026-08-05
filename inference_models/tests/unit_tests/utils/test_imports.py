import importlib.machinery
import os.path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from inference_models.utils.imports import LazyClass, import_class_from_file


def test_lazy_class_import_from_module_raising_error() -> None:
    # given
    lazy_class = LazyClass(
        module_name="tests.unit_tests.utils.lazy_class_test_package.broken",
        class_name="MyClass",
    )

    # when
    with pytest.raises(RuntimeError) as error:
        _ = lazy_class.resolve()

    # then
    assert "This error should be raised when module is accessed" in str(
        error.value
    ), "Expected the exact error to be raised"


def test_lazy_class_importing_non_existing_module() -> None:
    # given
    lazy_class = LazyClass(
        module_name="non.existing",
        class_name="MyClass",
    )

    # when
    with pytest.raises(ModuleNotFoundError):
        _ = lazy_class.resolve()


def test_lazy_class_importing_non_existing_class_from_existing_module() -> None:
    # given
    lazy_class = LazyClass(
        module_name="tests.unit_tests.utils.lazy_class_test_package.valid",
        class_name="NonExistingClass",
    )

    # when
    with pytest.raises(AttributeError):
        _ = lazy_class.resolve()


def test_lazy_class_importing_existing_class_from_existing_module() -> None:
    # given
    lazy_class = LazyClass(
        module_name="tests.unit_tests.utils.lazy_class_test_package.valid",
        class_name="MyClass",
    )

    # when
    my_class = lazy_class.resolve()
    instance = my_class()

    # then
    assert (
        instance.hello() == "hello"
    ), "Expected fixed method response as confirmation of correct import"


def test_import_class_from_file_when_valid_module_path_provided(
    existing_module_path: str,
) -> None:
    # when
    my_class = import_class_from_file(
        file_path=existing_module_path, class_name="MyClass"
    )
    instance = my_class()

    # then
    assert (
        instance.hello() == "hello"
    ), "Expected fixed method response as confirmation of correct import"


def test_import_class_from_file_does_not_write_bytecode_into_package_dir(
    empty_local_dir: str,
) -> None:
    package_dir = os.path.join(empty_local_dir, "model_package")
    os.makedirs(package_dir)
    module_path = os.path.join(package_dir, "hf_moondream.py")
    helper_path = os.path.join(package_dir, "helper.py")
    with open(helper_path, "w") as helper_file:
        helper_file.write("class Helper:\n    pass\n")
    with open(module_path, "w") as module_file:
        module_file.write(
            "from .helper import Helper\n\nclass HfMoondream(Helper):\n    pass\n"
        )

    model_class = import_class_from_file(
        file_path=module_path, class_name="HfMoondream"
    )

    assert model_class.__name__ == "HfMoondream"
    assert not os.path.exists(os.path.join(package_dir, "__pycache__"))


@pytest.mark.parametrize("initial_value", [False, True])
def test_import_class_from_file_restores_bytecode_setting(
    empty_local_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    initial_value: bool,
) -> None:
    module_path = os.path.join(empty_local_dir, "valid.py")
    with open(module_path, "w") as module_file:
        module_file.write("class MyClass:\n    pass\n")
    monkeypatch.setattr(sys, "dont_write_bytecode", initial_value)

    import_class_from_file(file_path=module_path, class_name="MyClass")

    assert sys.dont_write_bytecode is initial_value


def test_import_class_from_file_restores_global_state_after_error(
    empty_local_dir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = os.path.join(empty_local_dir, "broken.py")
    with open(module_path, "w") as module_file:
        module_file.write("raise RuntimeError('broken import')\n")
    original_sys_path = sys.path.copy()
    monkeypatch.setattr(sys, "dont_write_bytecode", False)

    with pytest.raises(RuntimeError, match="broken import"):
        import_class_from_file(file_path=module_path, class_name="MyClass")

    assert sys.dont_write_bytecode is False
    assert sys.path == original_sys_path


def test_import_class_from_file_serializes_global_state_changes(
    empty_local_dir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_lock = Lock()
    active_loaders = 0
    maximum_active_loaders = 0

    class TrackingLoader:

        @staticmethod
        def create_module(spec):
            return None

        @staticmethod
        def exec_module(module):
            nonlocal active_loaders, maximum_active_loaders
            with state_lock:
                active_loaders += 1
                maximum_active_loaders = max(
                    maximum_active_loaders,
                    active_loaders,
                )
            time.sleep(0.05)
            module.MyClass = type("MyClass", (), {})
            with state_lock:
                active_loaders -= 1

    def create_tracking_spec(module_name: str, file_path: str):
        return importlib.machinery.ModuleSpec(
            name=module_name,
            loader=TrackingLoader(),
            origin=file_path,
        )

    monkeypatch.setattr(
        "inference_models.utils.imports.importlib.util.spec_from_file_location",
        create_tracking_spec,
    )
    original_sys_path = sys.path.copy()
    module_paths = [
        os.path.join(empty_local_dir, "concurrent_a.py"),
        os.path.join(empty_local_dir, "concurrent_b.py"),
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        classes = list(
            executor.map(
                lambda module_path: import_class_from_file(
                    file_path=module_path,
                    class_name="MyClass",
                ),
                module_paths,
            )
        )

    assert [cls.__name__ for cls in classes] == ["MyClass", "MyClass"]
    assert maximum_active_loaders == 1
    assert sys.path == original_sys_path


def test_import_class_from_file_when_invalid_module_path_provided(
    non_existing_module_path: str,
) -> None:
    # when
    with pytest.raises(FileNotFoundError):
        _ = import_class_from_file(
            file_path=non_existing_module_path, class_name="MyClass"
        )
