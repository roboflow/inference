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


def test_import_class_from_file_when_invalid_module_path_provided(
    non_existing_module_path: str,
) -> None:
    # when
    with pytest.raises(FileNotFoundError):
        _ = import_class_from_file(
            file_path=non_existing_module_path, class_name="MyClass"
        )
