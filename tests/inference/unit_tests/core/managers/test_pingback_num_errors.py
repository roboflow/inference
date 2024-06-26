from inference.core.managers.base import ModelManager, ModelRegistry
from inference.core.managers.decorators.base import ModelManagerDecorator


def test_increment_num_errors():
    mm = ModelManager(ModelRegistry(dict()))
    mm_wrapper = ModelManagerDecorator(mm)
    mm_wrapper.init_pingback()
    mm_wrapper.num_errors += 1
    assert mm.num_errors == mm_wrapper.num_errors == 1
    mm.num_errors += 1
    assert mm.num_errors == mm_wrapper.num_errors == 2
