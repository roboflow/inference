import os
import shutil
import tempfile
import unittest
import torch
from loguru import logger


def get_checkpoint_folder():
    if os.path.exists("/checkpoint/ego-howto"):  # on RSC/AVA
        return f"/checkpoint/ego-howto/{os.environ['USER']}/lidra"
    return f"/checkpoint/{os.environ['USER']}/lidra"


class OverwriteTensorEquality:
    """Convenient tool for overwriting torch.Tensor's (in)equality functions.

    Examples
    --------

    ```python
    import torch

    tensor = torch.tensor([1, 2, 3])

    if tensor == tensor: # raise exception
        print("not executed")
    # >>> RuntimeError: Boolean value of Tensor with more than one value is ambiguous

    with OverwriteTensorEquality(torch):
        if tensor == tensor: # doesn't raise exception
            print("executed")
    # >>> executed
    ```

    """

    PYTORCH_EQ_FUNCTIONS = {}

    @staticmethod
    def _register_and_get_eq_function(torch_module):
        module_id = id(torch_module)
        return OverwriteTensorEquality.PYTORCH_EQ_FUNCTIONS.setdefault(
            module_id,
            torch_module.Tensor.__eq__,
        )

    def __init__(
        self,
        torch_module,
        check_shape=False,
        check_dtype=False,
        check_device=False,
        custom_eq_fn=None,
    ) -> None:
        """

        Parameters
        ----------
        torch_module : module
            Torch module to overwrite the torch.Tensor functions from.
        """

        self._torch_module = torch_module
        OverwriteTensorEquality._register_and_get_eq_function(
            self._torch_module
        )  # makes sure original functions are saved first

        self._pytorch_tensor_eq = self._torch_module.Tensor.__eq__
        self._pytorch_tensor_ne = self._torch_module.Tensor.__ne__

        self._check_shape = check_shape
        self._check_dtype = check_dtype
        self._check_device = check_device
        self._custom_eq_fn = custom_eq_fn

    def _test_tensor_eq(self):
        if self._custom_eq_fn is not None:
            return self._custom_eq_fn

        def eq(t0, t1):
            if self._check_shape and t0.shape != t1.shape:
                return False
            if self._check_dtype and t0.dtype != t1.dtype:
                return False
            if self._check_device and t0.device != t1.device:
                return False

            eq_fn = self._register_and_get_eq_function(self._torch_module)
            return eq_fn(t0, t1).all().item()

        return eq

    def _test_tensor_ne(self):
        eq = self._test_tensor_eq()

        def ne(t0, t1):
            return not eq(t0, t1)

        return ne

    def __enter__(self):
        self._torch_module.Tensor.__eq__ = self._test_tensor_eq()
        self._torch_module.Tensor.__ne__ = self._test_tensor_ne()
        return self

    def __exit__(self, type, value, traceback):
        self._torch_module.Tensor.__eq__ = self._pytorch_tensor_eq
        self._torch_module.Tensor.__ne__ = self._pytorch_tensor_ne


def string_to_tensor(string: str, *args, **kwargs):
    array = []
    for line in string.split("\n"):
        elements = line.split(",")
        if len(elements) > 1:
            array.append([float(el) for el in line.split(",") if len(el.strip()) > 0])
    return torch.tensor(array, *args, **kwargs)


def temporary_file():
    """Automatically creates a temporary file (for testing purposes) and cleans it once it gets out of scope.

    Examples
    --------

    ```python

    with temporary_file() as filepath:
        ... # do something with file at "filepath"

    # here the file "filepath" has been removed
    ```

    """
    return _TemporaryFile()


class _TemporaryFile:
    def __enter__(self):
        fd, self.filepath = tempfile.mkstemp()
        os.close(fd)
        return self.filepath

    def __exit__(self, type, value, traceback):
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)


def temporary_directory():
    """Automatically creates a temporary directory (for testing purposes) and cleans it once it gets out of scope.

    Examples
    --------

    ```python

    with temporary_directory() as dirpath:
        ... # do something with directory at "dirpath"

    # here the directory "dirpath" has been removed
    ```

    """
    return _TemporaryDirectory()


class _TemporaryDirectory:
    def __enter__(self):
        self.dirpath = tempfile.mkdtemp()
        return self.dirpath

    def __exit__(self, type, value, traceback):
        if os.path.isdir(self.dirpath):
            shutil.rmtree(self.dirpath)


def max_tensor_diff(tensor0, tensor1):
    return torch.max(torch.abs(tensor0 - tensor1)).detach().cpu().item()


def run_only_if_cuda_is_available(default_device=None):
    def wrapper(fn):
        def new_fn(*args, **kwargs):
            if torch.cuda.is_available():
                with torch.device(default_device):
                    fn(*args, **kwargs)
            else:
                raise unittest.SkipTest(
                    "skip test since CUDA is not available on this machine"
                )

        return new_fn

    return wrapper


def run_only_if_path_exists(path: str):
    def wrapper(fn):
        def new_fn(*args, **kwargs):
            if os.path.exists(path):
                fn(*args, **kwargs)
            else:
                raise unittest.SkipTest(
                    f'skip test since path "{path}" is not available on this machine'
                )

        return new_fn

    return wrapper


def run_unittest(test_cls: unittest.TestCase, debug: bool = True):
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_cls)
    if debug:
        test_suite.debug()
    else:
        test_runner = unittest.runner.TextTestRunner(
            verbosity=1,
            failfast=True,
            buffer=False,
            warnings=None,
        )
        test_runner.run(test_suite)
