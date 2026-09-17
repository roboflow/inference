#!/usr/bin/env python3
"""Build-time proof that Triton can JIT-compile kernels in this image.

Triton compiles GPU kernels lazily at first launch. That step shells out to a
host C compiler and builds the kernel launcher as a CPython extension, so a
runtime image needs: a discoverable C compiler (Triton checks the CC
environment variable, then well-known compiler names), libc headers, CPython
headers (Python.h), and the cuda.h header bundled inside the Triton wheel.
None of this needs a GPU, so it can and should be validated when the image is
built instead of surfacing as a per-frame RuntimeError at first inference
(the failure mode that killed RF-DETR on the GPU image: "Failed to find C
compiler. Please specify via CC environment variable.").

Checks performed:
  1. A C compiler is discoverable the way Triton discovers one.
  2. Python.h exists in the interpreter's include directory.
  3. The triton package imports.
  4. The Triton wheel ships its bundled cuda.h.
  5. A minimal CPython extension compiles with the discovered compiler and
     imports back into this interpreter (the same shape of work Triton's
     launcher build performs).

Exit code is non-zero with an actionable message when any check fails.
"""

import glob
import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile

_COMPILER_CANDIDATES = ("cc", "gcc", "clang")

_PROBE_MODULE_NAME = "roboflow_triton_jit_probe"
_PROBE_SOURCE = """
#include <Python.h>

static PyObject *probe_value(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromLong(42);
}

static PyMethodDef probe_methods[] = {
    {"probe_value", probe_value, METH_NOARGS, "Return a fixed probe value."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef probe_module = {
    PyModuleDef_HEAD_INIT,
    "%(module_name)s",
    NULL,
    -1,
    probe_methods,
};

PyMODINIT_FUNC PyInit_%(module_name)s(void) {
    return PyModule_Create(&probe_module);
}
""" % {"module_name": _PROBE_MODULE_NAME}


def _fail(message: str) -> None:
    print(f"verify_triton_jit_toolchain: FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def _check(label: str, value: str) -> None:
    print(f"verify_triton_jit_toolchain: OK: {label}: {value}")


def find_c_compiler() -> str:
    explicit = os.environ.get("CC")
    if explicit:
        resolved = shutil.which(explicit)
        if resolved is None:
            _fail(
                f"the CC environment variable points at {explicit!r}, which does "
                "not resolve to an executable"
            )
        return resolved
    for candidate in _COMPILER_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    _fail(
        "no C compiler found (checked the CC environment variable and "
        + ", ".join(_COMPILER_CANDIDATES)
        + "); install gcc in the runtime stage"
    )
    raise AssertionError("unreachable")


def find_python_include_dir() -> str:
    include_dir = sysconfig.get_paths()["include"]
    header = os.path.join(include_dir, "Python.h")
    if not os.path.isfile(header):
        _fail(
            f"Python.h not found at {header}; install the CPython development "
            "headers (libpython3.x-dev) in the runtime stage"
        )
    return include_dir


def check_triton_import() -> str:
    try:
        import triton
    except ImportError as error:
        _fail(f"the triton package is not importable: {error}")
        raise AssertionError("unreachable")
    version = getattr(triton, "__version__", "unknown")
    return f"{version} at {os.path.dirname(triton.__file__)}"


def check_bundled_cuda_header() -> str:
    import triton

    package_dir = os.path.dirname(triton.__file__)
    matches = sorted(
        glob.glob(os.path.join(package_dir, "**", "cuda.h"), recursive=True)
    )
    if not matches:
        _fail(
            f"no bundled cuda.h found under {package_dir}; Triton cannot compile "
            "its CUDA launcher without it"
        )
    return matches[0]


def check_extension_compile(compiler: str, python_include_dir: str) -> str:
    with tempfile.TemporaryDirectory(prefix="triton-jit-probe-") as workdir:
        source_path = os.path.join(workdir, f"{_PROBE_MODULE_NAME}.c")
        binary_path = os.path.join(workdir, f"{_PROBE_MODULE_NAME}.so")
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write(_PROBE_SOURCE)
        command = [
            compiler,
            source_path,
            "-O3",
            "-shared",
            "-fPIC",
            f"-I{python_include_dir}",
            "-o",
            binary_path,
        ]
        if sys.platform == "darwin":
            # Linux resolves Python symbols at import time; Apple's linker
            # needs to be told so explicitly. Only relevant when running this
            # script on a developer machine - images are Linux.
            command += ["-undefined", "dynamic_lookup"]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            _fail(
                "compiling a minimal CPython extension failed (command: "
                + " ".join(command)
                + "); output:\n"
                + completed.stdout
            )
        loader = importlib.machinery.ExtensionFileLoader(
            _PROBE_MODULE_NAME,
            binary_path,
        )
        spec = importlib.util.spec_from_loader(_PROBE_MODULE_NAME, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        if module.probe_value() != 42:
            _fail("the compiled probe extension returned an unexpected value")
        return " ".join(command)


def main() -> None:
    compiler = find_c_compiler()
    _check("C compiler", compiler)
    python_include_dir = find_python_include_dir()
    _check("Python.h", os.path.join(python_include_dir, "Python.h"))
    _check("triton", check_triton_import())
    _check("bundled cuda.h", check_bundled_cuda_header())
    _check("extension compile+import", check_extension_compile(compiler, python_include_dir))
    print("verify_triton_jit_toolchain: PASS")


if __name__ == "__main__":
    main()
