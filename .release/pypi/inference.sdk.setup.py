import os
import shutil
import sys

import setuptools
from setuptools import find_packages

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root)

shutil.copyfile(
    os.path.join(root, "inference/core/version.py"),
    os.path.join(root, "inference_sdk/version.py"),
)

# Read the version without importing the package - `inference/__init__.py`
# pulls in the full runtime (inference_models, torch, cv2), which wheel-build
# stages may not be able to load.
_version_namespace = {"__name__": "inference.core.version"}
with open(os.path.join(root, "inference", "core", "version.py")) as _version_file:
    exec(_version_file.read(), _version_namespace)
__version__ = _version_namespace["__version__"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read_requirements(path):
    if not isinstance(path, list):
        path = [path]
    requirements = []
    for p in path:
        with open(p) as fh:
            requirements.extend([line.strip() for line in fh])
    return requirements


setuptools.setup(
    name="inference-sdk",
    version=__version__,
    author="Roboflow",
    author_email="help@roboflow.com",
    description="With no prior knowledge of machine learning or device-specific deployment, you can deploy a computer vision model to a range of devices and environments using Roboflow Inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow/inference",
    packages=find_packages(
        where=root,
        exclude=(
            "docker",
            "docs",
            "requirements",
            "tests",
            "tests.*",
            "inference",
            "inference.*",
            "inference_cli",
            "inference_cli.*",
            "development",
            "development.*",
            "inference_models",
            "inference_models.*",
            "inference_model_manager",
            "inference_model_manager.*",
            "inference_server",
            "inference_server.*"
        ),
    ),
    install_requires=read_requirements(["requirements/requirements.sdk.http.txt"]),
    extras_require={
        "webrtc": read_requirements(["requirements/requirements.sdk.webrtc.txt"]),
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Typing :: Typed",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<3.13"
)
