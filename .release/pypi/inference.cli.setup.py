import os
import shutil
import sys

import setuptools
from setuptools import find_packages

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root)

shutil.copyfile(
    os.path.join(root, "inference/core/version.py"),
    os.path.join(root, "inference_cli/version.py"),
)
shutil.copyfile(
    os.path.join(root, "inference/core/version.py"),
    os.path.join(root, "inference_sdk/version.py"),
)

from inference.core.version import __version__

with open("inference_cli/README.md", "r", encoding="utf-8") as fh:
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
    name="inference-cli",
    version=__version__,
    author="Roboflow",
    author_email="help@roboflow.com",
    description="With no prior knowledge of machine learning or device-specific deployment, you can deploy a computer vision model to a range of devices and environments using Roboflow Inference CLI.",
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
            "development",
            "development.*",
            "inference_models",
            "inference_models.*"
        ),
    ),
    entry_points={
        "console_scripts": [
            "inference=inference_cli.main:app",
        ],
    },
    extras_require={
        "cloud-deploy": read_requirements("requirements/requirements.cloud_deploy.txt"),
        "cloud-storage": read_requirements("requirements/requirements.cloud_storage.txt"),
    },
    package_data={"": ["configs/*.yml"]},
    install_requires=read_requirements([
        "requirements/requirements.cli.txt",
        "requirements/requirements.sdk.http.txt",
    ]),
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
    python_requires=">=3.9,<3.13"
)
