import os
import setuptools
from setuptools import find_packages
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root)
from inference.core.version import __version__

with open("README.md", "r") as fh:
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
    name="inference-gpu",
    version=__version__,
    author="Roboflow",
    author_email="help@roboflow.com",
    description="With no prior knowledge of machine learning or device-specific deployment, you can deploy a computer vision model to a range of devices and environments using Roboflow Inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow/inference",
    install_requires=read_requirements(
        [
            "requirements/_requirements.txt",
            "requirements/requirements.cli.txt",
            "requirements/requirements.gpu.txt",
        ]
    ),
    packages=find_packages(
        where=root,
        exclude=(
            "docker",
            "docs",
            "requirements",
            "tests",
            "tests.*",
            "inference_sdk",
            "inference_sdk.*",
        ),
    ),
    entry_points={
        "console_scripts": [
            "inference=inference_cli.main:app",
        ],
    },
    extras_require={
        "clip": read_requirements("requirements/requirements.clip.txt"),
        "gaze": read_requirements("requirements/requirements.gaze.txt"),
        "hosted": read_requirements("requirements/requirements.hosted.txt"),
        "http": read_requirements("requirements/requirements.http.txt"),
        "sam": read_requirements("requirements/requirements.sam.txt"),
        "waf": read_requirements("requirements/requirements.waf.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
