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
    with open(path) as fh:
        return [line.strip() for line in fh]


setuptools.setup(
    name="inference-core",
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
            "test",
        ),
    ),
    install_requires=read_requirements("requirements/_requirements.txt"),
    extras_require={
        "cpu": read_requirements("requirements/requirements.cpu.txt"),
        "gpu": read_requirements("requirements/requirements.gpu.txt"),
        "hosted": read_requirements("requirements/requirements.hosted.txt"),
        "http": read_requirements("requirements/requirements.http.txt"),
        "waf": read_requirements("requirements/requirements.waf.txt"),
        "gaze": read_requirements("requirements/requirements.gaze.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
