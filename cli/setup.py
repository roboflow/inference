import os
import setuptools
from setuptools import find_packages
import sys
import shutil

root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(root)

shutil.copyfile(
    os.path.join(root, "../inference/core/version.py"),
    os.path.join(root, "inference_cli/version.py"),
)
from inference_cli.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements(path):
    with open(os.path.join(root, path)) as fh:
        return [line.strip() for line in fh]


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
        where="./",
    ),
    entry_points={
        "console_scripts": [
            "inference=inference_cli.main:app",
        ],
    },
    install_requires=read_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
