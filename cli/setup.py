import os
import setuptools
from setuptools import find_packages
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root)
from version import __version__


def read_requirements(path):
    with open(path) as fh:
        return [line.strip() for line in fh]


setuptools.setup(
    name="inference-cli",
    version="0.0.3",
    author="Roboflow",
    author_email="help@roboflow.com",
    description="Inference CLI coming soon!",
    long_description="Inference CLI coming soon!",
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow/inference",
    packages=find_packages(
        where="./",
    ),
    entry_points={
        "console_scripts": [
            "inference=inference_cli.__init__:main",
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
