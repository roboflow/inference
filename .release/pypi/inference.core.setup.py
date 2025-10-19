import os
import sys

import setuptools
from setuptools import find_packages

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root)
from inference.core.version import __version__

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
            "tests",
            "tests.*",
            "development",
            "development.*",
            "inference_experimental",
            "inference_experimental.*"
        ),
    ),
    package_data={
        "inference.models.perception_encoder.vision_encoder": [
            "bpe_simple_vocab_16e6.txt.gz"
        ],
    },
    entry_points={
        "console_scripts": [
            "inference=inference_cli.main:app",
        ],
    },
    install_requires=read_requirements(
        [
            "requirements/_requirements.txt",
            "requirements/requirements.cli.txt",
            "requirements/requirements.sdk.http.txt",
        ]
    ),
    extras_require={
        "clip": read_requirements("requirements/requirements.clip.txt"),
        "easy-ocr": read_requirements("requirements/requirements.easyocr.txt"),
        "cpu": read_requirements("requirements/requirements.cpu.txt"),
        "gaze": read_requirements("requirements/requirements.gaze.txt"),
        "gpu": read_requirements("requirements/requirements.gpu.txt"),
        "grounding-dino": read_requirements(
            "requirements/requirements.groundingdino.txt"
        ),
        "hosted": read_requirements("requirements/requirements.hosted.txt"),
        "http": read_requirements("requirements/requirements.http.txt"),
        "sam": read_requirements("requirements/requirements.sam.txt"),
        "yolo-world": read_requirements("requirements/requirements.yolo_world.txt"),
        "transformers": read_requirements("requirements/requirements.transformers.txt"),
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
    python_requires=">=3.9,<3.13"
)
