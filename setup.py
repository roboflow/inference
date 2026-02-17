import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding='utf-8') as fh:
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
    name="inference-development",
    author="Roboflow",
    author_email="help@roboflow.com",
    description="With no prior knowledge of machine learning or device-specific deployment, you can deploy a computer vision model to a range of devices and environments using Roboflow Inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow/inference",
    packages=find_packages(
        where=".",
        exclude=(
            "docker",
            "docs",
            "requirements",
            "tests",
            "tests.*",
            "development",
            "development.*",
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
            "requirements/requirements.cpu.txt",
            "requirements/requirements.cli.txt",
            "requirements/requirements.clip.txt",
            "requirements/requirements.http.txt",
            "requirements/requirements.sdk.http.txt",
            "requirements/requirements.gaze.txt",
            "requirements/requirements.groundingdino.txt",
            "requirements/requirements.hosted.txt",
            "requirements/requirements.modal.txt",
            "requirements/requirements.yolo_world.txt",
            "requirements/requirements.code_analysis.txt",
            "requirements/requirements.test.unit.txt",
            "requirements/requirements.test.integration.txt",
            "requirements/requirements.transformers.txt",
        ]
    ),
    extras_require={
        "sam": read_requirements("requirements/requirements.sam.txt"),
        "cloud-storage": read_requirements("requirements/requirements.cloud_storage.txt"),
        "two-frame-3d": read_requirements("requirements/requirements.two_frame_3d.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
