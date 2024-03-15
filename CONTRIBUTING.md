# Contributing to the Roboflow Inference Server 🛠️

Thank you for your interest in contributing to the Roboflow Inference Server!

We welcome any contributions to help us improve the quality of `inference-server` and expand the range of supported models.

## Contribution Guidelines

We welcome contributions to:

1. Add support for running inference on a new model.
2. Report bugs and issues in the project.
3. Submit a request for a new task or feature.
4. Improve our test coverage.

### Contributing Features

The goal of `inference` is to make it easy to adopt computer vision models. The package provides standardised interface
for making prediction, make it possible to expose model through HTTP API and enable making predictions from 
models on video. Our goal is also to make it seamless to integrate `inference` components with Roboflow platform.

We welcome contributions that add support for new models to the project. Before you begin, please make sure that another contributor has not already begun work on the model you want to add. You can check the [project README](https://github.com/roboflow/inference-server/blob/main/README.md) for our roadmap on adding more models.

We require documentation and tests for contributions (if applicable).

## How to Contribute Changes

First, fork this repository to your own GitHub account. Create a new branch that describes your changes (i.e. `line-counter-docs`). Push your changes to the branch on your fork and then submit a pull request to this repository.

When creating new functions, please ensure you have the following:

1. Docstrings for the function and all parameters.
2. Examples in the documentation for the function.
3. Created an entry in our docs to autogenerate the documentation for the function.

All pull requests will be reviewed by the maintainers of the project. We will provide feedback and ask for changes if necessary.

PRs must pass all tests and linting requirements before they can be merged.

## :wrench: Development environment
We recommend creating fresh conda environment:
```bash
conda create -n inference-development python=3.10
conda activate inference-development
```

Then, in repository root:
```bash
repo_root$ (inference-development) pip install -e .
```

That will install all requirements apart from SAM model. To install the latter:
```bash
repo_root$ (inference-development) pip install -e ".[sam]"
```
but in some OS (like MacOS) that would require installing additional libs ([this](https://medium.com/@vascofernandes_13322/how-to-install-gdal-on-macos-6a76fb5e24a4) guide should fix the issue for MacOS).

After installation, you should be able to run both tests and the library components without issues.

## 🐳 Building docker image with inference server

To test the changes related to `inference` server, you would probably need to build docker image locally.
This is to be done with the following command:

```bash
# template
repo_root$ docker build -t roboflow/roboflow-inference-server-{version}:dev -f docker/dockerfiles/Dockerfile.onnx.{version} .

# example build for CPU
repo_root$ docker build -t roboflow/roboflow-inference-server-cpu:dev -f docker/dockerfiles/Dockerfile.onnx.cpu .

# example build for GPU
repo_root$ docker build -t roboflow/roboflow-inference-server-gpu:dev -f docker/dockerfiles/Dockerfile.onnx.gpu .
```

## 🧹 Code quality 

We provide two handy commands inside the `Makefile`, namely:

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)


## 🧪 Tests 

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests. We have specific structure of tests to ensure stability on different 
platforms that we support (CPU, GPU, Jetson, etc.). 

### Unit tests

We would like all low-level components to be covered with unit tests. That tests must be:
* fast (if that's not possible for some reason, please use `@pytest.mark.slow`)
* deterministic (not flaky)
* covering all [equivalence classes](https://piketec.com/testing-with-equivalence-classes/#:~:text=Testing%20with%20equivalence%20classes&text=Equivalence%20classes%20in%20the%20test,class%20you%20use%20as%20input.)

Running the unit tests:

```bash
repo_root$ (inference-development) pytest tests/inference/unit_tests/ 
repo_root$ (inference-development) pytest tests/inference_cli/unit_tests/ 
repo_root$ (inference-development) pytest tests/inference_sdk/unit_tests/ 
```

With GH Actions defined in `.github` directory, the ones related to integration tests at `x86` platform 
will work after you fork repositories. Other actions may not work, as they require access to our internal resources
will not work (tests on Jetson devices, Tesla T4, integration tests for `inference` server). There is nothing wrong 
with that, we will make required checks as you submit PR to main repository.

### Integration tests

We would like to have decent coverage of most important components with integration tests suites.
Those should check specific functions e2e, including communication with external services (or their stubs if 
real service cannot be used for any reasons). Integration tests may be more bulky than unit tests, but we wish them
not to require burning a lot of resources, and be completed within max 20-30 minutes.

Running the integration tests locally is possible, but only in some cases. For instance, one may locally run:
```bash
repo_root$ (inference-development) pytest tests/inference/models_predictions_tests/ 
repo_root$ (inference-development) pytest tests/inference_cli/integration_tests/ 
```

But running 
```bash
repo_root$ (inference-development) pytest tests/inference/integration_tests/ 
```
will not be fully possible, as part of them require API key for Roboflow API.

#### :bulb:	Contribution idea

It would be a great contribution to make `inference` server integration tests running without API keys for Roboflow. 


## 📚 Documentation

Roboflow Inference uses mkdocs and mike to offer versioned documentation. The project documentation is hosted on [GitHub Pages](https://inference.roboflow.com).

To build the Inference documentation, first install the project development dependencies:

```bash
pip install -r requirements/requirements.docs.txt
```

To run the latest version of the documentation, run:

```bash
mike serve
```

Before a new release is published, a new version of the documentation should be built. To create a new version, run:

```bash
mike deploy <version-number>
```