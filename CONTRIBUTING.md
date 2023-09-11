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

The Inference Server provides a standard interface through which you can work with computer vision models. With Inference Server, you can use state-of-the-art models with your own weights without having to spend time installing dependencies, configuring environments, and writing inference code.

We welcome contributions that add support for new models to the project. Before you begin, please make sure that another contributor has not already begun work on the model you want to add. You can check the [project README](https://github.com/roboflow/inference-server/blob/main/README.md) for our roadmap on adding more models.

You will need to add documentation for your model and link to it from the `inference-server` README. You can add a new page to the `docs/models` directory that describes your model and how to use it. You can use the existing model documentation as a guide for how to structure your documentation.

## How to Contribute Changes

First, fork this repository to your own GitHub account. Create a new branch that describes your changes (i.e. `line-counter-docs`). Push your changes to the branch on your fork and then submit a pull request to this repository.

When creating new functions, please ensure you have the following:

1. Docstrings for the function and all parameters.
2. Examples in the documentation for the function.
3. Created an entry in our docs to autogenerate the documentation for the function.

All pull requests will be reviewed by the maintainers of the project. We will provide feedback and ask for changes if necessary.

PRs must pass all tests and linting requirements before they can be merged.

## 🧹 Code quality 

We provide two handy commands inside the `Makefile`, namely:

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)

## 🧪 Tests 

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.

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