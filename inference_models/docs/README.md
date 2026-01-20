# Inference Models Documentation

This directory contains the documentation source for the `inference-models` package.

## Building the Documentation

### Prerequisites

First, install the package in development mode:

```bash
# From the repository root
cd inference_models
pip install -e .
```

Then install documentation dependencies:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-gen-files mkdocs-literate-nav mike
```

### Local Development

Serve the documentation locally:

```bash
# From the inference_models directory
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

### Building Static Site

Build the documentation:

```bash
mkdocs build
```

The built site will be in the `site/` directory.

## Versioning with Mike

This documentation uses [mike](https://github.com/jimporter/mike) for version management.

### Deploy a New Version

```bash
# Deploy version 0.18.3 and set as latest
mike deploy 0.18.3 latest --update-aliases

# Deploy and set as default
mike set-default latest

# Push to gh-pages
mike deploy --push 0.18.3 latest
```

### List Versions

```bash
mike list
```

### Delete a Version

```bash
mike delete 0.18.3
```

## Documentation Structure

```
docs/
├── index.md                    # Homepage
├── getting-started/            # Installation and quick start
│   ├── overview.md
│   ├── installation.md
│   ├── principles.md
│   ├── cache.md
│   ├── hardware-compatibility.md
│   └── docker.md
├── models/                     # Model-specific documentation
│   ├── index.md
│   ├── rfdetr.md
│   ├── yolov8.md
│   └── ...
├── how-to/                     # How-to guides
│   ├── local-packages.md
│   ├── add-model.md
│   ├── custom-models.md
│   └── model-pipelines.md
├── contributors/               # Contributor guides
│   ├── architecture.md
│   ├── backends.md
│   ├── dependencies.md
│   ├── adding-models.md
│   └── testing.md
├── api-reference/              # Auto-generated API docs
└── scripts/                    # Documentation build scripts
    └── gen_ref_pages.py
```

## Writing Documentation

### Markdown Extensions

The documentation uses these markdown extensions:

- **Admonitions**: `!!! note`, `!!! warning`, `!!! tip`
- **Code blocks**: Syntax highlighting with language tags
- **Tabs**: Group related content
- **Tables**: Standard markdown tables

### Code Examples

Use fenced code blocks with language tags:

````markdown
```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")
```
````

### Admonitions

```markdown
!!! note "Optional Title"
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.

!!! danger
    This is a danger notice.
```

### Links

```markdown
# Internal links
[Installation Guide](getting-started/installation.md)

# External links
[Roboflow](https://roboflow.com)

# API reference
See [`AutoModel`][inference_models.AutoModel]
```

## Contributing

When adding new features:

1. **Update relevant documentation** - Don't just update code
2. **Add examples** - Show how to use the feature
3. **Update API reference** - Add docstrings to new functions/classes
4. **Test locally** - Run `mkdocs serve` to verify

### Documentation Checklist

- [ ] Updated relevant guide pages
- [ ] Added code examples
- [ ] Added docstrings to new code
- [ ] Tested locally with `mkdocs serve`
- [ ] Checked for broken links
- [ ] Updated navigation in `mkdocs.yml` if needed

## Deployment

Documentation is automatically deployed to GitHub Pages when:

1. A new release is created
2. Manual deployment is triggered

The deployment workflow is defined in `.github/workflows/docs.yml`.

## Questions?

- Open an issue on [GitHub](https://github.com/roboflow/inference/issues)
- Ask in [Discussions](https://github.com/roboflow/inference/discussions)
- Join our [Discord](https://discord.gg/roboflow)

