We recommend using [python virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html) to isolate dependencies of inference.

To install Inference via pip:

```bash
pip install inference
```

If you have an NVIDIA GPU, you can accelerate your inference with:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 inference-gpu  
# please adjust the --extra-index-url to CUDA version installed in your OS
# https://download.pytorch.org/whl/cu<major><minor>, for instance https://download.pytorch.org/whl/cu130 for CUDA 13.0
# alternativelly use
uv pip install inference-gpu
```