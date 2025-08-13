# E2B Sandbox REPL

A Python-based REPL for interacting with E2B sandboxes. Perfect for testing and debugging Custom Python Blocks in the Roboflow Inference sandbox environment.

## Quick Start

```bash
# Run the REPL (creates a new sandbox)
python3 e2b_repl.py

# Connect to an existing sandbox
python3 e2b_repl.py <sandbox_id>

# Use a specific template
python3 e2b_repl.py -t <template_id>
```

## Features

- **Interactive Python execution** in E2B sandboxes
- **File management** (upload, download, list, view)
- **Persistent sessions** for iterative development
- **Multi-line code support** for functions and classes
- **Built-in help** and sandbox information

## Available Commands

### Python Code
Just type Python code and press Enter:
```python
>>> import numpy as np
>>> arr = np.array([1, 2, 3, 4, 5])
>>> print(f"Mean: {arr.mean()}, Sum: {arr.sum()}")
Mean: 3.0, Sum: 15
```

### Special Commands
- `!help` - Show help information
- `!info` - Display sandbox details
- `!ls [path]` - List files in directory
- `!cat <path>` - Show file contents
- `!upload <local> <remote>` - Upload file to sandbox
- `!download <remote> <local>` - Download file from sandbox
- `!exit` or `!quit` - Exit the REPL

## Multi-line Code
The REPL automatically detects multi-line code:
```python
>>> def factorial(n):
...     if n <= 1:
...         return 1
...     return n * factorial(n - 1)
... 
>>> print(factorial(5))
120
```

## Inference Template
By default, uses the Roboflow Inference sandbox template (`qfupheopqmf6w7b36h6o`) which includes:
- NumPy, SciPy, Pandas
- OpenCV, PIL/Pillow
- Supervision
- Inference SDK modules
- All Custom Python Block dependencies

## Environment
The E2B API key is configured: `e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e`

To use a different API key:
```bash
export E2B_API_KEY=your_api_key
python3 e2b_repl.py
```

## Examples

### Testing Inference modules
```python
>>> import sys
>>> sys.path.append('/app')
>>> from inference.core.workflows.execution_engine.entities.base import Batch
>>> print("Inference modules loaded!")
```

### Working with Supervision
```python
>>> import supervision as sv
>>> import numpy as np
>>> detections = sv.Detections(
...     xyxy=np.array([[10, 10, 50, 50]]),
...     confidence=np.array([0.9]),
...     class_id=np.array([0])
... )
>>> print(f"Detections: {len(detections)}")
```

### File Operations
```python
>>> # Create a file in the sandbox
>>> with open('/tmp/test.txt', 'w') as f:
...     f.write('Hello from E2B!')
>>> 
>>> !cat /tmp/test.txt
Hello from E2B!
>>> 
>>> !download /tmp/test.txt ./test.txt
✅ Downloaded /tmp/test.txt to ./test.txt
```

## Notes

- Sandboxes persist for their configured timeout (default 5 minutes idle)
- All Python packages in the template are pre-installed
- The `/app` directory contains the Inference modules
- Use `Ctrl+C` to cancel multi-line input
- Use `Ctrl+D` or `!exit` to quit
