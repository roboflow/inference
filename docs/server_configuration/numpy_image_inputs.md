# NumPy image inputs

Use the explicit `npy` image type to send a NumPy image without pickle or lossy
JPEG compression. Both server stacks accept it in every configuration.

```python
import base64
import io

import numpy as np
import requests

# image is a uint8 array in BGR order, for example from cv2.imread().
buffer = io.BytesIO()
np.save(buffer, image, allow_pickle=False)
image_input = {
    "type": "npy",
    "value": base64.b64encode(buffer.getvalue()).decode("ascii"),
}
response = requests.post(
    "http://localhost:9001/infer/object_detection",
    json={"model_id": "your-project/1", "api_key": api_key, "image": image_input},
)
response.raise_for_status()
```

The payload must contain one complete NPY 1.0 or 2.0 array with `uint8` pixels.
Supported shapes are `(height, width, 3)` for BGR,
`(height, width)` for grayscale, and `(height, width, 1)` for grayscale.
Both dimensions must be positive. C and Fortran array layouts are supported.
The server rejects object arrays, other dtypes, NPZ archives, and truncated or
concatenated files. Existing request-size limits still apply.

Python callers of the image loader can pass the raw NPY bytes as `value`.
HTTP JSON callers must base64-encode them and declare `type: "npy"`;
the format is not inferred from an untyped string.

Workflow image inputs also accept this declaration when the server and its
`roboflow-workflows` package both include this migration. Upgrade both together;
older Workflows packages discard the type before decoding the value.

## Migrating pickled inputs

The pickled `numpy` image type has been removed: servers never call
`pickle.loads` on request data, because unpickling a client payload can run
arbitrary code. Both server stacks now reject `type: "numpy"` with HTTP 400 in
every configuration, with an error that points to `npy`. `ALLOW_NUMPY_INPUT`
no longer has any effect.

To migrate, replace `pickle.dumps(image)` with the `np.save` example above
and change the declared type from `numpy` to `npy`. For an existing uint8 BGR
image, this preserves every pixel and its channel order. Changing only the type
does not convert a pickle payload to NPY.

The Python SDK continues to encode NumPy inputs as JPEG by default. This
migration adds an explicit lossless input option; it does not change that SDK
default or the in-process `numpy_object` format.
