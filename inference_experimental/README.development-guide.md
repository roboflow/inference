This document forms hints when performing development activities related to the `inference-experimental` project.

## Update uv.lock

SAM2-real-time has requirements on presence of CUDA, hence in most cases it should not be taken under consideration when updating `uv.lock`

```bash
uv sync --no-extra sam2-real-time
```
