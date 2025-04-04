import sys
from cpu_http import app
import uvicorn


"""
convenient script to run server in debug 
(i.e. runs uvicorn directly)

see https://www.loom.com/share/48f71894427a473cac39eca25f6ac759

- uv venv
- source .venv/bin/activate
- uv pip install -e .
- # start debugrun.py in debug mode

"""

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=9001)
    except Exception as e:
        print("Error starting server:", e)
        sys.exit(1)