#!/bin/bash

PROJECT=roboflow-platform ENABLE_BUILDER=True ENABLE_STREAM_API=True watchmedo auto-restart --pattern="*.py" --recursive -- uvicorn cpu_http:app --port 9001
