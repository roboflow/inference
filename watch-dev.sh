#!/bin/bash

# PROJECT=roboflow-platform watchmedo auto-restart --pattern="*.py" --recursive -- uvicorn cpu_http:app --port 3000
PROJECT=roboflow-platform watchmedo auto-restart --pattern="*.py" --recursive -- uvicorn cpu_http:app --port 3000