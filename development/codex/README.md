# Codex: Agent Assisted Coding from OpenAI

chatgpt.com/codex is an AI-powered coding assistant from OpenAI that helps you write, understand, and improve code through natural language interactions.

## Setup Instructions

- Create an environment in codex pointing to this repo
- Add these environment variables:
    - PROJECT=roboflow-platform
    - ALLOW_NUMPY_INPUT=True
    - API_LOGGING_ENABLED=True
    - API_DEBUG=True
    - ENABLE_STREAM_API=True
- Add the secret `ROBOFLOW_API_KEY` with a production API key from Roboflow
- Configure the setup script with the contents of [`development/codex/setup.sh`](setup.sh)