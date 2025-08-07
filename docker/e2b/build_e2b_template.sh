#!/bin/bash

# Script to build and push E2B template for Inference Custom Python Blocks
# Usage: ./build_e2b_template.sh [push]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
E2B_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building E2B Template for Inference Custom Python Blocks${NC}"

# Check if e2b CLI is installed
if ! command -v e2b &> /dev/null; then
    echo -e "${RED}Error: e2b CLI is not installed${NC}"
    echo "Please install it with: npm install -g @e2b/cli"
    exit 1
fi

# Check if E2B_API_KEY is set
if [ -z "$E2B_API_KEY" ]; then
    echo -e "${YELLOW}Warning: E2B_API_KEY environment variable is not set${NC}"
    echo "You'll need to set it to push the template"
fi

# Get current inference version
INFERENCE_VERSION=$(grep "^__version__" ../../inference/core/version.py | head -1 | cut -d'"' -f2)
echo -e "${GREEN}Using Inference version: ${INFERENCE_VERSION}${NC}"

# Create template name with version
TEMPLATE_NAME="inference-sandbox-v${INFERENCE_VERSION//./-}"
echo -e "${GREEN}Building template: ${TEMPLATE_NAME}${NC}"

# Build the template
cd "$E2B_DIR"
# Build from the root of the repository with name specification
e2b template build --path ../.. --dockerfile docker/e2b/e2b.Dockerfile --config docker/e2b/e2b.toml --name "${TEMPLATE_NAME}"

# Check if we should push
if [ "$1" == "push" ]; then
    if [ -z "$E2B_API_KEY" ]; then
        echo -e "${RED}Error: Cannot push without E2B_API_KEY${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Pushing template to E2B...${NC}"
    e2b template push --path ../.. --config docker/e2b/e2b.toml
    
    echo -e "${GREEN}Template pushed successfully!${NC}"
    echo -e "${GREEN}Template Name: ${TEMPLATE_NAME}${NC}"
    
    # List templates to verify
    echo -e "${GREEN}Verifying template in E2B:${NC}"
    e2b template list | grep inference-sandbox || true
else
    echo -e "${YELLOW}Template built locally. To push to E2B, run: $0 push${NC}"
fi
