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
INFERENCE_VERSION=$(grep "__version__" ../../inference/core/version.py | cut -d'"' -f2)
echo -e "${GREEN}Using Inference version: ${INFERENCE_VERSION}${NC}"

# Update template_id in e2b.toml
TEMPLATE_ID="inference-sandbox-${INFERENCE_VERSION}"
sed -i.bak "s/template_id = \"inference-sandbox-.*\"/template_id = \"${TEMPLATE_ID}\"/" e2b.toml
rm e2b.toml.bak

echo -e "${GREEN}Building template: ${TEMPLATE_ID}${NC}"

# Build the template
cd "$E2B_DIR"
e2b template build

# Check if we should push
if [ "$1" == "push" ]; then
    if [ -z "$E2B_API_KEY" ]; then
        echo -e "${RED}Error: Cannot push without E2B_API_KEY${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Pushing template to E2B...${NC}"
    e2b template push
    
    echo -e "${GREEN}Template pushed successfully!${NC}"
    echo -e "${GREEN}Template ID: ${TEMPLATE_ID}${NC}"
    
    # List templates to verify
    echo -e "${GREEN}Verifying template in E2B:${NC}"
    e2b template list | grep inference-sandbox || true
else
    echo -e "${YELLOW}Template built locally. To push to E2B, run: $0 push${NC}"
fi
