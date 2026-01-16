# Workflow Block Filtering Configuration

## Overview
The workflow block filtering feature allows you to selectively disable certain types of blocks in your workflows. This is useful for various scenarios:

- **Infrastructure Testing/Mirroring**: Prevent duplicate side effects when mirroring requests
- **Cost Control**: Disable expensive foundation model blocks in development environments
- **Security Policies**: Restrict external API calls or data exfiltration
- **Environment Restrictions**: Different rules for dev, staging, and production
- **Compliance**: Ensure workflows meet regulatory requirements
- **Resource Management**: Limit resource-intensive operations

## How It Works
When enabled, the validation checks each workflow step during compilation and rejects workflows containing disabled blocks. The validation is performed at compilation time, providing immediate feedback before any execution occurs.

## Configuration

### Environment Variables

#### `WORKFLOW_SELECTIVE_BLOCKS_DISABLE`
- **Type**: Boolean (true/false)
- **Default**: `false`
- **Description**: Enables or disables the block filtering feature
- **Example**: `export WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true`
- **Backward Compatibility**: Also supports `WORKFLOW_MIRRORING_MODE` for legacy systems

#### `WORKFLOW_DISABLED_BLOCK_TYPES`
- **Type**: Comma-separated string
- **Default**: Empty (no types disabled by default)
- **Description**: Block type categories to disable (from block manifest's `block_type` field)
- **Valid Values**: `sink`, `model`, `transformation`, `visualization`, `analytics`, etc.
- **Example**: `export WORKFLOW_DISABLED_BLOCK_TYPES="sink,model"`
- **Backward Compatibility**: Falls back to `WORKFLOW_BLOCKED_BLOCK_TYPES` if set

#### `WORKFLOW_DISABLED_BLOCK_PATTERNS`
- **Type**: Comma-separated string
- **Default**: Empty (no patterns disabled by default)
- **Description**: Patterns to match in block identifiers (case-insensitive substring match)
- **Example**: `export WORKFLOW_DISABLED_BLOCK_PATTERNS="webhook,openai,anthropic"`
- **Backward Compatibility**: Falls back to `WORKFLOW_BLOCKED_BLOCK_PATTERNS` if set

#### `WORKFLOW_DISABLE_REASON`
- **Type**: String
- **Default**: `"These blocks are disabled by system configuration."`
- **Description**: Custom message explaining why blocks are disabled
- **Example**: `export WORKFLOW_DISABLE_REASON="Foundation models are disabled to reduce costs in development."`

## Usage Scenarios

### 1. Mirroring/Testing Infrastructure
Prevent duplicate side effects when testing with mirrored requests:
```bash
export WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true
export WORKFLOW_DISABLED_BLOCK_TYPES="sink,model"
export WORKFLOW_DISABLED_BLOCK_PATTERNS="webhook,email,database,openai,anthropic"
export WORKFLOW_DISABLE_REASON="Blocks disabled to prevent duplicate side effects during mirroring."
```

### 2. Development Environment
Reduce costs and prevent accidental production actions:
```bash
export WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true
export WORKFLOW_DISABLED_BLOCK_TYPES="sink"
export WORKFLOW_DISABLED_BLOCK_PATTERNS="openai,anthropic,google_gemini,stability_ai"
export WORKFLOW_DISABLE_REASON="External APIs and sinks disabled in development environment."
```

### 3. Cost Control
Disable expensive operations:
```bash
export WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true
export WORKFLOW_DISABLED_BLOCK_PATTERNS="gpt-4,claude,gemini-pro,stability_ai"
export WORKFLOW_DISABLE_REASON="Premium AI models are disabled. Please use standard models."
```

### 4. Security-Restricted Environment
Prevent data exfiltration and external communication:
```bash
export WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true
export WORKFLOW_DISABLED_BLOCK_TYPES="sink"
export WORKFLOW_DISABLED_BLOCK_PATTERNS="webhook,http,email,slack,database"
export WORKFLOW_DISABLE_REASON="External communication is restricted for security reasons."
```

### 5. Compliance Mode
Ensure GDPR/HIPAA compliance:
```bash
export WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true
export WORKFLOW_DISABLED_BLOCK_PATTERNS="google,openai,anthropic,external"
export WORKFLOW_DISABLE_REASON="Third-party AI services disabled for data privacy compliance."
```

## Docker Configuration

### Docker Compose
```yaml
version: '3.8'
services:
  inference:
    image: roboflow/inference-server
    environment:
      - WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true
      - WORKFLOW_DISABLED_BLOCK_TYPES=sink,model
      - WORKFLOW_DISABLED_BLOCK_PATTERNS=webhook,openai
      - WORKFLOW_DISABLE_REASON=External services disabled in this environment
```

### Dockerfile
```dockerfile
FROM roboflow/inference-server

# Disable expensive operations
ENV WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true
ENV WORKFLOW_DISABLED_BLOCK_TYPES=model
ENV WORKFLOW_DISABLED_BLOCK_PATTERNS=gpt-4,claude
ENV WORKFLOW_DISABLE_REASON="Premium models disabled. Use standard models instead."
```

### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: workflow-config
data:
  WORKFLOW_SELECTIVE_BLOCKS_DISABLE: "true"
  WORKFLOW_DISABLED_BLOCK_TYPES: "sink"
  WORKFLOW_DISABLED_BLOCK_PATTERNS: "webhook,database"
  WORKFLOW_DISABLE_REASON: "Sinks disabled in staging environment"
```

## Error Messages
When a workflow is rejected, users receive clear, actionable error messages:

```json
{
  "error": "WorkflowDefinitionError",
  "message": "Block type 'roboflow_core/webhook_sink@v1' (category: sink) is not allowed. Blocks of type 'sink' are disabled. External services disabled in this environment.",
  "context": "workflow_compilation | block_validation"
}
```

## Block Categories Reference

### Common Block Types
- **`sink`**: Output blocks that send data externally (webhooks, emails, databases)
- **`model`**: AI/ML models including foundation models and custom models
- **`transformation`**: Data transformation and manipulation blocks
- **`visualization`**: Blocks that generate visual outputs
- **`analytics`**: Blocks that perform analysis and metrics
- **`flow_control`**: Blocks that control workflow execution flow

### Common Block Patterns
- **Foundation Models**: `openai`, `anthropic_claude`, `google_gemini`, `stability_ai`
- **Communication**: `webhook`, `email`, `slack`, `twilio`, `mqtt`
- **Databases**: `sql_server`, `mongodb`, `postgresql`
- **File Systems**: `local_file`, `s3`, `azure_blob`
- **Custom Code**: `python_script`, `custom_function`

## Testing Your Configuration

### Basic Test Script
```python
import os
import json

# Set configuration
os.environ["WORKFLOW_SELECTIVE_BLOCKS_DISABLE"] = "true"
os.environ["WORKFLOW_DISABLED_BLOCK_TYPES"] = "sink"

# Test workflow
workflow = {
    "version": "1.0",
    "steps": [
        {"type": "roboflow_core/webhook_sink@v1", "name": "webhook", ...}
    ]
}

# This should raise a WorkflowDefinitionError
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow
try:
    compile_workflow(workflow, {})
except WorkflowDefinitionError as e:
    print(f"Blocked as expected: {e.public_message}")
```

## Best Practices

### 1. Environment-Specific Configuration
- **Development**: Disable sinks and expensive models
- **Staging**: Allow most blocks but disable production sinks
- **Production**: Minimal restrictions, audit logging enabled

### 2. Clear Communication
- Always set meaningful `WORKFLOW_DISABLE_REASON` messages
- Document which blocks are disabled and why
- Provide alternative workflows when blocks are disabled

### 3. Gradual Rollout
- Start with a few disabled patterns
- Monitor rejected workflows
- Adjust configuration based on usage patterns

### 4. Version Control
- Track configuration changes in version control
- Use infrastructure-as-code for consistency
- Document configuration in README files

## Troubleshooting

### Workflow Unexpectedly Rejected
```bash
# Check current configuration
echo "Selective disable: $WORKFLOW_SELECTIVE_BLOCKS_DISABLE"
echo "Disabled types: $WORKFLOW_DISABLED_BLOCK_TYPES"
echo "Disabled patterns: $WORKFLOW_DISABLED_BLOCK_PATTERNS"

# Temporarily disable filtering to test
export WORKFLOW_SELECTIVE_BLOCKS_DISABLE=false
```

### Workflow Not Rejected When Expected
1. Verify `WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true`
2. Check pattern matching (case-insensitive substring)
3. Verify block type in manifest matches configuration

### Performance Considerations
- Validation occurs during compilation only
- Compiled workflows are cached
- No runtime performance impact
- Minimal overhead (<1ms per workflow)

## Migration Guide

### From Mirroring Mode to Selective Blocks
```bash
# Old configuration
export WORKFLOW_MIRRORING_MODE=true
export WORKFLOW_BLOCKED_BLOCK_TYPES="sink,model"
export WORKFLOW_BLOCKED_BLOCK_PATTERNS="webhook,openai"

# New configuration (backward compatible)
export WORKFLOW_SELECTIVE_BLOCKS_DISABLE=true
export WORKFLOW_DISABLED_BLOCK_TYPES="sink,model"
export WORKFLOW_DISABLED_BLOCK_PATTERNS="webhook,openai"
export WORKFLOW_DISABLE_REASON="Blocks disabled for testing"
```

## Advanced Configuration

### Dynamic Configuration
```python
# Load configuration from external source
import requests
config = requests.get("https://config.example.com/workflow-blocks").json()
os.environ["WORKFLOW_DISABLED_BLOCK_PATTERNS"] = ",".join(config["disabled"])
```

### Conditional Logic
```bash
# Disable based on time of day (maintenance window)
HOUR=$(date +%H)
if [ $HOUR -ge 2 ] && [ $HOUR -le 4 ]; then
  export WORKFLOW_DISABLED_BLOCK_TYPES="sink"
  export WORKFLOW_DISABLE_REASON="Sinks disabled during maintenance window (2-4 AM)"
fi
```

## Future Enhancements
- Per-user or per-API-key configuration
- Allowlist in addition to blocklist
- Time-based rules (e.g., disable during business hours)
- Rate limiting instead of blocking
- Webhook for approval workflow
- Admin UI for configuration management
- Metrics and monitoring dashboard