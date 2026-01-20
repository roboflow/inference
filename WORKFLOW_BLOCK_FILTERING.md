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
When configured, disabled blocks are prevented from being loaded into the system at all. This means:
- Disabled blocks won't appear in the UI or API discovery endpoints
- Workflows attempting to use disabled blocks will fail with "unknown block type" errors
- The filtering happens at load time, before blocks are registered with the workflow engine
- Filtering is automatically enabled when either `WORKFLOW_DISABLED_BLOCK_TYPES` or `WORKFLOW_DISABLED_BLOCK_PATTERNS` is set

## Configuration

### Environment Variables

#### `WORKFLOW_DISABLED_BLOCK_TYPES`
- **Type**: Comma-separated string
- **Default**: Empty (no types disabled by default)
- **Description**: Block type categories to disable (from block manifest's `block_type` field)
- **Valid Values**: `sink`, `model`, `transformation`, `visualization`, `analytics`, etc.
- **Example**: `export WORKFLOW_DISABLED_BLOCK_TYPES="sink,model"`

#### `WORKFLOW_DISABLED_BLOCK_PATTERNS`
- **Type**: Comma-separated string
- **Default**: Empty (no patterns disabled by default)
- **Description**: Patterns to match in block identifiers (case-insensitive substring match)
- **Example**: `export WORKFLOW_DISABLED_BLOCK_PATTERNS="webhook,openai,anthropic"`

## Usage Scenarios

### 1. Mirroring/Testing Infrastructure
Prevent duplicate side effects when testing with mirrored requests:
```bash
export WORKFLOW_DISABLED_BLOCK_TYPES="sink,model"
export WORKFLOW_DISABLED_BLOCK_PATTERNS="webhook,email,database,openai,anthropic"
```

### 2. Development Environment
Reduce costs and prevent accidental production actions:
```bash
export WORKFLOW_DISABLED_BLOCK_TYPES="sink"
export WORKFLOW_DISABLED_BLOCK_PATTERNS="openai,anthropic,google_gemini,stability_ai"
```

### 3. Cost Control
Disable expensive operations:
```bash
export WORKFLOW_DISABLED_BLOCK_PATTERNS="gpt-4,claude,gemini-pro,stability_ai"
```

### 4. Security-Restricted Environment
Prevent data exfiltration and external communication:
```bash
export WORKFLOW_DISABLED_BLOCK_TYPES="sink"
export WORKFLOW_DISABLED_BLOCK_PATTERNS="webhook,http,email,slack,database"
```

### 5. Compliance Mode
Ensure GDPR/HIPAA compliance:
```bash
export WORKFLOW_DISABLED_BLOCK_PATTERNS="google,openai,anthropic,external"
```

## Docker Configuration

### Docker Compose
```yaml
version: '3.8'
services:
  inference:
    image: roboflow/inference-server
    environment:
      - WORKFLOW_DISABLED_BLOCK_TYPES=sink,model
      - WORKFLOW_DISABLED_BLOCK_PATTERNS=webhook,openai
```

### Dockerfile
```dockerfile
FROM roboflow/inference-server

# Disable expensive operations
ENV WORKFLOW_DISABLED_BLOCK_TYPES=model
ENV WORKFLOW_DISABLED_BLOCK_PATTERNS=gpt-4,claude
```

### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: workflow-config
data:
  WORKFLOW_DISABLED_BLOCK_TYPES: "sink"
  WORKFLOW_DISABLED_BLOCK_PATTERNS: "webhook,database"
```

## Error Messages
When a workflow attempts to use a disabled block, users receive a standard "unknown block type" error:

```json
{
  "error": "WorkflowDefinitionError",
  "message": "Unknown block type: 'roboflow_core/webhook_sink@v1'",
  "context": "workflow_compilation"
}
```

This is intentional - disabled blocks are completely removed from the system, so they appear as if they don't exist.

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

# Set configuration - filtering is automatically enabled when these are set
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
echo "Disabled types: $WORKFLOW_DISABLED_BLOCK_TYPES"
echo "Disabled patterns: $WORKFLOW_DISABLED_BLOCK_PATTERNS"

# Temporarily disable filtering by unsetting the variables
unset WORKFLOW_DISABLED_BLOCK_TYPES
unset WORKFLOW_DISABLED_BLOCK_PATTERNS
```

### Workflow Not Rejected When Expected
1. Verify that `WORKFLOW_DISABLED_BLOCK_TYPES` or `WORKFLOW_DISABLED_BLOCK_PATTERNS` is set
2. Check pattern matching (case-insensitive substring)
3. Verify block type in manifest matches configuration
4. Check that the server/process was restarted after changing environment variables (blocks are loaded once at startup)

### Performance Considerations
- Filtering occurs once at startup during block loading
- No runtime performance impact
- Disabled blocks consume no memory or resources
- API discovery endpoints automatically reflect filtered blocks

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