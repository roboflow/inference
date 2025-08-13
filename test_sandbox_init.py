#!/usr/bin/env python3
"""
Test E2B Sandbox initialization patterns to find the correct way to wait for sandbox readiness.
"""

def test_sandbox_initialization_pattern():
    """Test different ways to initialize E2B sandbox."""
    
    print("E2B Sandbox Initialization Patterns")
    print("=" * 60)
    
    # Check if Sandbox should be used as a context manager
    from e2b_code_interpreter import Sandbox
    import inspect
    
    # Check if Sandbox has context manager methods
    has_enter = hasattr(Sandbox, '__enter__')
    has_exit = hasattr(Sandbox, '__exit__')
    
    print(f"Has __enter__ method: {has_enter}")
    print(f"Has __exit__ method: {has_exit}")
    
    if has_enter and has_exit:
        print("✅ Sandbox supports context manager (with statement)")
        print("\nRecommended pattern:")
        print("```python")
        print("with Sandbox(api_key=key, template=template) as sandbox:")
        print("    result = sandbox.run_code(code)")
        print("```")
    else:
        print("❌ Sandbox doesn't support context manager")
    
    # Check for wait/ready methods
    sandbox_methods = [m for m in dir(Sandbox) if not m.startswith('_')]
    
    print("\nMethods that might indicate readiness:")
    wait_methods = [m for m in sandbox_methods if any(k in m.lower() for k in ['wait', 'ready', 'connect', 'start', 'open'])]
    for method in wait_methods:
        print(f"  - {method}")
    
    # Check for async methods
    print("\nAsync methods (if any):")
    async_methods = [m for m in sandbox_methods if inspect.iscoroutinefunction(getattr(Sandbox, m, None))]
    for method in async_methods:
        print(f"  - {method}")
    
    # Look for keepalive or health check
    print("\nHealth/keepalive methods:")
    health_methods = [m for m in sandbox_methods if any(k in m.lower() for k in ['health', 'alive', 'ping', 'check'])]
    for method in health_methods:
        print(f"  - {method}")
    
    return True

if __name__ == "__main__":
    test_sandbox_initialization_pattern()
