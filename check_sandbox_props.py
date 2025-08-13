#!/usr/bin/env python3
"""
Check if E2B Sandbox has any properties that indicate readiness.
"""

def check_sandbox_properties():
    """Check what properties/methods might indicate sandbox readiness."""
    
    from e2b_code_interpreter import Sandbox
    import inspect
    
    print("E2B Sandbox Properties/Methods Analysis")
    print("=" * 60)
    
    # Get all attributes
    all_attrs = dir(Sandbox)
    
    # Look for properties that might indicate state
    print("\nState/Status related attributes:")
    state_attrs = [a for a in all_attrs if any(k in a.lower() for k in ['state', 'status', 'ready', 'running', 'alive', 'active'])]
    for attr in state_attrs:
        print(f"  - {attr}")
    
    # Check for ID/session attributes
    print("\nID/Session attributes:")
    id_attrs = [a for a in all_attrs if any(k in a.lower() for k in ['id', 'session', 'url', 'endpoint'])]
    for attr in id_attrs:
        print(f"  - {attr}")
    
    # Properties (not methods)
    print("\nProperties (not callable):")
    for attr in all_attrs:
        if not attr.startswith('_'):
            try:
                prop = getattr(Sandbox, attr)
                if isinstance(prop, property):
                    print(f"  - {attr} (property)")
            except:
                pass
    
    print("\n" + "=" * 60)
    print("Based on E2B documentation, the sandbox is ready when:")
    print("1. Created successfully (201 response)")
    print("2. Internal services are running (takes a few seconds)")
    print("3. Port 49999 (code interpreter API) is accessible")
    print()
    print("Our fix handles this by:")
    print("- Waiting 3 seconds after creation")
    print("- Testing with simple code execution")
    print("- Retrying with exponential backoff if port not ready")

if __name__ == "__main__":
    check_sandbox_properties()
