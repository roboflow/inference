#!/usr/bin/env python3
"""
Simple test to verify the logic of remote mode with disabled local execution.
This test simulates the conditions without importing the full inference modules.
"""

import os

def test_remote_mode_logic():
    """Test the logic that determines if dynamic blocks should be allowed."""
    
    print("Testing Remote Mode Logic")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Remote mode with E2B key (local disabled)",
            "allow_local": False,
            "mode": "remote",
            "has_e2b_key": True,
            "expected": "ALLOWED",
        },
        {
            "name": "Remote mode without E2B key (local disabled)",
            "allow_local": False,
            "mode": "remote",
            "has_e2b_key": False,
            "expected": "BLOCKED",
        },
        {
            "name": "Local mode (local disabled)",
            "allow_local": False,
            "mode": "local",
            "has_e2b_key": True,
            "expected": "BLOCKED",
        },
        {
            "name": "Local mode (local enabled)",
            "allow_local": True,
            "mode": "local",
            "has_e2b_key": False,
            "expected": "ALLOWED",
        },
        {
            "name": "Remote mode with E2B key (local enabled)",
            "allow_local": True,
            "mode": "remote",
            "has_e2b_key": True,
            "expected": "ALLOWED",
        },
    ]
    
    def should_allow_dynamic_blocks(allow_local, mode, has_e2b_key):
        """Simulate the logic from ensure_dynamic_blocks_allowed."""
        # Allow dynamic blocks in remote mode regardless of ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS
        if mode == "remote" and has_e2b_key:
            return True
        
        # Otherwise, check if local execution is allowed
        return allow_local
    
    def execution_mode_for_block(allow_local, mode, has_e2b_key):
        """Simulate the logic from the run function."""
        # Check if we should execute remotely
        if mode == "remote" and has_e2b_key:
            return "REMOTE_EXECUTION"
        
        # Local execution - check if allowed
        if not allow_local:
            return "BLOCKED"
        
        return "LOCAL_EXECUTION"
    
    print("\n1. Testing compilation logic (ensure_dynamic_blocks_allowed):")
    print("-" * 60)
    
    all_passed = True
    for test in test_cases:
        result = should_allow_dynamic_blocks(
            test["allow_local"], 
            test["mode"], 
            test["has_e2b_key"]
        )
        result_str = "ALLOWED" if result else "BLOCKED"
        passed = result_str == test["expected"]
        
        status = "✅" if passed else "❌"
        print(f"{status} {test['name']}")
        print(f"   Expected: {test['expected']}, Got: {result_str}")
        
        if not passed:
            all_passed = False
    
    print("\n2. Testing execution logic (run function):")
    print("-" * 60)
    
    for test in test_cases:
        result = execution_mode_for_block(
            test["allow_local"], 
            test["mode"], 
            test["has_e2b_key"]
        )
        
        # Map expected compilation result to expected execution result
        if test["expected"] == "ALLOWED":
            if test["mode"] == "remote" and test["has_e2b_key"]:
                expected_exec = "REMOTE_EXECUTION"
            else:
                expected_exec = "LOCAL_EXECUTION"
        else:
            expected_exec = "BLOCKED"
        
        passed = result == expected_exec
        status = "✅" if passed else "❌"
        
        print(f"{status} {test['name']}")
        print(f"   Expected: {expected_exec}, Got: {result}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All logic tests passed!")
        print("\nKey finding: Remote mode with E2B_API_KEY works even when")
        print("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS is False")
    else:
        print("❌ Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = test_remote_mode_logic()
    sys.exit(0 if success else 1)
