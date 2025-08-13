#!/usr/bin/env python3
"""
E2B Sandbox REPL for Roboflow Inference Custom Python Blocks (Persistent Context)

This REPL maintains context between commands by accumulating code history.
Variables and imports persist across commands in the same session.

Usage:
    python3 e2b_inference_repl_persistent.py              # Create new sandbox
    python3 e2b_inference_repl_persistent.py <sandbox_id> # Connect to existing sandbox

Example Session:
    >>> import random
    >>> x = random.random()
    >>> print(x)  # x is still available!
    0.123456
"""

import os
import sys
import time
import traceback
from typing import Optional, List

# Auto-install e2b if needed
try:
    from e2b import Sandbox
except ImportError:
    print("Installing e2b package...")
    os.system("pip3 install -q e2b")
    from e2b import Sandbox

# Configuration
E2B_API_KEY = os.environ.get('E2B_API_KEY', 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e')
os.environ['E2B_API_KEY'] = E2B_API_KEY
INFERENCE_TEMPLATE = "qfupheopqmf6w7b36h6o"

class PersistentInferenceREPL:
    def __init__(self, sandbox_id: Optional[str] = None):
        self.sandbox = None
        self.sandbox_id = sandbox_id
        self.show_timing = False
        self.session_history: List[str] = []
        self.globals_dict = {}
        
    def connect(self):
        """Connect to or create sandbox."""
        try:
            if self.sandbox_id:
                print(f"🔌 Connecting to sandbox: {self.sandbox_id}")
                start_time = time.time()
                self.sandbox = Sandbox.connect(self.sandbox_id)
                connect_time = time.time() - start_time
                print(f"✅ Connected in {connect_time:.2f}s")
            else:
                print(f"🚀 Creating new sandbox...")
                start_time = time.time()
                self.sandbox = Sandbox(template=INFERENCE_TEMPLATE)
                create_time = time.time() - start_time
                self.sandbox_id = self.sandbox.sandbox_id
                print(f"✅ Sandbox created: {self.sandbox_id}")
                print(f"⏱️  Sandbox creation took {create_time:.2f}s")
                
            # Initialize the environment
            init_start = time.time()
            self._initialize_environment()
            init_time = time.time() - init_start
            print(f"⏱️  Environment initialization took {init_time:.2f}s")
            
            if not self.sandbox_id:
                total_time = time.time() - start_time
                print(f"⏱️  Total startup time: {total_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            sys.exit(1)
    
    def _initialize_environment(self):
        """Initialize the Python environment with base imports."""
        print("🔧 Initializing persistent environment...")
        
        # Base imports that are always available
        base_imports = [
            "import sys",
            "sys.path.insert(0, '/app')",
            "",
            "# Pre-imported packages",
            "import numpy as np",
            "import supervision as sv",
            "import cv2",
            "",
            "# Inference modules",
            "from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData",
            "from inference.core.workflows.prototypes.block import BlockResult",
        ]
        
        # Initialize session with base imports
        self.session_history = base_imports.copy()
        
        # Test the environment
        test_code = '\n'.join(self.session_history) + '\nprint("Environment ready")'
        self.sandbox.files.write("/home/user/init_test.py", test_code)
        result = self.sandbox.commands.run("python3 /home/user/init_test.py")
        
        if result.exit_code == 0:
            print("✅ Persistent environment ready!")
            print("\nPre-imported packages:")
            print("  - numpy as np")
            print("  - supervision as sv")
            print("  - cv2")
            print("  - inference modules (Batch, WorkflowImageData, BlockResult)")
            print("\n💡 Variables and imports persist between commands!")
            print("\nREPL Commands:")
            print("  !help    - Show this help")
            print("  !info    - Show sandbox info")
            print("  !vars    - Show defined variables")
            print("  !history - Show code history")
            print("  !timing  - Toggle execution timing")
            print("  !reset   - Reset session (clear variables)")
            print("  !ls      - List files")
            print("  !exit    - Exit REPL")
        else:
            print(f"⚠️  Initialization warning: {result.stderr}")
    
    def execute_python(self, code: str, show_timing: bool = False) -> str:
        """Execute Python code with persistent context."""
        start_time = time.time()
        
        # Build the full script with all history plus new code
        full_script = '\n'.join(self.session_history)
        
        # Special handling for expressions vs statements
        # Try to detect if this is just an expression that should be printed
        is_expression = False
        try:
            compile(code, '<string>', 'eval')
            is_expression = True
        except SyntaxError:
            pass
        
        if is_expression and not code.startswith('print'):
            # Wrap expression to print its value
            execution_code = f"\n__repl_result__ = {code}\nif __repl_result__ is not None:\n    print(__repl_result__)"
        else:
            execution_code = f"\n{code}"
        
        # Write complete script
        test_script = full_script + execution_code
        self.sandbox.files.write("/home/user/exec.py", test_script)
        
        # Execute
        result = self.sandbox.commands.run("python3 /home/user/exec.py")
        exec_time = time.time() - start_time
        
        # If successful, add to history
        if result.exit_code == 0:
            # Only add the actual code to history, not the print wrapper
            self.session_history.append(code)
        
        # Process output
        output = []
        if result.stdout:
            output.append(result.stdout.rstrip())
        if result.stderr:
            # Filter out common warnings
            stderr_lines = [l for l in result.stderr.split('\n') 
                           if l and not any(ignore in l for ignore in 
                           ["Matplotlib is building", "font cache", "WARNING"])]
            if stderr_lines:
                # Show error but don't add to history
                error_msg = '\n'.join(stderr_lines)
                if "Traceback" in error_msg:
                    # Full traceback
                    output.append(error_msg)
                else:
                    output.append(f"[stderr] {error_msg}")
        
        if show_timing:
            output.append(f"[Execution time: {exec_time:.3f}s]")
        
        return '\n'.join(output) if output else ""
    
    def show_variables(self):
        """Show currently defined variables."""
        check_code = """
import types
user_vars = []
for name, value in list(globals().items()):
    if not name.startswith('_') and not isinstance(value, types.ModuleType):
        if not name in ['In', 'Out', 'exit', 'quit', 'get_ipython']:
            try:
                type_name = type(value).__name__
                if hasattr(value, '__len__') and not isinstance(value, str):
                    user_vars.append(f"  {name}: {type_name} (len={len(value)})")
                else:
                    val_str = str(value)
                    if len(val_str) > 50:
                        val_str = val_str[:47] + "..."
                    user_vars.append(f"  {name}: {type_name} = {val_str}")
            except:
                user_vars.append(f"  {name}: {type(value).__name__}")

if user_vars:
    print("Defined variables:")
    for var in sorted(user_vars):
        print(var)
else:
    print("No user variables defined")
"""
        # Execute with current history
        full_script = '\n'.join(self.session_history) + '\n' + check_code
        self.sandbox.files.write("/home/user/show_vars.py", full_script)
        result = self.sandbox.commands.run("python3 /home/user/show_vars.py")
        return result.stdout if result.exit_code == 0 else "Error checking variables"
    
    def show_history(self):
        """Show code history."""
        # Skip the base imports
        user_code = [line for line in self.session_history[11:] if line.strip()]
        if user_code:
            print("Session history:")
            for i, line in enumerate(user_code, 1):
                print(f"  {i}: {line}")
        else:
            print("No commands in history yet")
    
    def terminate_sandbox(self):
        """Terminate the sandbox to free resources."""
        if self.sandbox:
            try:
                # The E2B SDK uses .kill() to terminate sandboxes
                self.sandbox.kill()
                print(f"✅ Sandbox {self.sandbox_id} terminated")
            except Exception as e:
                print(f"⚠️  Error terminating sandbox: {e}")
    
    def reset_session(self):
        """Reset the session to initial state."""
        print("🔄 Resetting session...")
        base_imports = [
            "import sys",
            "sys.path.insert(0, '/app')",
            "",
            "# Pre-imported packages",
            "import numpy as np",
            "import supervision as sv",
            "import cv2",
            "",
            "# Inference modules",
            "from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData",
            "from inference.core.workflows.prototypes.block import BlockResult",
        ]
        self.session_history = base_imports.copy()
        print("✅ Session reset - variables cleared, base imports restored")
    
    def run(self):
        """Run the interactive REPL."""
        print("\n" + "="*60)
        print("🐍 Inference Custom Python Blocks REPL (Persistent Context)")
        print("="*60 + "\n")
        
        # Buffer for multi-line input
        code_buffer = []
        in_multiline = False
        
        while True:
            try:
                # Choose prompt
                prompt = "... " if in_multiline else ">>> "
                line = input(prompt)
                
                # Handle special commands
                if not in_multiline and line.startswith('!'):
                    if line in ['!exit', '!quit']:
                        print("🛑 Terminating sandbox...")
                        self.terminate_sandbox()
                        print("👋 Goodbye!")
                        break
                    elif line == '!help':
                        print(__doc__)
                    elif line == '!info':
                        print(f"\nSandbox ID: {self.sandbox_id}")
                        print(f"Template: {INFERENCE_TEMPLATE}")
                        print(f"Show timing: {'ON' if self.show_timing else 'OFF'}")
                        user_lines = len([l for l in self.session_history[11:] if l.strip()])
                        print(f"Commands in history: {user_lines}")
                    elif line == '!vars':
                        print(self.show_variables())
                    elif line == '!history':
                        self.show_history()
                    elif line == '!timing':
                        self.show_timing = not self.show_timing
                        print(f"⏱️  Execution timing: {'ON' if self.show_timing else 'OFF'}")
                    elif line == '!reset':
                        self.reset_session()
                    elif line.startswith('!ls'):
                        path = line[3:].strip() or '/app'
                        result = self.sandbox.commands.run(f"ls -la {path}")
                        print(result.stdout if result.exit_code == 0 else result.stderr)
                    else:
                        print(f"Unknown command: {line}")
                        print("Commands: !help, !info, !vars, !history, !timing, !reset, !ls, !exit")
                    continue
                
                # Handle Python code
                if line.endswith(':'):
                    # Start multi-line mode
                    in_multiline = True
                    code_buffer.append(line)
                elif in_multiline:
                    if line == '':
                        # Empty line ends multi-line
                        code = '\n'.join(code_buffer)
                        code_buffer = []
                        in_multiline = False
                        
                        if code.strip():
                            output = self.execute_python(code, show_timing=self.show_timing)
                            if output:
                                print(output)
                    else:
                        # Continue multi-line
                        code_buffer.append(line)
                else:
                    # Single line execution
                    if line.strip():
                        output = self.execute_python(line, show_timing=self.show_timing)
                        if output:
                            print(output)
                            
            except KeyboardInterrupt:
                if in_multiline:
                    print("\n[Interrupted]")
                    code_buffer = []
                    in_multiline = False
                else:
                    print("\nUse !exit to quit")
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Persistent Inference E2B REPL')
    parser.add_argument('sandbox_id', nargs='?', help='Existing sandbox ID')
    args = parser.parse_args()
    
    repl = PersistentInferenceREPL(sandbox_id=args.sandbox_id)
    try:
        repl.connect()
        repl.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted - terminating sandbox...")
        repl.terminate_sandbox()
        print("👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        if repl.sandbox_id:
            print(f"ℹ️  Sandbox {repl.sandbox_id} may still be running")
            print(f"   Reconnect with: python3 {sys.argv[0]} {repl.sandbox_id}")

if __name__ == "__main__":
    main()
