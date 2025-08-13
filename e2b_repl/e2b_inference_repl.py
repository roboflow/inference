#!/usr/bin/env python3
"""
E2B Sandbox REPL for Roboflow Inference Custom Python Blocks

This REPL connects to the E2B sandbox with the Inference template and provides
an interactive Python environment for testing Custom Python Block code.

Usage:
    python3 e2b_inference_repl.py              # Create new sandbox
    python3 e2b_inference_repl.py <sandbox_id> # Connect to existing sandbox

Example Session:
    >>> import numpy as np
    >>> import supervision as sv
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> print(f"Mean: {arr.mean()}")
    Mean: 3.0
"""

import os
import sys
import time
import traceback
from typing import Optional

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

class InferenceREPL:
    def __init__(self, sandbox_id: Optional[str] = None):
        self.sandbox = None
        self.sandbox_id = sandbox_id
        self.show_timing = False  # Toggle for execution timing
        
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
        """Set up the Python environment in the sandbox."""
        print("🔧 Initializing environment...")
        
        # Create initialization script
        init_script = '''
import sys
sys.path.insert(0, '/app')

# Pre-import common modules for faster execution
import numpy as np
import supervision as sv
import cv2

# Import inference modules
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult

print("Environment initialized successfully!")
print("Available modules: numpy, supervision, cv2, inference")
print("sys.path includes /app for inference modules")
'''
        
        # Write and run initialization
        write_start = time.time()
        self.sandbox.files.write("/home/user/init_env.py", init_script)
        write_time = time.time() - write_start
        
        exec_start = time.time()
        result = self.sandbox.commands.run("python3 /home/user/init_env.py")
        exec_time = time.time() - exec_start
        
        if result.exit_code == 0:
            print(f"✅ Environment ready!")
            print(f"   (File write: {write_time:.3f}s, Script execution: {exec_time:.3f}s)")
            print("\nAvailable packages:")
            print("  - numpy, supervision, opencv (cv2)")
            print("  - inference modules from /app")
            print("\nREPL Commands:")
            print("  !help    - Show this help")
            print("  !info    - Show sandbox info")
            print("  !timing  - Toggle execution timing")
            print("  !ls      - List files")
            print("  !exit    - Exit REPL")
        else:
            print(f"⚠️  Initialization warning: {result.stderr}")
    
    def execute_python(self, code: str, show_timing: bool = False) -> str:
        """Execute Python code in the sandbox."""
        # Create a wrapper that includes sys.path setup
        wrapped_code = f'''
import sys
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

{code}
'''
        
        # Write code to file and execute
        start_time = time.time()
        self.sandbox.files.write("/home/user/exec.py", wrapped_code)
        result = self.sandbox.commands.run("python3 /home/user/exec.py")
        exec_time = time.time() - start_time
        
        # Combine output
        output = []
        if result.stdout:
            output.append(result.stdout.rstrip())
        if result.stderr and "Matplotlib is building" not in result.stderr:
            # Filter out matplotlib font cache warnings
            stderr_lines = [l for l in result.stderr.split('\n') 
                           if l and "Matplotlib" not in l]
            if stderr_lines:
                output.append("[stderr] " + '\n'.join(stderr_lines))
        
        if show_timing:
            output.append(f"[Execution time: {exec_time:.3f}s]")
        
        return '\n'.join(output) if output else ""
    
    def run(self):
        """Run the interactive REPL."""
        print("\n" + "="*60)
        print("🐍 Inference Custom Python Blocks REPL")
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
                        print("👋 Goodbye!")
                        break
                    elif line == '!help':
                        print(__doc__)
                    elif line == '!info':
                        print(f"\nSandbox ID: {self.sandbox_id}")
                        print(f"Template: {INFERENCE_TEMPLATE}")
                        print(f"API Key: {E2B_API_KEY[:20]}...")
                        print(f"Show timing: {'ON' if self.show_timing else 'OFF'}")
                    elif line == '!timing':
                        self.show_timing = not self.show_timing
                        print(f"⏱️  Execution timing: {'ON' if self.show_timing else 'OFF'}")
                    elif line.startswith('!ls'):
                        path = line[3:].strip() or '/app'
                        result = self.sandbox.commands.run(f"ls -la {path}")
                        print(result.stdout if result.exit_code == 0 else result.stderr)
                    else:
                        print(f"Unknown command: {line}")
                        print("Commands: !help, !info, !timing, !ls, !exit")
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inference E2B REPL')
    parser.add_argument('sandbox_id', nargs='?', help='Existing sandbox ID')
    args = parser.parse_args()
    
    repl = InferenceREPL(sandbox_id=args.sandbox_id)
    try:
        repl.connect()
        repl.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        if repl.sandbox_id:
            print(f"\nℹ️  Sandbox {repl.sandbox_id} still running")
            print(f"   Reconnect with: python3 {sys.argv[0]} {repl.sandbox_id}")

if __name__ == "__main__":
    main()
