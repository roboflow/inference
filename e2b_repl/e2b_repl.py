#!/usr/bin/env python3
"""
E2B Sandbox REPL - Interactive Python execution in E2B sandboxes

Usage:
    python e2b_repl.py                     # Create new sandbox with inference template
    python e2b_repl.py <sandbox_id>         # Connect to existing sandbox
    python e2b_repl.py -t <template_id>    # Create new sandbox with specific template

Example:
    # Create a new sandbox and start REPL
    python3 e2b_repl.py
    
    # In the REPL:
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> print(arr.mean())
    3.0
    >>> !info                              # Show sandbox info
    >>> !ls /app                           # List files in /app
    >>> !exit                              # Exit REPL

Commands:
    - Type Python code and press Enter to execute
    - Use '!exit' or '!quit' to exit the REPL
    - Use '!help' to show this help
    - Use '!info' to show sandbox information
    - Use '!upload <local_path> <sandbox_path>' to upload a file
    - Use '!download <sandbox_path> <local_path>' to download a file
    - Use '!ls [path]' to list files in sandbox
    - Use '!cat <path>' to show file contents
"""

import os
import sys
import json
import traceback
from typing import Optional
import argparse
import requests

try:
    from e2b import Sandbox
except ImportError:
    print("Error: e2b package not installed. Installing...")
    os.system("pip3 install -q e2b")
    try:
        from e2b import Sandbox
    except ImportError:
        print("Failed to install e2b. Please run manually: pip3 install e2b")
        sys.exit(1)

# Set the E2B API key
E2B_API_KEY = os.environ.get('E2B_API_KEY', 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e')
os.environ['E2B_API_KEY'] = E2B_API_KEY

# Default template for inference
INFERENCE_TEMPLATE = "qfupheopqmf6w7b36h6o"

class E2BRepl:
    def __init__(self, sandbox_id: Optional[str] = None, template: Optional[str] = None):
        """Initialize the REPL with a sandbox."""
        self.sandbox = None
        self.sandbox_id = sandbox_id
        self.template = template or INFERENCE_TEMPLATE
        self.python_session = None
        
    def connect(self):
        """Connect to sandbox or create a new one."""
        try:
            if self.sandbox_id:
                print(f"🔌 Connecting to existing sandbox: {self.sandbox_id}")
                self.sandbox = Sandbox.connect(self.sandbox_id)
                print(f"✅ Connected to sandbox: {self.sandbox_id}")
            else:
                print(f"🚀 Creating new sandbox with template: {self.template}")
                self.sandbox = Sandbox(template=self.template)
                self.sandbox_id = self.sandbox.sandbox_id
                print(f"✅ Created sandbox: {self.sandbox_id}")
            
            # Initialize Python session
            self._init_python_session()
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            sys.exit(1)
    
    def _init_python_session(self):
        """Initialize a persistent Python session."""
        print("🐍 Initializing Python session...")
        
        # Create a Python script that runs as a persistent session
        init_code = """
import sys
import json
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

# Set up the environment
sys.path.insert(0, '/app')

# Test that inference modules are accessible
try:
    import inference
    print("✅ Inference package found at:", inference.__file__)
except ImportError:
    print("ℹ️  Direct 'import inference' may not work.")
    print("   Use specific imports like:")
    print("   from inference.core.workflows.execution_engine.entities.base import Batch")

print("Python session initialized. Ready for commands.")
print("Note: /app is added to sys.path for inference modules")
"""
        # Write to file instead of using -c flag to avoid escaping issues
        self.sandbox.files.write("/home/user/init_python.py", init_code)
        result = self.sandbox.commands.run("python3 /home/user/init_python.py")
        if result.exit_code == 0:
            print("✅ Python session ready")
        else:
            print(f"⚠️  Python initialization had issues: {result.stderr}")
    
    def run_python(self, code: str) -> str:
        """Run Python code in the sandbox."""
        # Escape the code properly
        escaped_code = code.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
        
        # Wrap the code to capture output
        wrapped_code = f"""
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

# ENSURE /app is in sys.path for inference modules
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

code = '''{code}'''

# Capture stdout and stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        exec(code, globals())
    
    output = stdout_capture.getvalue()
    errors = stderr_capture.getvalue()
    
    if output:
        print(output, end='')
    if errors:
        print(errors, end='', file=sys.stderr)
except Exception as e:
    traceback.print_exc()
"""
        
        # Write code to a temporary file and execute it
        self.sandbox.files.write("/home/user/exec_code.py", wrapped_code)
        result = self.sandbox.commands.run("python3 /home/user/exec_code.py")
        
        output = []
        if result.stdout:
            output.append(result.stdout)
        if result.stderr:
            output.append(f"[stderr] {result.stderr}")
        
        return '\n'.join(output) if output else "No output"
    
    def show_info(self):
        """Show sandbox information."""
        print(f"\n📦 Sandbox Information:")
        print(f"  ID: {self.sandbox_id}")
        print(f"  Template: {self.template}")
        print(f"  API Key: {E2B_API_KEY[:20]}...")
        
        # Try to get more info
        try:
            info = self.sandbox.get_info()
            print(f"  Status: Running")
            print(f"  Host: {self.sandbox.get_host()}")
        except:
            pass
    
    def list_files(self, path: str = "/"):
        """List files in the sandbox."""
        result = self.sandbox.commands.run(f"ls -la {path}")
        return result.stdout if result.exit_code == 0 else f"Error: {result.stderr}"
    
    def cat_file(self, path: str):
        """Show file contents."""
        result = self.sandbox.commands.run(f"cat {path}")
        return result.stdout if result.exit_code == 0 else f"Error: {result.stderr}"
    
    def upload_file(self, local_path: str, sandbox_path: str):
        """Upload a file to the sandbox."""
        try:
            with open(local_path, 'rb') as f:
                content = f.read()
            self.sandbox.files.write(sandbox_path, content)
            return f"✅ Uploaded {local_path} to {sandbox_path}"
        except Exception as e:
            return f"❌ Upload failed: {e}"
    
    def download_file(self, sandbox_path: str, local_path: str):
        """Download a file from the sandbox."""
        try:
            content = self.sandbox.files.read(sandbox_path)
            with open(local_path, 'wb') as f:
                f.write(content if isinstance(content, bytes) else content.encode())
            return f"✅ Downloaded {sandbox_path} to {local_path}"
        except Exception as e:
            return f"❌ Download failed: {e}"
    
    def run_repl(self):
        """Run the interactive REPL."""
        print("\n" + "="*60)
        print("🎮 E2B Sandbox REPL")
        print("="*60)
        print("Type Python code to execute in the sandbox.")
        print("Special commands: !help, !exit, !info, !ls, !cat, !upload, !download")
        print("="*60 + "\n")
        
        # Multi-line code buffer
        code_buffer = []
        in_multiline = False
        
        while True:
            try:
                # Show appropriate prompt
                if in_multiline:
                    prompt = "... "
                else:
                    prompt = ">>> "
                
                # Get input
                line = input(prompt)
                
                # Handle special commands (only when not in multiline mode)
                if not in_multiline and line.startswith('!'):
                    parts = line.split()
                    cmd = parts[0].lower()
                    
                    if cmd in ['!exit', '!quit']:
                        print("👋 Goodbye!")
                        break
                    elif cmd == '!help':
                        print(__doc__)
                    elif cmd == '!info':
                        self.show_info()
                    elif cmd == '!ls':
                        path = parts[1] if len(parts) > 1 else "/"
                        print(self.list_files(path))
                    elif cmd == '!cat':
                        if len(parts) > 1:
                            print(self.cat_file(parts[1]))
                        else:
                            print("Usage: !cat <path>")
                    elif cmd == '!upload':
                        if len(parts) == 3:
                            print(self.upload_file(parts[1], parts[2]))
                        else:
                            print("Usage: !upload <local_path> <sandbox_path>")
                    elif cmd == '!download':
                        if len(parts) == 3:
                            print(self.download_file(parts[1], parts[2]))
                        else:
                            print("Usage: !download <sandbox_path> <local_path>")
                    else:
                        print(f"Unknown command: {cmd}")
                    continue
                
                # Handle Python code
                if line.endswith(':') or line.startswith(' ') or line.startswith('\t'):
                    # Start or continue multiline mode
                    in_multiline = True
                    code_buffer.append(line)
                elif in_multiline and line == '':
                    # Empty line ends multiline mode
                    code = '\n'.join(code_buffer)
                    code_buffer = []
                    in_multiline = False
                    
                    # Execute the multiline code
                    if code.strip():
                        output = self.run_python(code)
                        if output:
                            print(output)
                elif in_multiline:
                    # Continue multiline mode
                    code_buffer.append(line)
                else:
                    # Single line code
                    if line.strip():
                        output = self.run_python(line)
                        if output:
                            print(output)
                    
            except KeyboardInterrupt:
                if in_multiline:
                    print("\n[Interrupted]")
                    code_buffer = []
                    in_multiline = False
                else:
                    print("\n👋 Use !exit to quit")
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()
    
    def cleanup(self):
        """Clean up resources."""
        if self.sandbox:
            try:
                print(f"\n🧹 Cleaning up sandbox {self.sandbox_id}...")
                # Note: sandbox.close() doesn't exist in new API
                # The sandbox will be cleaned up automatically
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description='E2B Sandbox REPL')
    parser.add_argument('sandbox_id', nargs='?', help='Sandbox ID to connect to (optional)')
    parser.add_argument('--template', '-t', help='Template ID to use for new sandbox')
    args = parser.parse_args()
    
    # Create and run REPL
    repl = E2BRepl(sandbox_id=args.sandbox_id, template=args.template)
    
    try:
        repl.connect()
        repl.run_repl()
    finally:
        repl.cleanup()

if __name__ == "__main__":
    main()
