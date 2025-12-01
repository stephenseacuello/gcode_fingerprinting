#!/usr/bin/env python3
"""
Wrapper script to run train_multihead.py with correct environment.
This ensures W&B sweeps use the virtual environment Python and proper settings.
"""
import os
import sys
import subprocess
from pathlib import Path

# Change to project root directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Set environment variables
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTHONPATH'] = f"src:{os.environ.get('PYTHONPATH', '')}"

# Get the venv Python interpreter
venv_python = project_root / '.venv' / 'bin' / 'python'

# Run train_multihead.py with all arguments passed through
train_script = project_root / 'scripts' / 'train_multihead.py'
cmd = [str(venv_python), str(train_script), '--use-wandb'] + sys.argv[1:]

# Execute and forward exit code
sys.exit(subprocess.call(cmd))
