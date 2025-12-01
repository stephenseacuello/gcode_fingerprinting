"""
Setup script for G-code fingerprinting project.

NOTE: This file is kept for backward compatibility only.
For the canonical source of dependencies and project configuration,
see pyproject.toml which is the modern standard for Python projects.

To install:
    pip install -e .                    # Base dependencies
    pip install -e ".[ml]"              # With ML tracking (wandb, tensorboard)
    pip install -e ".[dashboard]"       # With Streamlit dashboard
    pip install -e ".[all]"             # All optional dependencies
    pip install -e ".[dev]"             # Development tools
"""
from setuptools import setup

# All configuration is in pyproject.toml
# This minimal setup.py exists for compatibility with tools that don't yet support pyproject.toml
setup()
