#!/bin/bash
# Wrapper script to run train_sweep.py with correct PYTHONPATH
export PYTHONPATH=src
exec python train_sweep.py "$@"
