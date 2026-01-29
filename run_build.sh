#!/bin/bash
export PYTHONPATH=.
echo "Build started at $(date)"
python3 -u src/builder.py md_vault
echo "Build finished at $(date)"
