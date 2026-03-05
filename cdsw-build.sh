#!/bin/bash
# CAI/CML build script — runs when the project environment is built
set -e

pip install --upgrade pip
pip install -r requirements.txt

echo "QA Spark CodeAgent environment ready."
