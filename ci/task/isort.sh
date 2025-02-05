#!/bin/bash
set -eo pipefail

source ~/.bashrc
micromamba activate ci-lint
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

isort --check-only -j $NUM_THREADS --profile black ./python/
isort --check-only -j $NUM_THREADS --profile black ./tests/python/
