#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH-}"

CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/base.yaml}"

uv run python -m fundamentals_large_models.train --config "${CONFIG_PATH}"
