#!/usr/bin/env bash
# Активация окружения llmbench с корректным LD_LIBRARY_PATH для vLLM
VENV_DIR="${VENV_DIR:-/opt/llmbench/.venv}"
CU13_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cu13/lib"

if [ -d "$CU13_LIB" ]; then
    export LD_LIBRARY_PATH="$CU13_LIB:${LD_LIBRARY_PATH:-}"
fi

# Активируем venv
source "$VENV_DIR/bin/activate"

echo "[llmbench] venv activated, LD_LIBRARY_PATH includes CUDA 13 runtime"
