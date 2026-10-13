#!/usr/bin/env bash
# Отдельный venv для VLMEvalKit. Не трогает .venv llmbench / lm_eval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-vlmeval"
KIT="$ROOT/third_party/VLMEvalKit"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel
if [[ -f "$KIT/requirements.txt" ]]; then
  pip install -r "$KIT/requirements.txt"
fi
pip install -e "$KIT" || pip install -e "$KIT" --no-deps
"$VENV/bin/python" -c "import vlmeval; print('vlmeval ok', getattr(vlmeval, '__file__', ''))"
echo "READY: export LLMBENCH_VLMEVAL_PYTHON=$VENV/bin/python"
