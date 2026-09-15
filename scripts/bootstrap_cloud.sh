#!/usr/bin/env bash
# Install only the requested environment. This never starts or rents a server.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-app}"
python_bin="${PYTHON_BIN:-python3.11}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python 3.11 is required. Set PYTHON_BIN to its path." >&2
  exit 1
fi
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Cloud environment expects Python 3.11"'
case "$mode" in
  app) env_dir=.venv-app ;;
  train) env_dir=.venv-train ;;
  inference) env_dir=.venv-inference ;;
  *) echo "Usage: bash scripts/bootstrap_cloud.sh [app|train|inference]" >&2; exit 2 ;;
esac
"$python_bin" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
if [[ "$mode" == "train" ]]; then
  "$env_dir/bin/python" -m pip install -r requirements/train.txt -e '.[dev,download]'
elif [[ "$mode" == "inference" ]]; then
  "$env_dir/bin/python" -m pip install -r requirements/inference.txt -e '.[dev,download]'
else
  "$env_dir/bin/python" -m pip install -e '.[dev,download]'
fi
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" -m pip freeze > "$env_dir/resolved-requirements.txt"
echo "Environment ready: source $env_dir/bin/activate"
