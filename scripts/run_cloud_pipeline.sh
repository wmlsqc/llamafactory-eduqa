#!/usr/bin/env bash
# Run on an already rented Linux GPU server after bootstrap_cloud.sh.
# Every stage is real; this script has no simulated model or prefilled metrics.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
train_python=.venv-train/bin/python
infer_python=.venv-inference/bin/python
profile="${EDUQA_TRAIN_PROFILE:-4090}"
model_dir="${EDUQA_BASE_MODEL_DIR:-models/Qwen2.5-7B-Instruct}"
server_pid=""
server_manifest=""
for python_path in "$train_python" "$infer_python"; do
  [[ -x "$python_path" ]] || { echo "Missing $python_path; run bootstrap first." >&2; exit 1; }
done
mkdir -p artifacts/vllm
stop_server() {
  if [[ -n "$server_pid" ]]; then
    # The Python wrapper owns a separate vLLM session and waits for all workers.
    # Signal this exact child PID; never kill unrelated GPU processes by name.
    kill -TERM "$server_pid" 2>/dev/null || true
    local result=0
    wait "$server_pid" 2>/dev/null || result=$?
    server_pid=""
    if [[ "$result" != 0 && "$result" != 130 ]]; then
      echo "vLLM wrapper failed during shutdown (exit $result); inspect $server_manifest" >&2
      return 1
    fi
    "$infer_python" - "$server_manifest" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if not report.get('shutdown', {}).get('complete'):
    raise SystemExit('vLLM worker shutdown was not confirmed; refusing to reuse the GPU.')
PY
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_server() {
  local model_path="$1"
  local label="$2"
  # Never accidentally benchmark an unrelated process on the target port.
  "$infer_python" - <<'PY'
import socket
with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(('127.0.0.1', 8000))
PY
  server_manifest="artifacts/vllm/$label.json"
  "$infer_python" -m eduqa start-vllm --model-dir "$model_path" \
    --manifest "artifacts/vllm/$label.json" > "artifacts/vllm/$label.log" 2>&1 &
  server_pid=$!
  for attempt in $(seq 1 180); do
    kill -0 "$server_pid" 2>/dev/null || { echo "vLLM failed; read artifacts/vllm/$label.log" >&2; exit 1; }
    if "$infer_python" - <<'PY'
import os, sys, urllib.request, json
key = os.getenv('EDUQA_BACKEND_API_KEY', '')
headers = {'Authorization': 'Bearer ' + key} if key else {}
try:
    req = urllib.request.Request('http://127.0.0.1:8000/v1/models', headers=headers)
    with urllib.request.urlopen(req, timeout=2) as response:
        models = json.load(response)['data']
    sys.exit(0 if any(m['id'] == 'eduqa' for m in models) else 1)
except Exception:
    sys.exit(1)
PY
    then
      return 0
    fi
    sleep 5
  done
  echo "vLLM readiness timeout; inspect its log." >&2
  exit 1
}

"$train_python" -m eduqa doctor --require-gpu --min-vram-gb 24
"$train_python" - <<'PY'
from pathlib import Path
from eduqa.data import prepare_data, sha256_file
from eduqa.training import _validate_data
import json
output = Path('artifacts/data')
train_source = Path('EDU-QA/data/science_ft_500.jsonl')
test_source = Path('EDU-QA/data/test.jsonl')
if output.exists() and any(output.iterdir()):
    _validate_data(output.resolve())
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    for name, source in [('train', train_source), ('test', test_source)]:
        if manifest['sources'][name]['sha256'] != sha256_file(source):
            raise SystemExit(f'Existing prepared data differs from {source}; inspect or use a fresh checkout.')
    if manifest.get('seed') != 42 or manifest.get('validation_ratio') != 0.1:
        raise SystemExit('Existing prepared data uses a different split; inspect or use a fresh checkout.')
    print('Reusing verified prepared datasets:', manifest['counts'])
else:
    prepare_data(train_source, test_source, output)
PY
"$train_python" -m eduqa download-model --output-dir "$model_dir"
"$train_python" -m eduqa train --profile "$profile" --model "$model_dir"
"$train_python" -m eduqa export --model "$model_dir"

start_server "$model_dir" base
"$infer_python" -m eduqa evaluate --label base --output artifacts/eval/base
stop_server
start_server artifacts/models/eduqa-merged tuned
"$infer_python" -m eduqa evaluate --label tuned --output artifacts/eval/tuned \
  --compare-to artifacts/eval/base/summary.json
"$infer_python" -m eduqa benchmark --requests 20 --concurrency 1 --output artifacts/benchmark/c1.json
"$infer_python" -m eduqa benchmark --requests 20 --concurrency 4 --output artifacts/benchmark/c4.json
stop_server
echo "Completed. Read artifacts/eval/tuned/report.md and artifacts/benchmark/*.json."
echo "To launch the application, follow docs/CLOUD_RUNBOOK.md."
