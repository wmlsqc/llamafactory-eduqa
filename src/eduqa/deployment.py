"""Cloud preflight, versioned model download and real vLLM process launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment_report() -> dict:
    packages = {}
    for name in ("llamafactory", "torch", "transformers", "peft", "bitsandbytes", "vllm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    executable = shutil.which("nvidia-smi")
    gpu = {"available": False, "devices": [], "error": None}
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                name, memory, driver = (part.strip() for part in line.rsplit(",", 2))
                gpu["devices"].append({"name": name, "memory_mib": int(memory), "driver": driver})
            gpu["available"] = bool(gpu["devices"])
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            gpu["error"] = type(exc).__name__
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "packages": packages, "gpu": gpu,
        "llamafactory_cli_available": shutil.which("llamafactory-cli") is not None,
    }


def doctor(args: argparse.Namespace) -> int:
    report = environment_report()
    _write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_gpu and not report["gpu"]["available"]:
        print("GPU preflight failed: nvidia-smi reported no usable GPU.", file=sys.stderr)
        return 1
    if args.min_vram_gb and not any(
        row["memory_mib"] >= args.min_vram_gb * 1000 for row in report["gpu"]["devices"]
    ):
        print(f"GPU preflight failed: at least {args.min_vram_gb} GB VRAM is required.", file=sys.stderr)
        return 1
    return 0


def download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the download extra: pip install -e '.[download]'") from exc
    output = Path(args.output_dir).resolve()
    manifest_path = output / "eduqa_download_manifest.json"
    # Resolve the floating model revision exactly once and record the immutable commit.
    requested_revision = args.revision
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior["repo_id"] != args.model:
            raise ValueError("Output directory belongs to a different model")
        if requested_revision is None:
            requested_revision = prior["resolved_revision"]
    info = HfApi().model_info(args.model, revision=requested_revision or "main")
    if not info.sha:
        raise RuntimeError("Could not resolve an immutable model revision")
    snapshot_download(repo_id=args.model, revision=info.sha, local_dir=output)
    _write_json(manifest_path, {
        "repo_id": args.model, "resolved_revision": info.sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_license": "See the downloaded model card and upstream license.",
    })
    print(f"Model downloaded to {output}; revision {info.sha}")
    return 0


def build_vllm_command(args: argparse.Namespace) -> list[str]:
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be between 0 and 1")
    if min(args.max_model_len, args.max_num_seqs, args.port) <= 0 or args.port > 65535:
        raise ValueError("Invalid vLLM length, concurrency or port")
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(Path(args.model_dir).resolve()),
        "--served-model-name", args.model_name,
        "--host", args.host, "--port", str(args.port),
        "--dtype", "bfloat16", "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager",
    ]


def _group_has_live_members(group_id: int) -> bool:
    """Ignore already-dead zombies when checking our Linux child process group."""
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                # comm (field 2) can contain spaces and parentheses.
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
                    return True
            except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
                continue
        return False
    try:
        os.killpg(group_id, 0)
        return True
    except ProcessLookupError:
        return False


def _signal_group(group_id: int, signum: int) -> None:
    try:
        os.killpg(group_id, signum)
    except ProcessLookupError:
        pass


def _stop_process_tree(process: subprocess.Popen, grace_seconds: float = 20.0) -> dict:
    """Stop only the session/group created for this server, including GPU workers."""
    group_id = process.pid  # Popen(start_new_session=True) makes pid == pgid.
    _signal_group(group_id, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while _group_has_live_members(group_id) and time.monotonic() < deadline:
        process.poll()  # Reap the group leader promptly if it has already exited.
        time.sleep(0.1)
    forced = _group_has_live_members(group_id)
    if forced:
        _signal_group(group_id, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while _group_has_live_members(group_id) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.1)
    process.wait(timeout=5.0)
    if _group_has_live_members(group_id):
        raise RuntimeError(f"Server process group {group_id} still has live workers after SIGKILL")
    return {"process_group": group_id, "forced_kill": forced, "complete": True}


def vllm(args: argparse.Namespace) -> int:
    command = build_vllm_command(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "command": command}, ensure_ascii=False, indent=2))
        return 0
    model = Path(args.model_dir).resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Missing merged/base model config: {model / 'config.json'}")
    if not any(model.glob("*.safetensors")) and not any(model.glob("pytorch_model*.bin")):
        raise FileNotFoundError("Model weights are missing; export or download the model first")
    try:
        metadata.version("vllm")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Activate the inference environment with vLLM installed") from exc
    if platform.system() != "Linux":
        raise RuntimeError("The vLLM GPU server requires Linux; use a cloud GPU server or --dry-run")
    env = os.environ.copy()
    if env.get("EDUQA_BACKEND_API_KEY"):
        env["VLLM_API_KEY"] = env["EDUQA_BACKEND_API_KEY"]
    report = {"command": command, "environment": environment_report(), "status": "starting",
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(Path(args.manifest), report)
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    process = None

    def terminate(signum, frame):
        report["termination_signal"] = signum
        raise KeyboardInterrupt

    for sig in previous_handlers:
        signal.signal(sig, terminate)
    try:
        process = subprocess.Popen(command, env=env, start_new_session=True, stdin=subprocess.DEVNULL)
        report.update(status="running", pid=process.pid, process_group=process.pid)
        _write_json(Path(args.manifest), report)
        returncode = process.wait()
        report.update(status="stopped" if returncode == 0 else "failed", exit_code=returncode)
        return returncode
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__)
        raise
    finally:
        # A repeated Ctrl+C/SIGTERM must not interrupt cleanup and orphan workers.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if process is not None:
                report["shutdown"] = _stop_process_tree(process)
        except Exception as exc:
            report.update(status="failed", shutdown={"complete": False, "error": str(exc)})
            raise RuntimeError("vLLM worker cleanup failed; do not start the next model on this GPU") from exc
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            report["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(Path(args.manifest), report)


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("doctor", help="Record actual device/software information")
    parser.add_argument("--output", default="artifacts/environment.json")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--min-vram-gb", type=int, default=0)
    parser.set_defaults(func=doctor)
    parser = subparsers.add_parser("download-model", help="Download and pin a model revision")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", default="models/Qwen2.5-7B-Instruct")
    parser.set_defaults(func=download)
    parser = subparsers.add_parser("start-vllm", help="Run a real vLLM backend")
    parser.add_argument("--model-dir", default="artifacts/models/eduqa-merged")
    parser.add_argument("--model-name", default="eduqa")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--manifest", default="artifacts/vllm/run.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(func=vllm)
