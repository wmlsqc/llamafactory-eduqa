"""Execute genuine LLaMA-Factory jobs with immutable resolved configs and run records."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import yaml

from .data import sha256_file


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_config(name: str) -> Path:
    # Works for both editable installs and a wheel installed while the repository
    # (including its reviewable configs/) is mounted as the working directory.
    for directory in (Path.cwd() / "configs", PROJECT_ROOT / "configs"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate configs/{name}; run from the repository root or pass --config")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
        raise ValueError(f"{path}: expected a YAML mapping with string keys")
    return config


def device_info() -> dict:
    """Record observed hardware only. A recommended profile is never a measurement."""
    result = {
        "os": platform.platform(), "python": platform.python_version(),
        "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [], "gpu_detection": "unavailable",
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        result["gpu_detection_reason"] = "nvidia-smi not found"
        return result
    try:
        output = subprocess.run(
            [nvidia_smi, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        for row in csv.reader(io.StringIO(output.stdout)):
            if len(row) != 4:
                raise ValueError(f"Unexpected nvidia-smi CSV row: {row!r}")
            index, name, memory, driver = (part.strip() for part in row)
            result["gpus"].append({"index": int(index), "name": name,
                                   "memory_mib": int(memory), "driver_version": driver})
        result["gpu_detection"] = "observed"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["gpus"] = []
        result["gpu_detection_reason"] = str(exc)
    return result


def _versions() -> dict[str, str | None]:
    versions = {}
    for package in ("llamafactory", "torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _model_reference(model: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path.resolve())
    if path.is_absolute() or model.startswith(("./", "../", ".\\", "..\\", "~")):
        raise FileNotFoundError(f"Local model path does not exist: {model}")
    return model


def _validate_data(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    for name in ("train.json", "val.json", "test.json", "dataset_info.json"):
        expected_hash = files.get(name, {}).get("sha256")
        if not expected_hash or sha256_file(data_dir / name) != expected_hash:
            raise ValueError(f"Dataset hash mismatch or absent manifest entry: {name}; rerun prepare-data")
    if manifest.get("train_test_prompt_overlap") != 0:
        raise ValueError("Data manifest does not certify zero train/test prompt overlap")
    for name in ("train", "val", "test"):
        if not isinstance(manifest.get("counts", {}).get(name), int) or manifest["counts"][name] < 1:
            raise ValueError(f"Data manifest must have a nonempty {name} split")
    return {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "counts": manifest["counts"]}


def _protect_output(output_dir: Path, overwrite: bool, inputs: tuple[Path, ...]) -> None:
    if output_dir == Path(output_dir.anchor) or output_dir == PROJECT_ROOT:
        raise ValueError("Model output must be a dedicated subdirectory")
    for source in inputs:
        if source == output_dir or output_dir in source.parents or source in output_dir.parents:
            raise ValueError(f"Model output must not overlap an input directory: {source}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Model output is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; choose a new output or pass --overwrite")


def _adapter_metadata(adapter_dir: Path, model: str, dry_run: bool) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists() and dry_run:
        return {"status": "planned", "path": str(adapter_dir), "reason": "Adapter absent; dry-run validates configuration only"}
    adapter = json.loads(config_path.read_text(encoding="utf-8"))
    if str(adapter.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"{config_path}: expected a LoRA adapter")
    adapter_base = adapter.get("base_model_name_or_path")
    if adapter_base and _model_reference(adapter_base) != model:
        raise ValueError(f"Adapter base {adapter_base!r} differs from selected model {model!r}; pass the exact original --model")
    weights = [adapter_dir / name for name in ("adapter_model.safetensors", "adapter_model.bin")]
    existing = [path for path in weights if path.is_file() and path.stat().st_size > 0]
    if not existing:
        raise FileNotFoundError(f"No nonempty adapter_model.safetensors or adapter_model.bin in {adapter_dir}")
    return {"status": "present", "path": str(adapter_dir), "config_sha256": sha256_file(config_path),
            "weights": [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in existing]}


def _validate_export_config(config: dict) -> None:
    forbidden = [key for key in config if "quantization" in key]
    if forbidden:
        raise ValueError(f"Adapter merge requires an unquantized base; remove keys: {', '.join(forbidden)}")
    base_config = Path(config["model_name_or_path"]) / "config.json"
    if base_config.is_file():
        metadata = json.loads(base_config.read_text(encoding="utf-8"))
        if metadata.get("quantization_config"):
            raise ValueError("Adapter merge cannot use a locally quantized base model; select the original model")


def _save_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_job(command: str, config: dict, args: argparse.Namespace, evidence: dict) -> dict:
    run_root = Path(args.run_dir).resolve()
    output_key = "output_dir" if command == "train" else "export_dir"
    output_dir = Path(config[output_key])
    if run_root == output_dir or output_dir in run_root.parents:
        raise ValueError("Run metadata must be stored outside the model output directory")
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{command}-", dir=run_root))
    config_path = run_dir / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cli = shutil.which("llamafactory-cli")
    invocation = [cli or "llamafactory-cli", command, str(config_path)]
    manifest = {
        "schema_version": 1, "command": command, "argv": invocation,
        "status": "dry_run" if args.dry_run else "running", "dry_run": args.dry_run,
        "started_at": _now(), "finished_at": None,
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "output_dir": str(output_dir), "model_name_or_path": config["model_name_or_path"],
        "observed_hardware": device_info(), "package_versions": _versions(),
        "inputs": evidence, "metrics": None,
    }
    manifest_path = run_dir / "manifest.json"
    _save_manifest(manifest_path, manifest)
    try:
        if not args.dry_run:
            if not cli:
                raise FileNotFoundError("llamafactory-cli not found; install LLaMA-Factory v0.9.3 in this environment")
            subprocess.run(invocation, check=True)
            if command == "train":
                manifest["adapter"] = _adapter_metadata(output_dir, config["model_name_or_path"], False)
                metrics_path = output_dir / "all_results.json"
                if metrics_path.is_file():
                    manifest["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                if not (output_dir / "config.json").is_file():
                    raise FileNotFoundError("Export command returned without writing config.json")
                weights = sorted(output_dir.glob("*.safetensors")) + sorted(output_dir.glob("pytorch_model*.bin"))
                if not weights or any(path.stat().st_size == 0 for path in weights):
                    raise FileNotFoundError("Export command returned without nonempty model weight files")
                manifest["exported_weights"] = [{"file": path.name, "bytes": path.stat().st_size} for path in weights]
            manifest["status"] = "succeeded"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.CalledProcessError):
            manifest["returncode"] = exc.returncode
        raise
    finally:
        manifest["finished_at"] = _now()
        _save_manifest(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "run_manifest": str(manifest_path),
                          "resolved_config": str(config_path), "argv": invocation}, ensure_ascii=False, indent=2))
    return manifest


def train(args: argparse.Namespace) -> dict:
    config_path = Path(args.config) if args.config else _default_config(f"train_{args.profile}.yaml")
    config = load_config(config_path)
    data_dir, output_dir = Path(args.data_dir).resolve(), Path(args.output_dir).resolve()
    model = _model_reference(args.model or config.get("model_name_or_path", DEFAULT_MODEL))
    inputs = (data_dir, Path(model)) if Path(model).is_dir() else (data_dir,)
    _protect_output(output_dir, args.overwrite, inputs)
    dataset = _validate_data(data_dir)
    config.update(model_name_or_path=model, dataset_dir=str(data_dir), output_dir=str(output_dir),
                  overwrite_output_dir=args.overwrite, dataset="eduqa_train", eval_dataset="eduqa_val")
    if config.get("stage") != "sft" or config.get("finetuning_type") != "lora" or config.get("do_train") is not True:
        raise ValueError("Training requires stage: sft, finetuning_type: lora, do_train: true")
    if config.get("val_size"):
        raise ValueError("val_size must be absent because prepare-data supplies an independent validation split")
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        config["max_steps"] = args.max_steps
    return _run_job("train", config, args, {"dataset": dataset, "requested_profile": args.profile,
                                           "source_config": str(config_path.resolve())})


def export(args: argparse.Namespace) -> dict:
    config_path = Path(args.config) if args.config else _default_config("export.yaml")
    config = load_config(config_path)
    adapter_dir, output_dir = Path(args.adapter_dir).resolve(), Path(args.output_dir).resolve()
    model = _model_reference(args.model or config.get("model_name_or_path", DEFAULT_MODEL))
    inputs = (adapter_dir, Path(model)) if Path(model).is_dir() else (adapter_dir,)
    _protect_output(output_dir, args.overwrite, inputs)
    config.update(model_name_or_path=model, adapter_name_or_path=str(adapter_dir), export_dir=str(output_dir))
    if args.export_device:
        config["export_device"] = args.export_device
    _validate_export_config(config)
    adapter = _adapter_metadata(adapter_dir, model, args.dry_run)
    return _run_job("export", config, args, {"adapter": adapter, "source_config": str(config_path.resolve())})


def add_parser(subparsers) -> None:
    train_parser = subparsers.add_parser("train", help="Run LLaMA-Factory QLoRA SFT with an auditable run manifest")
    train_parser.add_argument("--profile", choices=("4090", "a100"), default="4090")
    train_parser.add_argument("--data-dir", default="artifacts/data")
    train_parser.add_argument("--output-dir", default="artifacts/models/eduqa-adapter")
    train_parser.add_argument("--max-steps", type=int, help="Override training steps for a real GPU smoke test")
    train_parser.set_defaults(func=train)
    export_parser = subparsers.add_parser("export", help="Reload the unquantized base and merge a trained LoRA adapter")
    export_parser.add_argument("--adapter-dir", default="artifacts/models/eduqa-adapter")
    export_parser.add_argument("--output-dir", default="artifacts/models/eduqa-merged")
    export_parser.add_argument("--export-device", choices=("cpu", "auto"))
    export_parser.set_defaults(func=export)
    for parser in (train_parser, export_parser):
        parser.add_argument("--config", help="Alternative LLaMA-Factory YAML configuration")
        parser.add_argument("--model", help="Original Hugging Face model ID or local model directory")
        parser.add_argument("--run-dir", default="artifacts/runs")
        parser.add_argument("--dry-run", action="store_true", help="Save resolved config and manifest without loading model weights")
        parser.add_argument("--overwrite", action="store_true", help="Explicitly allow writing an existing model output")
