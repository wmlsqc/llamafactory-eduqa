import argparse
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from eduqa.data import prepare_data
from eduqa import training


def parse_args(*argv):
    parser = argparse.ArgumentParser()
    training.add_parser(parser.add_subparsers(required=True))
    return parser.parse_args(argv)


@pytest.fixture
def data_dir(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name, questions in (("train", [f"question {i}" for i in range(10)]), ("test", ["held-out question"])):
        rows = [{"instruction": question, "input": "", "output": "a worked answer"} for question in questions]
        (source / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "data"
    prepare_data(source / "train.jsonl", source / "test.jsonl", output)
    return output


def make_adapter(path, base=training.DEFAULT_MODEL):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": base}))
    (path / "adapter_model.safetensors").write_bytes(b"test fixture, never used as real model weights")


def train_args(tmp_path, data_dir, *extra):
    return parse_args("train", "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "adapter"),
                      "--run-dir", str(tmp_path / "runs"), *extra)


@pytest.mark.parametrize("profile,batch,accum", [("4090", 1, 8), ("a100", 2, 4)])
def test_dry_run_resolves_official_profile_and_never_trains(tmp_path, data_dir, monkeypatch, profile, batch, accum):
    monkeypatch.setattr(training, "device_info", lambda: {"gpus": [], "gpu_detection": "unavailable"})
    monkeypatch.setattr(training.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run invoked subprocess"))
    args = train_args(tmp_path, data_dir, "--dry-run", "--profile", profile)
    result = args.func(args)
    assert result["status"] == "dry_run" and result["metrics"] is None
    assert result["observed_hardware"]["gpus"] == []
    assert not (tmp_path / "adapter").exists()
    config = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert config["dataset_dir"] == str(data_dir.resolve())
    assert config["eval_dataset"] == "eduqa_val"
    assert config["quantization_bit"] == 4 and config["quantization_type"] == "nf4"
    assert config["template"] == "qwen" and config["gradient_checkpointing"]
    assert config["per_device_train_batch_size"] == batch
    assert config["gradient_accumulation_steps"] == accum


def test_tampered_dataset_blocks_training_before_job_creation(tmp_path, data_dir):
    (data_dir / "train.json").write_text("[]")
    args = train_args(tmp_path, data_dir, "--dry-run")
    with pytest.raises(ValueError, match="hash mismatch"):
        args.func(args)
    assert not (tmp_path / "runs").exists()


def test_existing_output_needs_explicit_overwrite(tmp_path, data_dir, monkeypatch):
    make_adapter(tmp_path / "adapter")
    args = train_args(tmp_path, data_dir, "--dry-run")
    with pytest.raises(FileExistsError, match="overwrite"):
        args.func(args)
    monkeypatch.setattr(training, "device_info", lambda: {})
    args = train_args(tmp_path, data_dir, "--dry-run", "--overwrite")
    assert args.func(args)["status"] == "dry_run"


def test_real_train_launches_cli_and_records_actual_outputs(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(training, "device_info", lambda: {"gpus": [{"name": "test-only GPU"}]})
    monkeypatch.setattr(training.shutil, "which", lambda name: "/venv/bin/llamafactory-cli")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        make_adapter(tmp_path / "adapter")
        (tmp_path / "adapter" / "all_results.json").write_text('{"train_loss": 1.234}')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(training.subprocess, "run", run)
    args = train_args(tmp_path, data_dir, "--max-steps", "2")
    result = args.func(args)
    assert calls[0][0][:2] == ["/venv/bin/llamafactory-cli", "train"]
    assert calls[0][1] == {"check": True}
    assert result["status"] == "succeeded" and result["metrics"] == {"train_loss": 1.234}
    assert result["adapter"]["weights"][0]["sha256"]
    assert yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))["max_steps"] == 2


def test_subprocess_failure_persists_failed_manifest(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(training, "device_info", lambda: {})
    monkeypatch.setattr(training.shutil, "which", lambda name: "/venv/bin/llamafactory-cli")

    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(7, argv)

    monkeypatch.setattr(training.subprocess, "run", fail)
    args = train_args(tmp_path, data_dir)
    with pytest.raises(subprocess.CalledProcessError):
        args.func(args)
    manifest_path = next((tmp_path / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["returncode"] == 7
    assert manifest["finished_at"] and manifest["metrics"] is None


def test_missing_cli_is_recorded_without_claiming_training(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(training, "device_info", lambda: {})
    monkeypatch.setattr(training.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="llamafactory-cli not found"):
        train_args(tmp_path, data_dir).func(train_args(tmp_path, data_dir))
    manifest = json.loads(next((tmp_path / "runs").glob("*/manifest.json")).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_export_missing_adapter_is_only_allowed_for_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "device_info", lambda: {})
    base = ["export", "--adapter-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "merged"),
            "--run-dir", str(tmp_path / "runs")]
    dry = parse_args(*base, "--dry-run")
    result = dry.func(dry)
    assert result["status"] == "dry_run" and result["inputs"]["adapter"]["status"] == "planned"
    config = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert not any("quantization" in key for key in config)
    real = parse_args(*base)
    with pytest.raises(FileNotFoundError):
        real.func(real)


@pytest.mark.parametrize("key", ["quantization_bit", "export_quantization_bit", "quantization_method"])
def test_export_rejects_quantization_configuration(tmp_path, key):
    config = {"model_name_or_path": training.DEFAULT_MODEL, key: 4}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    args = parse_args("export", "--config", str(path), "--adapter-dir", str(tmp_path / "adapter"),
                      "--output-dir", str(tmp_path / "merged"), "--dry-run")
    with pytest.raises(ValueError, match="unquantized base"):
        args.func(args)


def test_export_rejects_mismatched_base(tmp_path):
    make_adapter(tmp_path / "adapter", base="Qwen/Qwen2.5-3B-Instruct")
    args = parse_args("export", "--adapter-dir", str(tmp_path / "adapter"),
                      "--output-dir", str(tmp_path / "merged"), "--dry-run")
    with pytest.raises(ValueError, match="differs from selected model"):
        args.func(args)


def test_local_quantized_base_is_not_mergeable(tmp_path):
    base = tmp_path / "quantized-base"
    base.mkdir()
    (base / "config.json").write_text('{"quantization_config":{"bits":4}}')
    args = parse_args("export", "--model", str(base), "--adapter-dir", str(tmp_path / "adapter"),
                      "--output-dir", str(tmp_path / "merged"), "--dry-run")
    with pytest.raises(ValueError, match="locally quantized"):
        args.func(args)


def test_real_export_runs_unquantized_merge_and_validates_result(tmp_path, monkeypatch):
    make_adapter(tmp_path / "adapter")
    monkeypatch.setattr(training, "device_info", lambda: {})
    monkeypatch.setattr(training.shutil, "which", lambda name: "/venv/bin/llamafactory-cli")

    def run(argv, **kwargs):
        assert argv[1] == "export" and kwargs["check"] is True
        config = yaml.safe_load(Path(argv[2]).read_text(encoding="utf-8"))
        assert "quantization_bit" not in config
        destination = Path(config["export_dir"])
        destination.mkdir()
        (destination / "config.json").write_text("{}")
        (destination / "model.safetensors").write_bytes(b"test fixture")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(training.subprocess, "run", run)
    args = parse_args("export", "--adapter-dir", str(tmp_path / "adapter"),
                      "--output-dir", str(tmp_path / "merged"), "--run-dir", str(tmp_path / "runs"))
    result = args.func(args)
    assert result["status"] == "succeeded" and result["exported_weights"][0]["bytes"] > 0


def test_device_detection_uses_nvidia_smi_output(monkeypatch):
    monkeypatch.setattr(training.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(training.subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, '0, NVIDIA RTX 4090, 24564, 550.54.14\n'))
    observed = training.device_info()
    assert observed["gpu_detection"] == "observed"
    assert observed["gpus"] == [{"index": 0, "name": "NVIDIA RTX 4090", "memory_mib": 24564, "driver_version": "550.54.14"}]


def test_no_nvidia_smi_means_unknown_hardware(monkeypatch):
    monkeypatch.setattr(training.shutil, "which", lambda name: None)
    observed = training.device_info()
    assert observed["gpu_detection"] == "unavailable" and observed["gpus"] == []


def test_training_output_cannot_overlap_input(tmp_path, data_dir):
    args = train_args(tmp_path, data_dir, "--output-dir", str(data_dir / "model"), "--dry-run")
    with pytest.raises(ValueError, match="overlap"):
        args.func(args)
