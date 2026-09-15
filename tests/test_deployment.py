import argparse
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import pytest
from eduqa import deployment


def parse(*args):
    parser = argparse.ArgumentParser()
    deployment.add_parser(parser.add_subparsers(required=True))
    return parser.parse_args(args)


def test_vllm_configuration_is_local_and_secrets_are_not_arguments(monkeypatch):
    monkeypatch.setenv("EDUQA_BACKEND_API_KEY", "private-value")
    args = parse("start-vllm", "--dry-run")
    command = deployment.build_vllm_command(args)
    assert "private-value" not in " ".join(command)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--max-model-len") + 1] == "2048"
    assert "--enforce-eager" in command


@pytest.mark.parametrize("value", ["0", "1", "1.2", "-0.1"])
def test_bad_memory_fraction_is_rejected(value):
    with pytest.raises(ValueError):
        deployment.build_vllm_command(parse("start-vllm", "--gpu-memory-utilization", value))


def test_dry_run_never_launches_gpu_process(monkeypatch, capsys):
    monkeypatch.setattr(deployment.subprocess, "run", lambda *a, **k: pytest.fail("process launched"))
    monkeypatch.setattr(deployment.subprocess, "Popen", lambda *a, **k: pytest.fail("process launched"))
    assert deployment.vllm(parse("start-vllm", "--dry-run")) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"]


def test_missing_gpu_fails_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(deployment, "environment_report", lambda: {"gpu": {"available": False, "devices": []}})
    args = parse("doctor", "--require-gpu", "--output", str(tmp_path / "device.json"))
    assert deployment.doctor(args) == 1
    assert (tmp_path / "device.json").is_file()


def test_download_resumes_at_recorded_immutable_revision(monkeypatch, tmp_path):
    import sys
    seen = {}
    class API:
        def model_info(self, repo_id, revision):
            seen["revision"] = revision
            return SimpleNamespace(sha="immutable-sha")
    def snapshot_download(**kwargs):
        seen["snapshot_revision"] = kwargs["revision"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, snapshot_download=snapshot_download))
    output = tmp_path / "model"
    output.mkdir()
    (output / "eduqa_download_manifest.json").write_text(json.dumps({"repo_id": "Qwen/Qwen2.5-7B-Instruct", "resolved_revision": "old-pinned-sha"}))
    assert deployment.download(parse("download-model", "--output-dir", str(output))) == 0
    assert seen == {"revision": "old-pinned-sha", "snapshot_revision": "immutable-sha"}


@pytest.mark.parametrize("error,status", [(OSError("failed"), "failed"), (KeyboardInterrupt(), "interrupted")])
def test_failed_or_interrupted_launch_is_recorded(monkeypatch, tmp_path, error, status):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test fixture, not real weights")
    manifest = tmp_path / "run.json"
    monkeypatch.setattr(deployment.metadata, "version", lambda name: "0.8.5")
    monkeypatch.setattr(deployment, "environment_report", lambda: {})
    monkeypatch.setattr(deployment.platform, "system", lambda: "Linux")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(deployment.subprocess, "Popen", fail)
    with pytest.raises(type(error)):
        deployment.vllm(parse("start-vllm", "--model-dir", str(model), "--manifest", str(manifest)))
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == status


def model_args(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test-only weights")
    return parse("start-vllm", "--model-dir", str(model), "--manifest", str(tmp_path / "run.json"))


def mock_server_environment(monkeypatch):
    monkeypatch.setattr(deployment.metadata, "version", lambda name: "0.8.5")
    monkeypatch.setattr(deployment, "environment_report", lambda: {})
    monkeypatch.setattr(deployment.platform, "system", lambda: "Linux")


@pytest.mark.parametrize("interrupt", [None, "keyboard", "sigterm"])
def test_server_isolated_session_always_cleans_workers_before_return(monkeypatch, tmp_path, interrupt):
    args = model_args(tmp_path)
    mock_server_environment(monkeypatch)
    monkeypatch.setenv("EDUQA_BACKEND_API_KEY", "server-test-secret")
    events = []
    original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    class Process:
        pid = 43210

        def wait(self):
            events.append("wait")
            if interrupt == "keyboard":
                raise KeyboardInterrupt
            if interrupt == "sigterm":
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return 0

    def launch(command, **kwargs):
        events.append("launch")
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["env"]["VLLM_API_KEY"] == "server-test-secret"
        assert "server-test-secret" not in " ".join(command)
        return Process()

    def cleanup(process):
        events.append("cleanup")
        assert process.pid == 43210
        assert all(signal.getsignal(sig) == signal.SIG_IGN for sig in original_handlers)
        return {"process_group": process.pid, "complete": True, "forced_kill": False}

    monkeypatch.setattr(deployment.subprocess, "Popen", launch)
    monkeypatch.setattr(deployment, "_stop_process_tree", cleanup)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            deployment.vllm(args)
    else:
        assert deployment.vllm(args) == 0
    assert events == ["launch", "wait", "cleanup"]
    assert all(signal.getsignal(sig) == handler for sig, handler in original_handlers.items())
    report = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    assert report["status"] == ("interrupted" if interrupt else "stopped")
    assert report["shutdown"]["complete"] and report["process_group"] == 43210
    assert report["ended_at_utc"]
    assert "server-test-secret" not in Path(args.manifest).read_text(encoding="utf-8")


def test_cleanup_failure_is_recorded_and_blocks_next_stage(monkeypatch, tmp_path):
    args = model_args(tmp_path)
    mock_server_environment(monkeypatch)
    monkeypatch.setattr(deployment.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=12345, wait=lambda: 0))

    def cleanup(process):
        raise RuntimeError("worker remains alive")

    monkeypatch.setattr(deployment, "_stop_process_tree", cleanup)
    with pytest.raises(RuntimeError, match="do not start the next model"):
        deployment.vllm(args)
    report = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["shutdown"]["complete"] is False


@pytest.mark.parametrize("forced", [False, True])
def test_process_tree_cleanup_targets_own_group_then_waits(monkeypatch, forced):
    events = []
    monkeypatch.setattr(deployment.signal, "SIGKILL", 9, raising=False)
    # First loop check, forced-kill decision, second-loop check, final assertion.
    alive = iter([False, False, False] if not forced else [True, True, False, False])
    monkeypatch.setattr(deployment, "_group_has_live_members", lambda group: next(alive))
    monkeypatch.setattr(deployment, "_signal_group", lambda group, sig: events.append((group, sig)))
    process = SimpleNamespace(pid=41721, poll=lambda: None, wait=lambda **kwargs: events.append(("wait", kwargs)))
    result = deployment._stop_process_tree(process, grace_seconds=0)
    assert events[0] == (41721, signal.SIGTERM)
    if forced:
        assert events[1] == (41721, 9)
    assert events[-1] == ("wait", {"timeout": 5.0})
    assert result == {"process_group": 41721, "forced_kill": forced, "complete": True}


def test_parent_exit_does_not_skip_waiting_for_live_worker(monkeypatch):
    events = []
    # Leader already exited; its child remains in the group for one polling cycle.
    alive = iter([True, False, False, False])
    monkeypatch.setattr(deployment, "_group_has_live_members", lambda group: next(alive))
    monkeypatch.setattr(deployment, "_signal_group", lambda group, sig: events.append("TERM"))
    monkeypatch.setattr(deployment.time, "sleep", lambda seconds: events.append("waiting for worker"))
    process = SimpleNamespace(pid=41721, poll=lambda: events.append("reap leader"),
                              wait=lambda **kwargs: events.append("wait complete"))
    assert deployment._stop_process_tree(process)["complete"]
    assert events == ["TERM", "reap leader", "waiting for worker", "wait complete"]


def test_linux_group_scan_ignores_zombies_and_unrelated_processes(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    for pid, state, group in ((101, "Z", 123), (102, "S", 999), (103, "S", 123)):
        directory = proc / str(pid)
        directory.mkdir()
        (directory / "stat").write_text(f"{pid} (worker (with spaces)) {state} 1 {group} 123 0 0")
    real_path = Path
    monkeypatch.setattr(deployment, "Path", lambda value: proc if value == "/proc" else real_path(value))
    assert deployment._group_has_live_members(123)
    (proc / "103" / "stat").write_text("103 (worker) Z 1 123 123 0 0")
    assert not deployment._group_has_live_members(123)


def test_pipeline_reuses_only_verified_matching_prepared_data(monkeypatch, tmp_path, capsys):
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_cloud_pipeline.sh").read_text(encoding="utf-8")
    snippet = script.split('"$train_python" - <<\'PY\'\n', 1)[1].split("\nPY", 1)[0]
    source = tmp_path / "EDU-QA" / "data"
    source.mkdir(parents=True)
    for name, questions in (("science_ft_500.jsonl", ["one", "two"]), ("test.jsonl", ["three"])):
        (source / name).write_text("\n".join(json.dumps({"instruction": q, "output": "answer"}) for q in questions), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    exec(compile(snippet, "pipeline-data-stage", "exec"), {})
    manifest = tmp_path / "artifacts" / "data" / "manifest.json"
    initial = manifest.read_bytes()
    exec(compile(snippet, "pipeline-data-stage", "exec"), {})
    assert "Reusing verified prepared datasets" in capsys.readouterr().out
    assert manifest.read_bytes() == initial
    (source / "test.jsonl").write_text('{"instruction":"changed","output":"answer"}')
    with pytest.raises(SystemExit, match="differs"):
        exec(compile(snippet, "pipeline-data-stage", "exec"), {})
    assert manifest.read_bytes() == initial
