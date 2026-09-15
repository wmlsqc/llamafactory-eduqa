import argparse
import csv
import json

import httpx
import pytest

from eduqa import evaluation


def test_text_metrics_are_character_overlap_not_semantic_accuracy():
    assert evaluation.text_metrics("Ａ B，答！", "ab答") == {
        "character_f1": 1.0, "character_rouge_l_f1": 1.0, "normalized_exact_match": 1.0,
    }
    # Same character multiset, different order: F1 cannot establish factual correctness.
    scores = evaluation.text_metrics("甲乙", "乙甲")
    assert scores == {"character_f1": 1.0, "character_rouge_l_f1": 0.5, "normalized_exact_match": 0.0}
    assert evaluation.text_metrics("", "有答案")["character_f1"] == 0
    assert evaluation.text_metrics("", "")["character_rouge_l_f1"] == 1


@pytest.mark.parametrize("left,right", [("1+1", "1-1"), ("2×3", "2÷3"), ("x<1", "x>1"),
                                       ("3.14", "314"), ("(a+b)*c", "a+b*c"), ("10%", "10"),
                                       ("x=−1", "x=1"), ("2^3", "23")])
def test_math_operators_decimal_points_and_brackets_remain_distinct(left, right):
    assert evaluation.text_metrics(left, right)["normalized_exact_match"] == 0
    assert evaluation.normalize_text("+-*/=<>^%−×÷±") == "+-*/=<>^%−×÷±"


def test_load_alpaca_and_jsonl_preserves_question_context_and_hash(tmp_path):
    rows = [{"id": "a", "instruction": "解释", "input": "重力", "output": "地球吸引物体的力"}]
    json_path, jsonl_path = tmp_path / "test.json", tmp_path / "test.jsonl"
    json_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    jsonl_path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    records = evaluation.load_records(json_path)
    assert records[0]["question"] == "解释\n重力"
    assert evaluation.dataset_hash(records) == evaluation.dataset_hash(evaluation.load_records(jsonl_path))
    changed = [{**records[0], "reference": "不同答案"}]
    assert evaluation.dataset_hash(records) != evaluation.dataset_hash(changed)


@pytest.mark.parametrize("rows", [[], [None], [{"instruction": "问题"}],
                                      [{"instruction": "问题", "input": [], "output": "答案"}],
                                      [{"id": "a", "instruction": "问题", "output": "答案"}] * 2])
def test_invalid_dataset_rejected(tmp_path, rows):
    path = tmp_path / "test.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.load_records(path)


def test_real_openai_request_shape_and_failure_accounting():
    bodies = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
                                             "usage": {"completion_tokens": 2}})
        return httpx.Response(503, json={"error": "private provider details"})

    records = [{"id": str(i), "question": "问题", "reference": "答案"} for i in range(2)]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        results = evaluation.evaluate_records(records, client, base_url="http://test/v1/", model="eduqa", max_tokens=20)
    assert bodies[0] == {"model": "eduqa", "messages": [{"role": "user", "content": "问题"}],
                         "temperature": 0, "max_tokens": 20, "stream": False}
    assert results[0]["status"] == "success"
    assert results[1]["prediction"] is None
    assert results[1]["metrics"] is None
    assert results[1]["error"] == "HTTP 503"
    assert "private provider details" not in json.dumps(results)
    summary = evaluation.summarize(results)
    assert summary["success_rate"] == 0.5
    assert summary["failed"] == 1
    assert summary["metrics_on_successful_requests_only"]["character_f1"] == 1


@pytest.mark.parametrize("payload", [{"choices": []}, {"choices": [{"message": {"content": ""}}]},
                                       {"error": {"message": "secret"}}, []])
def test_malformed_or_empty_answers_are_failures(payload):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        results = evaluation.evaluate_records([{"id": "1", "question": "q", "reference": "r"}], client,
                                             base_url="http://test/v1", model="test", max_tokens=8)
    assert results[0]["status"] == "failed"
    assert evaluation.summarize(results)["metrics_on_successful_requests_only"]["character_f1"] is None


def test_zero_results_have_null_scores():
    summary = evaluation.summarize([])
    assert summary["success_rate"] is None
    assert summary["mean_success_latency_seconds"] is None
    assert all(value is None for value in summary["metrics_on_successful_requests_only"].values())


def test_comparison_requires_identical_dataset_hash():
    parameters = {"temperature": 0, "max_tokens": 512}
    summary = {"dataset_sha256": "one", "model": "tuned", "parameters": parameters, "metrics_on_successful_requests_only": {"character_f1": 0.8}}
    previous = {"dataset_sha256": "one", "model": "base", "parameters": parameters, "metrics_on_successful_requests_only": {"character_f1": 0.5}}
    assert evaluation.check_comparison(summary, previous)["metric_delta"]["character_f1"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="different test sets"):
        evaluation.check_comparison(summary, {**previous, "dataset_sha256": "two"})
    with pytest.raises(ValueError):
        evaluation.check_comparison({}, {})


@pytest.mark.parametrize("parameters", [None, {}, {"temperature": 1, "max_tokens": 512},
                                       {"temperature": 0, "max_tokens": 128}])
def test_comparison_requires_same_recorded_generation_parameters(parameters):
    summary = {"dataset_sha256": "one", "parameters": {"temperature": 0, "max_tokens": 512}}
    with pytest.raises(ValueError, match="generation parameters"):
        evaluation.check_comparison(summary, {"dataset_sha256": "one", "parameters": parameters})


def test_outputs_and_nonzero_cli_status_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "test.json"
    path.write_text(json.dumps([{"instruction": "q", "output": "r"}]), encoding="utf-8")
    observed = {}
    real_client = httpx.Client

    def client_factory(**kwargs):
        observed.update(kwargs)
        return real_client(**kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(500)))

    monkeypatch.setenv("TEST_EVAL_KEY", "secret-value-not-to-save")
    monkeypatch.setattr(evaluation.httpx, "Client", client_factory)
    args = argparse.Namespace(data=str(path), output=str(tmp_path / "out"), compare_to=None,
                              api_key_env="TEST_EVAL_KEY", timeout=10, max_tokens=10,
                              base_url="http://test/v1", model="test", label="tuned")
    assert evaluation.run(args) == 1
    assert observed["headers"] == {"Authorization": "Bearer secret-value-not-to-save"}
    outputs = list((tmp_path / "out").iterdir())
    assert {file.name for file in outputs} == {"predictions.jsonl", "summary.json", "report.md", "blind_review.csv"}
    assert all("secret-value-not-to-save" not in file.read_text(encoding="utf-8-sig") for file in outputs)
    with (tmp_path / "out" / "blind_review.csv").open(encoding="utf-8-sig", newline="") as file:
        assert next(csv.reader(file))[-3:] == ["正确性", "完整性", "指令遵循"]


def test_cli_comparison_mismatch_fails_before_any_api_request(tmp_path, monkeypatch):
    data, previous = tmp_path / "test.json", tmp_path / "previous.json"
    data.write_text(json.dumps([{"instruction": "q", "output": "r"}]), encoding="utf-8")
    previous.write_text(json.dumps({"dataset_sha256": "wrong"}), encoding="utf-8")

    def forbidden(**kwargs):
        pytest.fail("Mismatched dataset must be rejected before opening an HTTP client")

    monkeypatch.setattr(evaluation.httpx, "Client", forbidden)
    with pytest.raises(ValueError, match="different test sets"):
        evaluation.run(argparse.Namespace(data=str(data), compare_to=str(previous), max_tokens=10, timeout=10))


def test_cli_parameter_mismatch_fails_before_any_api_request(tmp_path, monkeypatch):
    data, previous = tmp_path / "test.json", tmp_path / "previous.json"
    data.write_text(json.dumps([{"instruction": "q", "output": "r"}]), encoding="utf-8")
    digest = evaluation.dataset_hash(evaluation.load_records(data))
    previous.write_text(json.dumps({"dataset_sha256": digest, "parameters": {"temperature": 0, "max_tokens": 99}}), encoding="utf-8")

    def forbidden(**kwargs):
        pytest.fail("Parameter mismatch must fail before an HTTP request")

    monkeypatch.setattr(evaluation.httpx, "Client", forbidden)
    with pytest.raises(ValueError, match="generation parameters"):
        evaluation.run(argparse.Namespace(data=str(data), compare_to=str(previous), max_tokens=10, timeout=10))


def test_truncated_answers_are_scored_but_marked_in_all_outputs(tmp_path):
    payload = {"choices": [{"message": {"content": "答案"}, "finish_reason": "length"}]}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        results = evaluation.evaluate_records([{"id": "1", "question": "q", "reference": "答案"}], client,
                                             base_url="http://test/v1", model="test", max_tokens=8)
    assert results[0]["status"] == "success"
    assert results[0]["truncated"] is True
    summary = {**evaluation.summarize(results), "dataset_sha256": "one", "model": "test", "label": "test"}
    assert summary["truncated_successful_requests"] == 1
    assert summary["truncated_rate_among_successful"] == 1
    assert summary["metrics_on_successful_requests_only"]["character_f1"] == 1
    evaluation.write_outputs(tmp_path / "out", results, summary)
    with (tmp_path / "out" / "blind_review.csv").open(encoding="utf-8-sig", newline="") as file:
        row = list(csv.DictReader(file))[0]
    assert row["生成结束原因"] == "length"
    assert row["是否长度截断"] == "是"
    assert "长度截断：1" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluation.write_outputs(tmp_path / "out", results, summary)
    evaluation.write_outputs(tmp_path / "out", results, summary, overwrite=True)


def test_cli_refuses_existing_output_before_any_api_request(tmp_path, monkeypatch):
    data, output = tmp_path / "test.json", tmp_path / "out"
    data.write_text(json.dumps([{"instruction": "q", "output": "r"}]), encoding="utf-8")
    output.mkdir()
    (output / "summary.json").write_text("existing evidence", encoding="utf-8")

    def forbidden(**kwargs):
        pytest.fail("Existing output must fail before an HTTP request")

    monkeypatch.setattr(evaluation.httpx, "Client", forbidden)
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluation.run(argparse.Namespace(data=str(data), output=str(output), compare_to=None, max_tokens=10, timeout=10))
    assert (output / "summary.json").read_text(encoding="utf-8") == "existing evidence"
