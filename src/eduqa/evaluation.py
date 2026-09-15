"""Evaluate actual OpenAI-compatible answers with explicitly labelled text metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import unicodedata
from typing import Any

import httpx


def normalize_text(text: str) -> str:
    """Normalize case/width and prose punctuation while retaining mathematical syntax."""
    math_punctuation = set("+-*/=<>^%−×÷±.()[]{}|:_\\")
    return "".join(
        char for char in unicodedata.normalize("NFKC", text).lower()
        if not char.isspace() and (not unicodedata.category(char).startswith("P") or char in math_punctuation)
    )


def text_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Character overlap measures; none of these establishes factual correctness."""
    pred, ref = normalize_text(prediction), normalize_text(reference)
    if not pred or not ref:
        overlap_score = float(pred == ref)
        return {"character_f1": overlap_score, "character_rouge_l_f1": overlap_score,
                "normalized_exact_match": overlap_score}
    overlap = sum((Counter(pred) & Counter(ref)).values())
    f1 = 2 * overlap / (len(pred) + len(ref))
    # LCS needs only one row of working memory.
    long, short = (pred, ref) if len(pred) >= len(ref) else (ref, pred)
    row = [0] * (len(short) + 1)
    for char in long:
        previous = 0
        for index, other in enumerate(short, 1):
            saved = row[index]
            row[index] = previous + 1 if char == other else max(row[index], row[index - 1])
            previous = saved
    return {"character_f1": f1,
            "character_rouge_l_f1": 2 * row[-1] / (len(pred) + len(ref)),
            "normalized_exact_match": float(pred == ref)}


def load_records(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    rows = json.loads(text) if path.suffix.lower() == ".json" else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation dataset must be a nonempty JSON array or JSONL file")
    records, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {index + 1} must be an object")
        question = row.get("instruction", row.get("question"))
        extra = row.get("input", "")
        reference = row.get("output", row.get("reference", row.get("answer")))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Dataset row {index + 1} has no nonempty instruction/question")
        if not isinstance(extra, str) or not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Dataset row {index + 1} needs string input and nonempty output/reference")
        identity = str(row.get("id", index + 1))
        if identity in seen:
            raise ValueError(f"Duplicate evaluation ID: {identity}")
        seen.add(identity)
        records.append({"id": identity, "question": question + ("\n" + extra if extra else ""),
                        "reference": reference})
    return records


def dataset_hash(records: list[dict[str, str]]) -> str:
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_records(records: list[dict[str, str]], client: httpx.Client, *,
                     base_url: str, model: str, max_tokens: int) -> list[dict[str, Any]]:
    results = []
    for record in records:
        started = time.perf_counter()
        result: dict[str, Any] = {**record, "prediction": None, "status": "failed", "metrics": None,
                                  "finish_reason": None, "truncated": None}
        try:
            response = client.post(base_url.rstrip("/") + "/chat/completions", json={
                "model": model, "messages": [{"role": "user", "content": record["question"]}],
                "temperature": 0, "max_tokens": max_tokens, "stream": False,
            })
            response.raise_for_status()
            payload = response.json()
            prediction = payload["choices"][0]["message"]["content"]
            if not isinstance(prediction, str) or not prediction.strip():
                raise ValueError("Missing or empty assistant answer")
            result["latency_seconds"] = time.perf_counter() - started
            result.update(prediction=prediction, status="success", metrics=text_metrics(prediction, record["reference"]))
            result["finish_reason"] = payload["choices"][0].get("finish_reason")
            result["truncated"] = result["finish_reason"] == "length" if result["finish_reason"] is not None else None
            if isinstance(payload.get("usage"), dict):
                result["usage"] = payload["usage"]
        except httpx.HTTPStatusError as exc:
            result["error"] = f"HTTP {exc.response.status_code}"
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            # Never serialize request headers or arbitrary provider error bodies.
            result["error"] = type(exc).__name__
        if "latency_seconds" not in result:
            result["latency_seconds"] = time.perf_counter() - started
        results.append(result)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if item["status"] == "success"]
    truncated = sum(item.get("finish_reason") == "length" for item in successful)
    metric_names = ("character_f1", "character_rouge_l_f1", "normalized_exact_match")
    return {
        "total": len(results), "successful": len(successful), "failed": len(results) - len(successful),
        "success_rate": len(successful) / len(results) if results else None,
        "truncated_successful_requests": truncated,
        "truncated_rate_among_successful": truncated / len(successful) if successful else None,
        "successful_requests_with_unknown_finish_reason": sum(item.get("finish_reason") is None for item in successful),
        "metrics_on_successful_requests_only": {
            name: sum(item["metrics"][name] for item in successful) / len(successful) if successful else None
            for name in metric_names
        },
        "mean_success_latency_seconds": (
            sum(item["latency_seconds"] for item in successful) / len(successful) if successful else None
        ),
        "metric_definition": "NFKC + lowercase; strip whitespace and prose punctuation, but retain Unicode symbols, "
                             "operators (+-*/=<>^%−×÷±), decimal points, brackets and mathematical syntax (: _ \\ |). "
                             "character F1 and LCS-based character ROUGE-L F1. Exact match uses the same normalization. "
                             "These are text-overlap metrics, not factual accuracy; failures are excluded and reported separately. "
                             "finish_reason=length answers remain scored, but are counted as truncated; request success does not establish answer completeness.",
    }


def validate_comparison(summary: dict[str, Any], other: dict[str, Any]) -> None:
    if not summary.get("dataset_sha256") or summary["dataset_sha256"] != other.get("dataset_sha256"):
        raise ValueError("Cannot compare different test sets: dataset_sha256 does not match")
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict) or not {"temperature", "max_tokens"} <= parameters.keys():
        raise ValueError("Comparison requires recorded generation parameters: temperature and max_tokens")
    if parameters != other.get("parameters"):
        raise ValueError("Cannot compare different generation parameters: temperature/max_tokens must match")


def check_comparison(summary: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    validate_comparison(summary, other)
    current, previous = summary["metrics_on_successful_requests_only"], other["metrics_on_successful_requests_only"]
    return {"reference_model": other.get("model"), "reference_label": other.get("label"),
            "reference_successful": other.get("successful"), "reference_total": other.get("total"),
            "reference_truncated_rate_among_successful": other.get("truncated_rate_among_successful"),
            "warning": "Metric deltas are descriptive only; failures can change the scored subset. Use blind human review for correctness.",
            "metric_delta": {name: value - previous[name] if value is not None and previous.get(name) is not None else None
                             for name, value in current.items()}}


def validate_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Evaluation output is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Evaluation output already contains files: {output}; use a new path or --overwrite")


def write_outputs(output: Path, results: list[dict[str, Any]], summary: dict[str, Any], *, overwrite: bool = False) -> None:
    validate_output(output, overwrite)
    output.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with (output / "predictions.jsonl").open(mode, encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output / "summary.json").open(mode, encoding="utf-8") as file:
        file.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    # Anonymous label keeps the worksheet usable by a reviewer without the model name.
    with (output / "blind_review.csv").open(mode, encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["模型标签", "ID", "问题", "参考", "答案", "生成结束原因", "是否长度截断", "正确性", "完整性", "指令遵循"])
        for result in results:
            if result["status"] == "success":
                reason = result.get("finish_reason")
                cells = ["模型A", result["id"], result["question"], result["reference"], result["prediction"],
                         reason or "未知", "是" if reason == "length" else ("未知" if reason is None else "否"), "", "", ""]
                # Keep model/dataset text literal when the CSV is opened in Excel.
                writer.writerow(["'" + cell if cell.lstrip().startswith(("=", "+", "-", "@")) else cell for cell in cells])
    lines = ["# 教育问答文本评测", "", f"模型：{summary['model']}；标签：{summary['label']}",
             f"测试集 SHA-256：`{summary['dataset_sha256']}`", "",
             f"请求总数：{summary['total']}；成功：{summary['successful']}；失败：{summary['failed']}。", "",
             f"成功请求中长度截断：{summary['truncated_successful_requests']}；结束原因未知：{summary['successful_requests_with_unknown_finish_reason']}。",
             "finish_reason=length 表示答案因长度上限截断，仍参与文本打分；接口成功不等于答案完整。", "",
             "以下指标只计算成功返回的答案，衡量文本重合，不代表知识正确率。失败请求单独保存在 predictions.jsonl。", "",
             "| 文本指标 | 分数（0–1） |", "|---|---|" ]
    for name, value in summary["metrics_on_successful_requests_only"].items():
        lines.append(f"| {name} | {value:.4f} |" if value is not None else f"| {name} | 无有效结果 |")
    lines.extend(["", "blind_review.csv 中模型A对应本报告模型；请将表格交由不知道模型身份的审阅者填写。",
                  "正确性、完整性、指令遵循可各按 1–5 分评分；涉及事实、计算和推导应人工复核。", ""])
    if "comparison" in summary:
        lines.extend(["## 与参考评测的比较", "", "仅在测试集哈希和生成参数一致时生成。若失败数量不同，分数差值不可直接归因于模型改进。", "",
                      "```json", json.dumps(summary["comparison"], ensure_ascii=False, indent=2), "```", ""])
    with (output / "report.md").open(mode, encoding="utf-8") as file:
        file.write("\n".join(lines))


def run(args: argparse.Namespace) -> int:
    if args.max_tokens <= 0 or args.timeout <= 0:
        raise ValueError("max-tokens and timeout must be positive")
    records = load_records(Path(args.data))
    digest = dataset_hash(records)
    parameters = {"temperature": 0, "max_tokens": args.max_tokens}
    comparison = None
    if args.compare_to:
        comparison = json.loads(Path(args.compare_to).read_text(encoding="utf-8"))
        validate_comparison({"dataset_sha256": digest, "parameters": parameters}, comparison)
    output, overwrite = Path(args.output), getattr(args, "overwrite", False)
    validate_output(output, overwrite)
    key = os.environ.get(args.api_key_env, "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        results = evaluate_records(records, client, base_url=args.base_url, model=args.model, max_tokens=args.max_tokens)
    summary = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "model": args.model, "label": args.label or args.model,
               "dataset_sha256": digest, "dataset_path": str(Path(args.data)),
               "parameters": parameters, **summarize(results)}
    if comparison is not None:
        summary["comparison"] = check_comparison(summary, comparison)
    write_outputs(output, results, summary, overwrite=overwrite)
    print(f"Evaluation: {summary['successful']}/{summary['total']} successful; output={args.output}")
    return 0 if not summary["failed"] else 1


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate a real OpenAI-compatible model endpoint")
    parser.add_argument("--data", default="artifacts/data/test.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="eduqa")
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", default="artifacts/eval/label")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing evaluation outputs")
    parser.add_argument("--api-key-env", default="EDUQA_BACKEND_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--compare-to", help="Earlier summary.json using the same dataset and generation parameters")
    parser.set_defaults(func=run)
