"""Measure real streamed HTTP requests without estimating unknown token counts."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

import httpx


async def sse_events(response: httpx.Response) -> AsyncIterator[str]:
    """Parse SSE event boundaries, comments, CRLF, and multiline data fields."""
    lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            value = line[5:]
            lines.append(value[1:] if value.startswith(" ") else value)
    if lines:
        yield "\n".join(lines)


async def stream_request(client: httpx.AsyncClient, *, base_url: str, model: str,
                         prompt: str, max_tokens: int, request_id: int) -> dict[str, Any]:
    started = perf_counter()
    result: dict[str, Any] = {"request_id": request_id, "status": "failed", "ttft_seconds": None,
                              "completion_tokens": None, "token_count_source": "unknown", "output_characters": 0,
                              "finish_reason": None, "truncated": None, "received_done": False}
    content_parts: list[str] = []
    try:
        async with client.stream("POST", base_url.rstrip("/") + "/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
        }) as response:
            response.raise_for_status()
            async for event in sse_events(response):
                if event.strip() == "[DONE]":
                    result["received_done"] = True
                    break
                payload = json.loads(event)
                if not isinstance(payload, dict) or "error" in payload:
                    raise ValueError("Provider returned an invalid/error stream event")
                usage = payload.get("usage")
                tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                    result["completion_tokens"] = tokens
                    result["token_count_source"] = "server_usage"
                choices = payload.get("choices", [])
                if not isinstance(choices, list):
                    raise ValueError("Invalid choices in SSE event")
                for choice in choices:
                    if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
                        raise ValueError("Invalid choice/delta in SSE event")
                    delta = choice.get("delta", {}).get("content")
                    if delta is not None and not isinstance(delta, str):
                        raise ValueError("Non-text content delta")
                    if delta:
                        if result["ttft_seconds"] is None:
                            result["ttft_seconds"] = perf_counter() - started
                        content_parts.append(delta)
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
            if not result["received_done"]:
                raise ValueError("Stream ended without [DONE]")
            if result["ttft_seconds"] is None:
                raise ValueError("Stream completed without nonempty text content")
            result["status"] = "success"
    except httpx.HTTPStatusError as exc:
        result["error"] = f"HTTP {exc.response.status_code}"
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        # Only controlled protocol messages are included; never save provider bodies/headers.
        result["error"] = str(exc) if type(exc) is ValueError else type(exc).__name__
    result["end_to_end_seconds"] = perf_counter() - started
    result["truncated"] = result["finish_reason"] == "length" if result["finish_reason"] is not None else None
    result["output_characters"] = sum(len(part) for part in content_parts)
    result["output_sha256"] = hashlib.sha256("".join(content_parts).encode("utf-8")).hexdigest()
    return result


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    offset = (len(ordered) - 1) * quantile
    low, high = math.floor(offset), math.ceil(offset)
    return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)


def summarize(results: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    successful = [item for item in results if item["status"] == "success"]
    ttfts = [item["ttft_seconds"] for item in successful if item["ttft_seconds"] is not None]
    durations = [item["end_to_end_seconds"] for item in successful]
    known = [item["completion_tokens"] for item in successful if item["completion_tokens"] is not None]
    truncated = sum(item.get("finish_reason") == "length" for item in successful)
    return {
        "total_requests": len(results), "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "success_rate": len(successful) / len(results) if results else None,
        "truncated_successful_requests": truncated,
        "truncated_rate_among_successful": truncated / len(successful) if successful else None,
        "successful_requests_with_unknown_finish_reason": sum(item.get("finish_reason") is None for item in successful),
        "measurement_wall_seconds": wall_seconds,
        "successful_requests_per_second": len(successful) / wall_seconds if wall_seconds > 0 else None,
        "ttft_seconds": {"p50": percentile(ttfts, 0.5), "p95": percentile(ttfts, 0.95)},
        "end_to_end_seconds": {"p50": percentile(durations, 0.5), "p95": percentile(durations, 0.95)},
        "requests_with_known_completion_tokens": len(known),
        "requests_with_unknown_completion_tokens": len(successful) - len(known),
        "known_completion_tokens": sum(known) if known else None,
        "known_completion_tokens_per_wall_second": sum(known) / wall_seconds if known and wall_seconds > 0 else None,
        "complete_token_accounting": bool(successful) and len(known) == len(successful),
        "metric_definition": "Warmup excluded. TTFT starts when a request enters an available concurrency slot and ends at the first nonempty content delta; role-only frames do not count. "
                             "Latency percentiles use successful requests only, with linear interpolation. Throughput uses the entire measured wall time including failures. "
                             "Known tokens are summed over successful requests and come only from server usage; partial known-token throughput is not total token throughput. "
                             "finish_reason=length outputs count as truncated and remain in latency/throughput statistics; stream success does not establish answer completeness.",
    }


async def benchmark(client: httpx.AsyncClient, *, base_url: str, model: str, prompt: str,
                    max_tokens: int, requests: int, concurrency: int, warmup: int) -> dict[str, Any]:
    if requests <= 0 or concurrency <= 0 or max_tokens <= 0 or warmup < 0:
        raise ValueError("requests, concurrency and max_tokens must be positive; warmup must be nonnegative")
    request_options = {"base_url": base_url, "model": model, "prompt": prompt, "max_tokens": max_tokens}
    warmup_results = [await stream_request(client, **request_options, request_id=-(index + 1))
                      for index in range(warmup)]
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int) -> dict[str, Any]:
        async with semaphore:
            return await stream_request(client, **request_options, request_id=index + 1)

    started = perf_counter()
    results = await asyncio.gather(*(worker(index) for index in range(requests)))
    wall_seconds = perf_counter() - started
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model, "configuration": {"requests": requests, "concurrency": concurrency, "warmup": warmup,
                                                "max_tokens": max_tokens, "temperature": 0, "stream": True,
                                                "prompt": prompt, "prompt_characters": len(prompt),
                                                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()},
            "summary": summarize(results, wall_seconds), "requests": results,
            "warmup": {"total": warmup, "failed": sum(item["status"] != "success" for item in warmup_results),
                       "requests": warmup_results}}


def run(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    output, overwrite = Path(args.output), getattr(args, "overwrite", False)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Benchmark report already exists: {output}; use a new path or --overwrite")
    if output.exists() and not output.is_file():
        raise FileExistsError(f"Benchmark output is not a file: {output}")
    key = os.environ.get(args.api_key_env, "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    async def execute() -> dict[str, Any]:
        async with httpx.AsyncClient(headers=headers, timeout=args.timeout,
                                     limits=httpx.Limits(max_connections=max(args.concurrency, 1),
                                                         max_keepalive_connections=max(args.concurrency, 1))) as client:
            return await benchmark(client, base_url=args.base_url, model=args.model, prompt=args.prompt,
                                   max_tokens=args.max_tokens, requests=args.requests,
                                   concurrency=args.concurrency, warmup=args.warmup)

    report = asyncio.run(execute())
    report["configuration"]["timeout_seconds"] = args.timeout
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = report["summary"]
    print(f"Benchmark: {summary['successful_requests']}/{summary['total_requests']} successful; output={output}")
    return 1 if summary["failed_requests"] or report["warmup"]["failed"] else 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("benchmark", help="Measure real concurrent streamed model requests")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="eduqa")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="请用初中生能理解的语言解释牛顿第二定律，并举一个生活中的例子。")
    parser.add_argument("--output", default="artifacts/benchmark/report.json")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing benchmark report")
    parser.add_argument("--api-key-env", default="EDUQA_BACKEND_API_KEY")
    parser.add_argument("--timeout", type=float, default=120)
    parser.set_defaults(func=run)
