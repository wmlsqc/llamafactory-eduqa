import argparse
import asyncio
import json

import httpx
import pytest

from eduqa import benchmark as bm


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def make_sse(include_usage=True, done=True):
    frames = [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "牛顿"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "第二定律"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    if include_usage:
        frames.append({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4}})
    payload = ": keepalive\r\n\r\n" + "".join("data: " + json.dumps(frame, ensure_ascii=False) + "\r\n\r\n" for frame in frames)
    return (payload + ("data: [DONE]\r\n\r\n" if done else "")).encode("utf-8")


def execute_single(transport):
    async def execute():
        async with httpx.AsyncClient(transport=transport) as client:
            return await bm.stream_request(client, base_url="http://test/v1", model="eduqa", prompt="问题", max_tokens=20, request_id=1)
    return asyncio.run(execute())


def test_stream_chunk_boundaries_usage_and_ttft_ignore_role_frames(monkeypatch):
    payload = make_sse()
    # One-byte transport chunks also cut Chinese UTF-8 codepoints and CRLF boundaries.
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=ChunkStream([payload[i:i + 1] for i in range(len(payload))])))
    clock = iter([10.0, 12.0, 14.0])
    monkeypatch.setattr(bm, "perf_counter", lambda: next(clock))
    result = execute_single(transport)
    assert result["status"] == "success"
    assert result["ttft_seconds"] == 2
    assert result["end_to_end_seconds"] == 4
    assert result["completion_tokens"] == 4
    assert result["token_count_source"] == "server_usage"
    assert result["output_characters"] == 6
    assert result["received_done"] is True
    assert result["finish_reason"] == "stop"


def test_missing_usage_never_estimated_from_characters():
    result = execute_single(httpx.MockTransport(lambda _: httpx.Response(200, stream=ChunkStream([make_sse(include_usage=False)]))))
    assert result["status"] == "success"
    assert result["output_characters"] == 6
    assert result["completion_tokens"] is None
    assert result["token_count_source"] == "unknown"
    summary = bm.summarize([result], 1)
    assert summary["known_completion_tokens"] is None
    assert summary["known_completion_tokens_per_wall_second"] is None
    assert summary["complete_token_accounting"] is False


def test_truncated_stream_is_reported_even_with_done():
    payload = make_sse().replace(b'"stop"', b'"length"')
    result = execute_single(httpx.MockTransport(lambda _: httpx.Response(200, stream=ChunkStream([payload]))))
    assert result["status"] == "success"
    assert result["received_done"] is True
    assert result["truncated"] is True
    summary = bm.summarize([result], 1)
    assert summary["truncated_successful_requests"] == 1
    assert summary["truncated_rate_among_successful"] == 1
    assert summary["successful_requests_with_unknown_finish_reason"] == 0


@pytest.mark.parametrize("payload,error", [
    (make_sse(done=False), "Stream ended without [DONE]"),
    (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: [DONE]\n\n', "without nonempty text"),
    (b'data: {"error":{"message":"private detail"}}\n\n', "invalid/error stream"),
    (b'data: invalid json\n\n', "JSONDecodeError"),
])
def test_broken_streams_record_failure(payload, error):
    result = execute_single(httpx.MockTransport(lambda _: httpx.Response(200, stream=ChunkStream([payload]))))
    assert result["status"] == "failed"
    assert error in result["error"]
    assert "private detail" not in json.dumps(result)


def test_http_failure_has_no_provider_body():
    result = execute_single(httpx.MockTransport(lambda _: httpx.Response(429, text="private rate-limit body")))
    assert result["status"] == "failed"
    assert result["error"] == "HTTP 429"
    assert "private rate-limit body" not in json.dumps(result)


def test_multiline_sse_events_and_comments():
    async def execute():
        response = httpx.Response(200, stream=ChunkStream([
            b': comment\nevent: message\ndata: {"choices":\n',
            b'data: []}\n\ndata: [DONE]\n\n',
        ]))
        return [event async for event in bm.sse_events(response)]
    assert asyncio.run(execute()) == ['{"choices":\n[]}', '[DONE]']


def test_summary_percentiles_failure_wall_time_and_partial_tokens():
    results = [
        {"status": "success", "ttft_seconds": 1.0, "end_to_end_seconds": 2.0, "completion_tokens": 10},
        {"status": "success", "ttft_seconds": 3.0, "end_to_end_seconds": 6.0, "completion_tokens": None},
        {"status": "failed", "ttft_seconds": 99.0, "end_to_end_seconds": 99.0, "completion_tokens": 100},
    ]
    summary = bm.summarize(results, 10.0)
    assert summary["ttft_seconds"] == {"p50": 2.0, "p95": 2.9}
    assert summary["end_to_end_seconds"]["p95"] == pytest.approx(5.8)
    assert summary["successful_requests_per_second"] == 0.2
    assert summary["known_completion_tokens_per_wall_second"] == 1.0
    assert summary["failed_requests"] == 1
    assert summary["requests_with_unknown_completion_tokens"] == 1
    assert summary["complete_token_accounting"] is False
    empty = bm.summarize([], 0)
    assert empty["ttft_seconds"]["p50"] is None
    assert empty["known_completion_tokens"] is None
    assert empty["success_rate"] is None
    assert empty["successful_requests_per_second"] is None


def test_benchmark_limits_concurrency_excludes_warmup_and_uses_protocol():
    state = {"active": 0, "peak": 0, "bodies": []}

    class ConcurrentStream(httpx.AsyncByteStream):
        active = False

        async def __aiter__(self):
            self.active = True
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0)
            yield make_sse()

        async def aclose(self):
            if self.active:
                self.active = False
                state["active"] -= 1

    def handler(request):
        state["bodies"].append(json.loads(request.content))
        return httpx.Response(200, stream=ConcurrentStream())

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await bm.benchmark(client, base_url="http://test/v1", model="eduqa", prompt="q",
                                      max_tokens=8, requests=5, concurrency=2, warmup=1)

    report = asyncio.run(execute())
    assert len(state["bodies"]) == 6
    assert state["peak"] == 2
    assert report["summary"]["total_requests"] == 5
    assert report["summary"]["known_completion_tokens"] == 20
    assert report["warmup"]["total"] == 1
    assert report["warmup"]["requests"][0]["request_id"] == -1
    assert report["configuration"]["concurrency"] == 2
    assert state["bodies"][0]["stream_options"] == {"include_usage": True}


def test_cli_reports_failure_and_auth_is_not_persisted(tmp_path, monkeypatch):
    real_client = httpx.AsyncClient
    observed = {}

    def client_factory(**kwargs):
        observed.update(kwargs)
        return real_client(**kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(500)))

    monkeypatch.setattr(bm.httpx, "AsyncClient", client_factory)
    monkeypatch.setenv("BENCH_TEST_KEY", "must-not-save-this-secret")
    output = tmp_path / "report.json"
    args = argparse.Namespace(api_key_env="BENCH_TEST_KEY", timeout=10, concurrency=1, requests=1,
                              warmup=1, max_tokens=8, base_url="http://test/v1", model="test", prompt="question", output=str(output))
    assert bm.run(args) == 1
    assert observed["headers"]["Authorization"] == "Bearer must-not-save-this-secret"
    text = output.read_text(encoding="utf-8")
    assert "must-not-save-this-secret" not in text
    report = json.loads(text)
    assert report["summary"]["failed_requests"] == 1
    assert report["warmup"]["failed"] == 1
    # Explicit opt-in permits replacing an earlier report; default preserves it.
    with pytest.raises(FileExistsError, match="overwrite"):
        bm.run(args)
    args.overwrite = True
    assert bm.run(args) == 1


def test_cli_refuses_existing_report_before_any_api_request(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("existing evidence", encoding="utf-8")

    def forbidden(**kwargs):
        pytest.fail("Existing report must fail before an HTTP request")

    monkeypatch.setattr(bm.httpx, "AsyncClient", forbidden)
    with pytest.raises(FileExistsError, match="overwrite"):
        bm.run(argparse.Namespace(output=str(output), timeout=10))
    assert output.read_text(encoding="utf-8") == "existing evidence"


@pytest.mark.parametrize("values", [{"requests": 0}, {"concurrency": 0}, {"warmup": -1}, {"max_tokens": 0}])
def test_invalid_configuration_rejected_without_requests(values):
    async def execute():
        options = {"base_url": "http://test/v1", "model": "test", "prompt": "q", "max_tokens": 8,
                   "requests": 1, "concurrency": 1, "warmup": 0, **values}
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("No request expected"))) as client:
            await bm.benchmark(client, **options)
    with pytest.raises(ValueError):
        asyncio.run(execute())
