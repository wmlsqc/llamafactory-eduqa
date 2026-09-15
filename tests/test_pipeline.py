"""Cross-module HTTP integration, using test-only backend responses.

These tests verify gateway/client interoperability, not model quality, GPU
performance or real network TTFT (ASGITransport buffers the response body).
"""

import asyncio
import json

from fastapi.testclient import TestClient
import httpx

from eduqa import benchmark as bm
from eduqa import evaluation
from eduqa.service import DEFAULT_SYSTEM_PROMPT, Settings, create_app


def test_evaluation_through_authenticated_gateway_injects_education_prompt():
    records = [
        {"id": "integration-1", "question": "测试问题一", "reference": "测试参考一"},
        {"id": "integration-2", "question": "测试问题二", "reference": "测试参考二"},
    ]
    answers = {record["question"]: record["reference"] for record in records}
    backend_requests = []
    usage = {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48}

    def backend(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer backend-eval-test-key"
        payload = json.loads(request.content)
        backend_requests.append(payload)
        assert payload["model"] == "eduqa"
        assert payload["stream"] is False
        assert payload["temperature"] == 0
        assert payload["max_tokens"] == 128
        assert payload["messages"][0] == {
            "role": "system", "content": DEFAULT_SYSTEM_PROMPT,
        }
        question = payload["messages"][1]["content"]
        return httpx.Response(200, json={
            "id": "integration-evaluation", "object": "chat.completion",
            "model": "eduqa",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answers[question]},
                         "finish_reason": "stop"}],
            "usage": usage,
        })

    app = create_app(
        Settings(api_key="gateway-eval-test-key", backend_api_key="backend-eval-test-key"),
        httpx.MockTransport(backend),
    )
    with TestClient(app) as client:
        rejected = evaluation.evaluate_records(
            records[:1], client, base_url="http://testserver/v1", model="eduqa", max_tokens=128,
        )
        assert rejected[0]["status"] == "failed"
        assert rejected[0]["error"] == "HTTP 401"
        assert not backend_requests

        client.headers["Authorization"] = "Bearer gateway-eval-test-key"
        results = evaluation.evaluate_records(
            records, client, base_url="http://testserver/v1", model="eduqa", max_tokens=128,
        )
        summary = evaluation.summarize(results)
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["metrics_on_successful_requests_only"]["normalized_exact_match"] == 1.0
        assert [result["prediction"] for result in results] == [record["reference"] for record in records]
        assert all(result["usage"] == usage for result in results)
        assert len(backend_requests) == 2
        assert app.state.limiter.active == 0
    serialized = json.dumps(results)
    assert "gateway-eval-test-key" not in serialized
    assert "backend-eval-test-key" not in serialized


def test_benchmark_through_authenticated_gateway_preserves_sse_usage():
    async def run():
        streams = []
        backend_requests = []
        frames = [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "测试回答"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46}},
        ]
        sse = ("".join("data: " + json.dumps(frame, ensure_ascii=False) + "\r\n\r\n" for frame in frames)
               + "data: [DONE]\r\n\r\n").encode("utf-8")

        class BackendStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                # Deliberately split UTF-8/SSE frames across chunks and allow
                # concurrent requests to interleave at the ASGI boundary.
                for offset in range(0, len(sse), 7):
                    await asyncio.sleep(0)
                    yield sse[offset:offset + 7]

            async def aclose(self):
                self.closed = True

        def backend(request):
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer backend-bench-test-key"
            payload = json.loads(request.content)
            backend_requests.append(payload)
            assert payload["model"] == "eduqa"
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": True}
            assert payload["messages"] == [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "用于网关整合测试的问题"},
            ]
            stream = BackendStream()
            streams.append(stream)
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

        app = create_app(
            Settings(api_key="gateway-bench-test-key", backend_api_key="backend-bench-test-key",
                     max_concurrent=2),
            httpx.MockTransport(backend),
        )
        options = {"base_url": "http://gateway/v1", "model": "eduqa",
                   "prompt": "用于网关整合测试的问题", "max_tokens": 64}
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway",
            ) as client:
                rejected = await bm.benchmark(client, **options, requests=1, concurrency=1, warmup=0)
                assert rejected["summary"]["failed_requests"] == 1
                assert rejected["requests"][0]["error"] == "HTTP 401"
                assert not backend_requests

                client.headers["Authorization"] = "Bearer gateway-bench-test-key"
                report = await bm.benchmark(client, **options, requests=3, concurrency=2, warmup=1)
                assert report["summary"]["successful_requests"] == 3
                assert report["summary"]["failed_requests"] == 0
                assert report["summary"]["known_completion_tokens"] == 18
                assert report["summary"]["complete_token_accounting"] is True
                assert report["warmup"]["total"] == 1
                assert report["warmup"]["failed"] == 0
                assert len(backend_requests) == 4  # One warmup is excluded from totals.
                assert all(row["completion_tokens"] == 6 for row in report["requests"])
                assert all(row["token_count_source"] == "server_usage" for row in report["requests"])
                assert all(row["received_done"] and row["finish_reason"] == "stop" for row in report["requests"])
                assert all(row["output_characters"] == len("测试回答") for row in report["requests"])
                assert all(stream.closed for stream in streams)
                assert app.state.limiter.active == 0
        serialized = json.dumps(report)
        assert "gateway-bench-test-key" not in serialized
        assert "backend-bench-test-key" not in serialized

    asyncio.run(run())
