"""Offline integration tests validate transport behavior, not model quality."""

import asyncio
import json

from fastapi.testclient import TestClient
import httpx
import pytest
from starlette.requests import ClientDisconnect

from eduqa.service import DEFAULT_SYSTEM_PROMPT, Settings, create_app


QUESTION = {"messages": [{"role": "user", "content": "为什么有四季？"}]}
ANSWER = {"id": "test-completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "测试传输内容"}, "finish_reason": "stop"}]}


class TrackingStream(httpx.AsyncByteStream):
    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


def client(handler, **settings):
    return TestClient(create_app(Settings(**settings), httpx.MockTransport(handler)))


def test_health_is_local_and_home_is_chinese():
    def unreachable(request):
        raise AssertionError("liveness and static page must not contact backend")

    with client(unreachable, api_key="private") as session:
        assert session.get("/healthz").json() == {"status": "ok"}
        page = session.get("/")
        assert page.status_code == 200
        assert "教育问答" in page.text


def test_ready_requires_configured_model_and_backend_key():
    def models(request):
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer backend-secret"
        return httpx.Response(200, json={"data": [{"id": "eduqa"}]})

    with client(models, api_key="gateway-secret", backend_api_key="backend-secret") as session:
        assert session.get("/readyz").status_code == 401
        response = session.get("/readyz", headers={"Authorization": "Bearer gateway-secret"})
        assert response.json() == {"status": "ready", "model": "eduqa"}


@pytest.mark.parametrize("payload", [{"data": [{"id": "different"}]}, {}, [], {"data": None}])
def test_ready_rejects_missing_model(payload):
    with client(lambda request: httpx.Response(200, json=payload)) as session:
        assert session.get("/readyz").status_code == 503


def test_completion_injects_prompt_and_defaults_and_keeps_secrets_separate():
    def completion(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer model-key"
        payload = json.loads(request.content)
        assert payload["model"] == "eduqa"
        assert payload["max_tokens"] == 512
        assert payload["messages"][0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        assert payload["messages"][1:] == QUESTION["messages"]
        return httpx.Response(200, json=ANSWER)

    with client(completion, api_key="web-key", backend_api_key="model-key") as session:
        assert session.post("/v1/chat/completions", json=QUESTION).status_code == 401
        assert session.post("/v1/chat/completions", json=QUESTION, headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = session.post("/v1/chat/completions", json=QUESTION, headers={"Authorization": "Bearer web-key"})
        assert response.json() == ANSWER
        assert session.app.state.limiter.active == 0


def test_custom_prompt_is_preserved():
    def completion(request):
        payload = json.loads(request.content)
        assert payload["messages"][0]["content"] == "自定义教学提示"
        assert len(payload["messages"]) == 2
        return httpx.Response(200, json=ANSWER)

    with client(completion) as session:
        response = session.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "自定义教学提示"}, *QUESTION["messages"]]})
        assert response.status_code == 200


@pytest.mark.parametrize("payload,status", [
    ({"messages": []}, 422),
    ({"messages": [{"role": "tool", "content": "x"}]}, 422),
    ({"messages": [{"role": "user", "content": ""}]}, 422),
    ({"messages": [{"role": "user", "content": "x" * 32769}]}, 422),
    ({"messages": [{"role": "assistant", "content": "hello"}]}, 422),
    ({**QUESTION, "max_tokens": 2049}, 422),
    ({**QUESTION, "max_tokens": True}, 422),
    ({**QUESTION, "temperature": 2.1}, 422),
    ({**QUESTION, "tools": []}, 422),
    ({**QUESTION, "stream_options": {"include_usage": True}}, 422),
    ({**QUESTION, "stream": True, "stream_options": {"include_usage": "yes"}}, 422),
    ({**QUESTION, "stream": True, "stream_options": {"include_usage": True, "unsupported": True}}, 422),
    ({**QUESTION, "model": "unknown-model"}, 400),
])
def test_invalid_input_never_reaches_backend(payload, status):
    def reject(request):
        raise AssertionError("invalid input reached backend")

    with client(reject) as session:
        assert session.post("/v1/chat/completions", json=payload).status_code == status


@pytest.mark.parametrize("error,status", [(httpx.ConnectError("secret backend URL"), 503), (httpx.ReadTimeout("private info"), 504)])
def test_backend_failures_are_sanitized_and_release_slot(error, status):
    def broken(request):
        raise error

    with client(broken) as session:
        response = session.post("/v1/chat/completions", json=QUESTION)
        assert response.status_code == status
        assert "secret" not in response.text and "private" not in response.text
        assert session.app.state.limiter.active == 0


@pytest.mark.parametrize("upstream_status,expected", [(400, 502), (401, 502), (429, 429), (500, 502), (302, 502)])
def test_backend_status_is_not_exposed(upstream_status, expected):
    stream = TrackingStream([b"internal secret"])
    with client(lambda request: httpx.Response(upstream_status, stream=stream)) as session:
        result = session.post("/v1/chat/completions", json=QUESTION)
        assert result.status_code == expected
        assert "internal secret" not in result.text
        assert stream.closed
        assert session.app.state.limiter.active == 0


def test_stream_forwards_chunk_boundaries_and_closes_connection():
    expected = 'data: {"choices":[{"delta":{"content":"学习"}}]}\n\ndata: [DONE]\n\n'.encode()
    # Includes a split inside a multibyte character; gateway preserves raw payload.
    chunks = [expected[:44], expected[44:47], expected[47:]]
    stream = TrackingStream(chunks)
    with client(lambda request: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)) as session:
        response = session.post("/v1/chat/completions", json={**QUESTION, "stream": True})
        assert response.content == expected
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        assert stream.closed
        assert session.app.state.limiter.active == 0


def test_stream_error_is_sse_and_slot_is_reusable():
    stream = TrackingStream([b'data: {"choices":[]}\n\n'], httpx.ReadTimeout("secret trace"))
    with client(lambda request: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)) as session:
        response = session.post("/v1/chat/completions", json={**QUESTION, "stream": True})
        assert response.status_code == 200
        assert '"error"' in response.text
        assert "data: [DONE]" in response.text
        assert "secret trace" not in response.text
        assert stream.closed
        assert session.app.state.limiter.active == 0


def test_stream_usage_options_are_forwarded_and_usage_chunk_is_preserved():
    usage = {"prompt_tokens": 35, "completion_tokens": 12, "total_tokens": 47}
    chunks = [
        b'data: {"choices":[{"delta":{"content":"test"}}]}\n\n',
        ("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode(),
        b"data: [DONE]\n\n",
    ]
    stream = TrackingStream(chunks)

    def completion(request):
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

    with client(completion) as session:
        response = session.post(
            "/v1/chat/completions",
            json={**QUESTION, "stream": True, "stream_options": {"include_usage": True}},
        )
        assert response.status_code == 200
        assert response.content == b"".join(chunks)
        events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert json.loads(events[-2])["usage"] == usage
        assert events[-1] == "[DONE]"
        assert stream.closed
        assert session.app.state.limiter.active == 0


def test_backend_invalid_json_and_stream_type():
    with client(lambda request: httpx.Response(200, text="not json")) as session:
        assert session.post("/v1/chat/completions", json=QUESTION).status_code == 502
        assert session.post("/v1/chat/completions", json={**QUESTION, "stream": True}).status_code == 502
        assert session.app.state.limiter.active == 0


def test_concurrency_limit_returns_429_then_recovers():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow(request):
            entered.set()
            await release.wait()
            return httpx.Response(200, json=ANSWER)

        app = create_app(Settings(max_concurrent=1), httpx.MockTransport(slow))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as session:
                first = asyncio.create_task(session.post("/v1/chat/completions", json=QUESTION))
                await asyncio.wait_for(entered.wait(), 2)
                rejected = await session.post("/v1/chat/completions", json=QUESTION)
                assert rejected.status_code == 429
                assert rejected.headers["retry-after"] == "2"
                release.set()
                assert (await first).status_code == 200
                assert app.state.limiter.active == 0
                assert (await session.post("/v1/chat/completions", json=QUESTION)).status_code == 200

    asyncio.run(run())


@pytest.mark.parametrize("asgi_spec", ["2.3", "2.4"])
def test_browser_disconnect_closes_upstream_and_releases_slot(asgi_spec):
    async def run():
        started = asyncio.Event()
        never = asyncio.Event()

        class EndlessStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b'data: {"choices":[]}\n\n'
                await never.wait()

            async def aclose(self):
                self.closed = True

        stream = EndlessStream()
        app = create_app(Settings(max_concurrent=1), httpx.MockTransport(lambda request: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)))
        body = json.dumps({**QUESTION, "stream": True}).encode()
        sent_request = False

        async def receive():
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": body, "more_body": False}
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                started.set()
                if asgi_spec == "2.4":
                    raise OSError("browser disconnected")

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": asgi_spec}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 1234), "server": ("test", 80)}
        async with app.router.lifespan_context(app):
            if asgi_spec == "2.4":
                with pytest.raises(ClientDisconnect):
                    await asyncio.wait_for(app(scope, receive, send), 3)
            else:
                await asyncio.wait_for(app(scope, receive, send), 3)
            assert stream.closed
            assert app.state.limiter.active == 0

    asyncio.run(run())


@pytest.mark.parametrize("options", [{"backend_url": "file:///tmp/model"}, {"backend_url": "http://user:secret@host/v1"}, {"max_tokens": 0}, {"max_concurrent": 0}, {"timeout_seconds": float("nan")}])
def test_invalid_settings_fail_early(options):
    with pytest.raises(ValueError):
        Settings(**options)
