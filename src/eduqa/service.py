"""Authenticated, bounded OpenAI-compatible gateway for a local vLLM server.

The gateway never loads model weights or synthesizes fallback answers. Start
vLLM first and point EDUQA_BACKEND_URL at its /v1 endpoint.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import json
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_SYSTEM_PROMPT = (
    "你是一位耐心、严谨的中文教育问答助手。请优先解释概念与解题思路，"
    "再给出必要的推导和结论；数学表达应清晰，单位应一致。"
    "题目信息不足时先说明缺少的条件，不编造条件、事实或引用。"
    "无法确定时明确说明不确定，并建议核对教材或向老师请教。"
)
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    backend_url: str = "http://127.0.0.1:8000/v1"
    model: str = "eduqa"
    api_key: str = ""
    backend_api_key: str = ""
    timeout_seconds: float = 120.0
    max_concurrent: int = 4
    max_tokens: int = 2048
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        parsed = urlsplit(self.backend_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("EDUQA_BACKEND_URL must be an HTTP(S) URL without credentials/query/fragment")
        if not self.model.strip():
            raise ValueError("EDUQA_MODEL must not be empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("EDUQA_TIMEOUT_SECONDS must be a finite positive number")
        if self.max_concurrent < 1 or self.max_tokens < 1:
            raise ValueError("EDUQA_MAX_CONCURRENT and EDUQA_MAX_TOKENS must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            backend_url=os.getenv("EDUQA_BACKEND_URL", cls.backend_url).rstrip("/"),
            model=os.getenv("EDUQA_MODEL", cls.model),
            api_key=os.getenv("EDUQA_API_KEY", ""),
            backend_api_key=os.getenv("EDUQA_BACKEND_API_KEY", ""),
            timeout_seconds=float(os.getenv("EDUQA_TIMEOUT_SECONDS", "120")),
            max_concurrent=int(os.getenv("EDUQA_MAX_CONCURRENT", "4")),
            max_tokens=int(os.getenv("EDUQA_MAX_TOKENS", "2048")),
            system_prompt=os.getenv("EDUQA_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        )


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32768)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = Field(default=False, strict=True)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = Field(default=None, min_length=1, max_length=256)
    messages: list[Message] = Field(min_length=1, max_length=100)
    max_tokens: int | None = Field(default=None, ge=1, strict=True)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    stream: bool = False
    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def validate_conversation(self) -> "ChatRequest":
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages must contain at least one user question")
        if sum(len(message.content) for message in self.messages) > 100000:
            raise ValueError("conversation exceeds 100000 characters; start a new conversation")
        return self


class _Limiter:
    """Reject excess work immediately instead of building an unbounded queue."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self.lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    async def release(self) -> None:
        async with self.lock:
            self.active -= 1


class _ClosingStreamingResponse(StreamingResponse):
    """Close even when the ASGI server reports a disconnect by failing send()."""

    def __init__(self, *args, cleanup, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.cleanup()


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create an isolated app; an injected transport enables offline integration tests."""

    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        headers = {}
        if config.backend_api_key:
            headers["Authorization"] = f"Bearer {config.backend_api_key}"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds),
            limits=httpx.Limits(
                max_connections=config.max_concurrent + 2,
                max_keepalive_connections=config.max_concurrent + 2,
            ),
            headers=headers,
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            app.state.backend_client = client
            app.state.limiter = _Limiter(config.max_concurrent)
            yield

    app = FastAPI(title="教育问答模型服务", version="1.0.0", lifespan=lifespan)
    app.state.settings = config

    async def authenticate(authorization: str | None = Header(default=None)) -> None:
        if not config.api_key:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.encode("utf-8"), config.api_key.encode("utf-8")
        ):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz", dependencies=[Depends(authenticate)])
    async def ready(request: Request):
        try:
            result = await request.app.state.backend_client.get(
                f"{config.backend_url.rstrip('/')}/models"
            )
            result.raise_for_status()
            payload = result.json()
            models = payload.get("data", []) if isinstance(payload, dict) else []
            if not any(isinstance(item, dict) and item.get("id") == config.model for item in models):
                raise ValueError("configured model is not loaded")
        except (httpx.HTTPError, ValueError, TypeError):
            raise HTTPException(status_code=503, detail="Configured model is not ready") from None
        return {"status": "ready", "model": config.model}

    @app.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
    async def completions(body: ChatRequest, request: Request):
        if body.model is not None and body.model != config.model:
            raise HTTPException(status_code=400, detail="Requested model is not configured")
        if body.max_tokens is not None and body.max_tokens > config.max_tokens:
            raise HTTPException(
                status_code=422, detail=f"max_tokens must not exceed {config.max_tokens}"
            )
        limiter = request.app.state.limiter
        if not await limiter.acquire():
            raise HTTPException(
                status_code=429, detail="Model is busy; retry later", headers={"Retry-After": "2"}
            )

        upstream: httpx.Response | None = None
        handed_off = False
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            # Starlette cancels an in-flight iterator on browser disconnect.
            # Shield closing the backend stream and releasing its admission slot.
            with anyio.CancelScope(shield=True):
                try:
                    if upstream is not None:
                        await upstream.aclose()
                finally:
                    await limiter.release()

        try:
            messages = [message.model_dump() for message in body.messages]
            if config.system_prompt and not any(item["role"] == "system" for item in messages):
                messages.insert(0, {"role": "system", "content": config.system_prompt})
            payload = body.model_dump(exclude_none=True)
            payload.update(
                model=config.model,
                messages=messages,
                max_tokens=body.max_tokens or min(512, config.max_tokens),
            )
            backend = request.app.state.backend_client
            upstream = await backend.send(
                backend.build_request(
                    "POST", f"{config.backend_url.rstrip('/')}/chat/completions", json=payload
                ),
                stream=True,
            )
            if upstream.status_code == 429:
                raise HTTPException(status_code=429, detail="Model backend is busy", headers={"Retry-After": "2"})
            if upstream.status_code != 200:
                raise HTTPException(status_code=502, detail="Model backend rejected the request")

            if body.stream:
                if "text/event-stream" not in upstream.headers.get("content-type", "").lower():
                    raise HTTPException(status_code=502, detail="Backend did not return an event stream")

                async def events():
                    try:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk
                    except httpx.TimeoutException:
                        yield b'data: {"error":{"message":"Model backend timed out"}}\n\ndata: [DONE]\n\n'
                    except httpx.HTTPError:
                        yield b'data: {"error":{"message":"Model backend stream interrupted"}}\n\ndata: [DONE]\n\n'
                    finally:
                        await cleanup()

                result = _ClosingStreamingResponse(
                    events(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    cleanup=cleanup,
                )
                handed_off = True
                return result

            chunks = bytearray()
            async for chunk in upstream.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > MAX_RESPONSE_BYTES:
                    raise HTTPException(status_code=502, detail="Backend response exceeds size limit")
            try:
                result = json.loads(chunks)
                if not isinstance(result, dict) or not isinstance(result.get("choices"), list):
                    raise ValueError("invalid chat completion")
            except (ValueError, UnicodeError):
                raise HTTPException(status_code=502, detail="Invalid model backend response") from None
            return JSONResponse(result)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Model backend timed out") from None
        except httpx.HTTPError:
            raise HTTPException(status_code=503, detail="Model backend is unavailable") from None
        finally:
            if not handed_off:
                await cleanup()

    return app


def _serve(args) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("serve", help="启动教育问答网页与 OpenAI-compatible 网关")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.set_defaults(func=_serve)
