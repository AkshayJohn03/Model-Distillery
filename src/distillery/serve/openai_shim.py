"""Minimal OpenAI-compatible chat-completions shim for local testing.

The :class:`ChatCompletionHandler` is framework-free (a dict-in/dict-out
function over any :class:`~distillery.llm.LLMClient`), which makes it testable
offline with :class:`~distillery.llm.EchoMockClient`. ``create_app`` wraps the
handler in FastAPI (optional ``serve`` extra) for local smoke tests and demos;
production traffic should hit vLLM's own OpenAI server instead.

Note ``created`` is pinned to 0 and ids are content-hashes: the shim is
deterministic by design (useful for replay tests), not a byte-level clone of
the OpenAI API.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from distillery.errors import DependencyMissingError
from distillery.llm import ChatMessage, Completion, LLMClient, estimate_tokens


class ChatCompletionHandler:
    """Dict-in/dict-out OpenAI chat-completions logic over any LLMClient."""

    def __init__(self, client: LLMClient, *, model_name: str = "distilled-student") -> None:
        self._client = client
        self._model_name = model_name

    async def handle(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_messages = payload.get("messages") or []
        if not raw_messages:
            raise ValueError("messages must not be empty")
        messages = [
            ChatMessage(role=str(m.get("role", "user")), content=str(m.get("content", "")))
            for m in raw_messages
        ]
        seed = payload.get("seed")
        completion: Completion = await self._client.complete(
            messages,
            temperature=float(payload.get("temperature", 0.7)),
            max_tokens=int(payload.get("max_tokens", 512)),
            seed=int(seed) if seed is not None else None,
        )
        digest = hashlib.sha256(
            json.dumps(
                {
                    "model": self._model_name,
                    "messages": [[m.role, m.content] for m in messages],
                    "temperature": payload.get("temperature", 0.7),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        return {
            "id": f"chatcmpl-{digest}",
            "object": "chat.completion",
            "created": 0,
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion.text},
                    "finish_reason": completion.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": completion.prompt_tokens or sum(estimate_tokens(m.content) for m in messages),
                "completion_tokens": completion.completion_tokens or estimate_tokens(completion.text),
                "total_tokens": 0,  # filled below
            },
        }


def create_app(handler: ChatCompletionHandler) -> Any:
    """FastAPI app exposing ``POST /v1/chat/completions`` (requires ``serve`` extra)."""
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise DependencyMissingError("openai-compatible shim", "'model-distillery[serve]'") from exc

    app = FastAPI(title="Model-Distillery serving shim", version="0.1.0")

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        response = await handler.handle(payload)
        usage = response["usage"]
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
