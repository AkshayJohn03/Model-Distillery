"""LLM access layer.

Every teacher/student interaction goes through the :class:`LLMClient` protocol so
the pipeline runs against interchangeable backends:

* :class:`OpenAICompatClient` -- any OpenAI-compatible endpoint (vLLM, Ollama, OpenAI).
* :class:`EchoMockClient` -- a deterministic offline teacher used by tests and the
  ``--offline`` pipeline. Answers are derived from the prompt text with a seeded
  RNG, so runs are fully reproducible without network access.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from distillery.config import DistillerySettings
from distillery.errors import TeacherClientError

# Distinctive multi-word phrases; deliberately specific so that ordinary
# summarization/extraction passages never trip the refusal path.
_REFUSAL_KEYWORDS = (
    "hack into",
    "steal credit card",
    "build a weapon",
    "building a weapon",
    "spread malware",
    "phishing kit",
    "evade taxes",
    "counterfeit invoices",
    "bypass a school exam",
    "manufacturing identity documents",
    "doxxing",
    "keygen",
    "intercept package",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class ChatMessage:
    """A single chat message (role: system | user | assistant)."""

    role: str
    content: str


@dataclass(frozen=True)
class Completion:
    """Result of a non-streaming completion call."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


def estimate_tokens(text: str) -> int:
    """Cheap chars/4 token estimate (good enough for accounting in offline mode)."""
    return max(1, len(text) // 4)


@runtime_checkable
class LLMClient(Protocol):
    """Async chat-completion protocol with optional streaming."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> Completion: ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> AsyncIterator[str]: ...


class OpenAICompatClient:
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "sk-local",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers, transport=transport
        )

    @classmethod
    def from_settings(cls, settings: DistillerySettings) -> OpenAICompatClient:
        return cls(
            settings.teacher_base_url,
            settings.teacher_model,
            api_key=settings.teacher_api_key,
            timeout=settings.request_timeout,
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> Completion:
        payload = self._payload(messages, temperature, top_p, max_tokens, seed, stop, stream=False)
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.HTTPError as exc:
            raise TeacherClientError(f"teacher request failed: {exc}") from exc
        return self._parse_completion(data)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, temperature, 1.0, max_tokens, seed, None, stream=True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    delta = self._parse_sse_line(line)
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise TeacherClientError(f"teacher stream failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
        stop: Sequence[str] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = list(stop)
        return payload

    def _parse_completion(self, data: Mapping[str, Any]) -> Completion:
        try:
            choice = data["choices"][0]
            text: str = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            preview = json.dumps(data)[:200]
            raise TeacherClientError(f"unexpected teacher payload: {preview}") from exc
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=str(data.get("model", self._model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "stop")),
        )

    @staticmethod
    def _parse_sse_line(line: str) -> str | None:
        line = line.strip()
        if not line.startswith("data:"):
            return None
        chunk = line[len("data:") :].strip()
        if chunk == "[DONE]":
            return None
        try:
            event = json.loads(chunk)
        except json.JSONDecodeError:
            return None
        try:
            delta: str | None = event["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError):
            return None
        return delta


class EchoMockClient:
    """Deterministic offline teacher.

    The client extracts informative sentences from the user message (everything
    after the first blank-line separated block, i.e. the "input" material),
    samples a subset as key points, and renders a structured answer. Properties:

    * Deterministic: identical ``(seed, system, user, fidelity)`` always produce
      identical text -- no network, no wall-clock dependence.
    * Realistic failure modes: higher temperatures occasionally produce degenerate
      repetitive output so the quality filter has something realistic to catch.
    * Simulated capability gap: ``fidelity`` (1.0 for the simulated teacher,
      ~0.7 for the simulated student) controls how many key points survive,
      which yields a measurable teacher-vs-student quality delta in the bench.
    """

    def __init__(self, *, fidelity: float = 1.0, model: str = "echo-mock") -> None:
        if not 0.0 < fidelity <= 1.0:
            raise ValueError("fidelity must be in (0.0, 1.0]")
        self._fidelity = fidelity
        self._model = model

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> Completion:
        text = self._answer(list(messages), temperature=temperature, seed=seed)
        if max_tokens and len(text) > max_tokens * 4:
            text = text[: max_tokens * 4]
        return Completion(
            text=text,
            model=self._model,
            prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
            completion_tokens=estimate_tokens(text),
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        text = self._answer(list(messages), temperature=temperature, seed=seed)
        for i in range(0, len(text), 24):
            yield text[i : i + 24]

    def _answer(self, messages: list[ChatMessage], *, temperature: float, seed: int | None) -> str:
        system = " ".join(m.content for m in messages if m.role == "system")
        user = messages[-1].content if messages else ""
        rng = random.Random(f"{seed}:{system}:{user}")
        lowered = user.lower()
        if any(k in lowered for k in _REFUSAL_KEYWORDS):
            return self._refusal(rng)
        points = self._key_points(user, rng)
        degenerate = rng.random() < (0.02 if temperature < 0.7 else 0.15)
        if degenerate:
            return ("Key points:\n" + f"- {points[0]}\n" * 9).strip()
        body = "\n".join(f"- {p}" for p in points)
        lead = rng.choice(["Key points:", "Here is a structured answer:", "Summary of the request:"])
        close = (
            "Prepared with the requested level of detail."
            if rng.random() < 0.5
            else "Let me know if you want a deeper dive."
        )
        if "step by step" in system.lower() or "reasoning:" in system.lower():
            steps = "\n".join(f"{i + 1}. Restated: {p}" for i, p in enumerate(points))
            return f"Reasoning:\n{steps}\n\nAnswer:\n{body}\n\n{close}"
        return f"{lead}\n{body}\n\n{close}"

    def _key_points(self, user: str, rng: random.Random) -> list[str]:
        blocks = [b.strip() for b in user.split("\n\n") if b.strip()]
        source = "\n".join(blocks[1:]) if len(blocks) > 1 else blocks[0] if blocks else user
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(source) if len(s.strip()) >= 30]
        if not sentences:
            sentences = [source.strip() or "the request"]
        n_points = max(1, round(3 * self._fidelity))
        return rng.sample(sentences, min(n_points, len(sentences)))

    @staticmethod
    def _refusal(rng: random.Random) -> str:
        leads = (
            "I can't help with that request.",
            "I'm not able to assist with this.",
            "That request falls outside what I can support.",
        )
        alternatives = (
            "If you have a legitimate version of this task, I am happy to help with safe alternatives.",
            "For lawful alternatives -- such as official channels or vendor support -- I can assist.",
            "If this is for a class or audit, I can point you to authorised resources instead.",
            "I'd be glad to help with a compliant version of the underlying goal.",
            "Reach out to the relevant provider or authority for a sanctioned path forward.",
        )
        return f"{rng.choice(leads)} {rng.choice(alternatives)}"
