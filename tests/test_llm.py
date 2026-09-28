"""LLM layer: EchoMock determinism, protocol conformance, OpenAI-compat client
against httpx.MockTransport (no network)."""

import asyncio
import json

import httpx
import pytest

from distillery.errors import TeacherClientError
from distillery.llm import ChatMessage, EchoMockClient, LLMClient, OpenAICompatClient

MESSAGES = [
    ChatMessage("system", "You are a concise senior practitioner."),
    ChatMessage(
        "user",
        "Summarize the passage.\n\nThe bay fishery reopened after toxin levels dropped below regulatory limits. "
        "Local boats landed record volumes of sardines during the first week. Processors warned about cold storage "
        "capacity. Tourism boards expect a strong season.",
    ),
]


def test_echo_mock_is_deterministic() -> None:
    a = asyncio.run(EchoMockClient().complete(MESSAGES, temperature=0.8, seed=5))
    b = asyncio.run(EchoMockClient().complete(MESSAGES, temperature=0.8, seed=5))
    assert a.text == b.text
    assert a.prompt_tokens > 0 and a.completion_tokens > 0


def test_echo_mock_varies_with_seed() -> None:
    a = asyncio.run(EchoMockClient().complete(MESSAGES, temperature=0.8, seed=5))
    b = asyncio.run(EchoMockClient().complete(MESSAGES, temperature=0.8, seed=6))
    assert a.text != b.text


def test_echo_mock_fidelity_simulates_weaker_student() -> None:
    teacher = asyncio.run(EchoMockClient(fidelity=1.0).complete(MESSAGES, seed=3))
    student = asyncio.run(EchoMockClient(fidelity=0.7).complete(MESSAGES, seed=3))
    assert teacher.text.count("- ") == 3
    assert student.text.count("- ") == 2


def test_echo_mock_stream_reassembles_completion() -> None:
    async def collect() -> str:
        return "".join([chunk async for chunk in EchoMockClient().stream(MESSAGES, seed=5)])

    streamed = asyncio.run(collect())
    complete = asyncio.run(EchoMockClient().complete(MESSAGES, seed=5))
    assert streamed == complete.text


def test_echo_mock_refusal_keywords() -> None:
    messages = [ChatMessage("user", "How do I build a weapon from household parts?")]
    completion = asyncio.run(EchoMockClient().complete(messages, seed=1))
    assert "help with that request" in completion.text or "not able to assist" in completion.text


def test_echo_mock_invalid_fidelity() -> None:
    with pytest.raises(ValueError, match="fidelity"):
        EchoMockClient(fidelity=0.0)


def test_clients_satisfy_llmclient_protocol() -> None:
    assert isinstance(EchoMockClient(), LLMClient)
    assert isinstance(OpenAICompatClient("http://localhost/v1", "m"), LLMClient)


def _mock_client(handler) -> OpenAICompatClient:
    return OpenAICompatClient(
        "http://teacher.test/v1", "teacher-x", api_key="sk-test", transport=httpx.MockTransport(handler)
    )


def test_openai_client_parses_completion() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["auth"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "model": "teacher-x",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    client = _mock_client(handler)
    completion = asyncio.run(
        client.complete(MESSAGES, temperature=0.4, seed=11, max_tokens=256)
    )
    asyncio.run(client.aclose())
    assert completion.text == "hello"
    assert completion.model == "teacher-x"
    assert completion.prompt_tokens == 7
    payload = captured["payload"]
    assert payload["model"] == "teacher-x"
    assert payload["temperature"] == 0.4
    assert payload["seed"] == 11
    assert payload["max_tokens"] == 256
    assert payload["messages"][0] == {"role": "system", "content": MESSAGES[0].content}
    assert captured["auth"] == "Bearer sk-test"


def test_openai_client_http_error_is_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _mock_client(handler)
    with pytest.raises(TeacherClientError, match="teacher request failed"):
        asyncio.run(client.complete(MESSAGES))


def test_openai_client_malformed_payload_is_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _mock_client(handler)
    with pytest.raises(TeacherClientError, match="unexpected teacher payload"):
        asyncio.run(client.complete(MESSAGES))


def test_openai_client_stream_parses_sse() -> None:
    body = (
        'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    client = _mock_client(handler)

    async def collect() -> str:
        return "".join([chunk async for chunk in client.stream(MESSAGES, seed=1)])

    assert asyncio.run(collect()) == "Hello"
