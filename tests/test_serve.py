"""Serving configs: vLLM command builder, Ollama Modelfile, OpenAI shim."""

import asyncio
import importlib.util

import pytest

from distillery.llm import EchoMockClient
from distillery.serve.ollama import build_create_command, build_modelfile
from distillery.serve.openai_shim import ChatCompletionHandler
from distillery.serve.vllm_config import VLLMServeConfig


def test_vllm_command_flags_are_paired() -> None:
    config = VLLMServeConfig(
        model="student-awq/",
        served_model_name="distilled-student",
        quantization="awq",
        port=8001,
        api_key="secret-key",
    )
    command = config.build_serve_command()
    assert command[:3] == ["vllm", "serve", "student-awq/"]
    assert command[command.index("--quantization") : command.index("--quantization") + 2] == ["--quantization", "awq"]
    assert "--enable-prefix-caching" in command
    assert command[command.index("--port") : command.index("--port") + 2] == ["--port", "8001"]
    for i, arg in enumerate(command):
        if arg.startswith("--") and arg != "--enable-prefix-caching":
            assert i + 1 < len(command) and not command[i + 1].startswith("--")


def test_vllm_command_omits_unset_optionals() -> None:
    command = VLLMServeConfig(model="m/").build_serve_command()
    assert "--quantization" not in command
    assert "--served-model-name" not in command
    assert "--api-key" not in command


def test_vllm_yaml_redacts_api_key() -> None:
    config = VLLMServeConfig(model="m/", api_key="super-secret")
    rendered = config.to_yaml()
    assert "super-secret" not in rendered
    assert "${VLLM_API_KEY}" in rendered


def test_ollama_modelfile_and_commands() -> None:
    modelfile = build_modelfile(gguf_path="student-q4.gguf", temperature=0.2, num_predict=256)
    assert "FROM student-q4.gguf" in modelfile
    assert "PARAMETER temperature 0.2" in modelfile
    assert "PARAMETER num_predict 256" in modelfile
    assert 'SYSTEM """' in modelfile
    assert build_create_command("student", "Modelfile") == ["ollama", "create", "student", "-f", "Modelfile"]


def _handler() -> ChatCompletionHandler:
    return ChatCompletionHandler(EchoMockClient(), model_name="distilled-student")


def test_shim_handler_returns_openai_shape() -> None:
    payload = {
        "model": "ignored",
        "messages": [{"role": "user", "content": "Summarize the quarterly report."}],
        "temperature": 0.3,
        "seed": 5,
    }
    response = asyncio.run(_handler().handle(payload))
    assert response["id"].startswith("chatcmpl-")
    assert response["object"] == "chat.completion"
    assert response["model"] == "distilled-student"
    choice = response["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"]
    assert response["usage"]["prompt_tokens"] > 0
    assert response["usage"]["total_tokens"] == 0  # filled by the FastAPI layer


def test_shim_handler_is_deterministic_per_input() -> None:
    payload = {"messages": [{"role": "user", "content": "hello"}], "seed": 1}
    a = asyncio.run(_handler().handle(payload))
    b = asyncio.run(_handler().handle(payload))
    assert a["id"] == b["id"]
    other = asyncio.run(_handler().handle({"messages": [{"role": "user", "content": "bye"}], "seed": 1}))
    assert other["id"] != a["id"]


def test_shim_handler_rejects_empty_messages() -> None:
    with pytest.raises(ValueError, match="messages"):
        asyncio.run(_handler().handle({"messages": []}))


def test_shim_handler_works_over_streaming_client_semantics() -> None:
    # The handler only needs `complete`; verify composition with a chat-shaped input.
    payload = {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ]
    }
    response = asyncio.run(_handler().handle(payload))
    assert isinstance(response["choices"][0]["message"]["content"], str)


def test_create_app_guarded_without_fastapi() -> None:
    if importlib.util.find_spec("fastapi") is None:
        from distillery.errors import DependencyMissingError

        with pytest.raises(DependencyMissingError, match="serve"):
            from distillery.serve.openai_shim import create_app

            create_app(_handler())
    else:  # pragma: no cover - only when the serve extra is installed
        from distillery.serve.openai_shim import create_app

        app = create_app(_handler())
        assert any(getattr(route, "path", "") == "/v1/chat/completions" for route in app.routes)
