"""Ollama serving path for GGUF-quantized students.

Ollama is the zero-ops option for local/edge serving: wrap the GGUF artifact in
a Modelfile, ``ollama create``, and the model is reachable on
``http://localhost:11434`` with an OpenAI-compatible API at ``/v1`` (usable
directly by ``distillery.llm.OpenAICompatClient``).

Typical flow::

    ollama serve
    ollama create distilled-student -f Modelfile
    ollama run distilled-student "Summarize: ..."
"""

from __future__ import annotations


def build_modelfile(
    *,
    gguf_path: str,
    system: str = "You are a concise, accurate assistant.",
    temperature: float = 0.3,
    num_predict: int = 512,
    stop: list[str] | None = None,
) -> str:
    """Render an Ollama Modelfile for a local GGUF artifact."""
    lines = [
        f"FROM {gguf_path}",
        "",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER num_predict {num_predict}",
        "",
        f'SYSTEM """{system}"""',
    ]
    if stop:
        for token in stop:
            lines.append(f'PARAMETER stop "{token}"')
    return "\n".join(lines) + "\n"


def build_create_command(model_name: str, modelfile_path: str) -> list[str]:
    return ["ollama", "create", model_name, "-f", modelfile_path]


def build_serve_command() -> list[str]:
    return ["ollama", "serve"]
