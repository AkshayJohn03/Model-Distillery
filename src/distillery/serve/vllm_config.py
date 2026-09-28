"""vLLM serving configuration for the quantized student.

Builds the canonical ``vllm serve`` command and a YAML deployment descriptor.
Key production flags and their trade-offs:

* ``--quantization awq`` -- required for AWQ checkpoints; vLLM loads the packed
  4-bit weights directly (~3.3x VRAM reduction vs fp16, higher throughput).
* ``--max-model-len`` -- KV-cache size scales linearly with this; set it to the
  real traffic ceiling, not the model's maximum, or a few long requests will
  evict the whole cache and destroy batching efficiency.
* ``--gpu-memory-utilization`` -- fraction of VRAM for weights + KV cache; 0.90
  is a good default on dedicated cards, lower when co-tenanted.
* ``--enable-prefix-caching`` -- huge win for chat-style traffic that resends a
  static system prompt: shared prefixes are computed once.
* ``--tensor-parallel-size`` -- split across GPUs when a model does not fit;
  adds NCCL overhead, so only enable when needed.

Once served, the endpoint speaks the OpenAI protocol, which is exactly what
``distillery.llm.OpenAICompatClient`` (and the OpenAI SDK) expect::

    curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \\
      -d '{"model": "distilled-student", "messages": [{"role": "user", "content": "hi"}]}'
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class VLLMServeConfig:
    model: str
    served_model_name: str | None = None
    quantization: str | None = None  # "awq" | "gptq" | None (fp16/bf16)
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True
    port: int = 8000
    api_key: str | None = None

    def build_serve_command(self) -> list[str]:
        """Well-formed ``vllm serve`` argv (flag/value pairs, no blanks)."""
        command = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
        ]
        if self.quantization:
            command += ["--quantization", self.quantization]
        if self.enable_prefix_caching:
            command.append("--enable-prefix-caching")
        if self.served_model_name:
            command += ["--served-model-name", self.served_model_name]
        if self.api_key:
            command += ["--api-key", self.api_key]
        return command

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "model": self.model,
            "port": self.port,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enable_prefix_caching": self.enable_prefix_caching,
        }
        if self.quantization:
            data["quantization"] = self.quantization
        if self.served_model_name:
            data["served_model_name"] = self.served_model_name
        if self.api_key:
            data["api_key"] = "${VLLM_API_KEY}"  # never serialize secrets
        return data

    def to_yaml(self) -> str:
        """Deployment-descriptor YAML (secrets redacted)."""
        return yaml.safe_dump(self.as_dict(), sort_keys=False)
