"""GGUF quantization via llama.cpp (command builder; requires local tooling).

Workflow
--------
1. Export the merged HF checkpoint to an f16 GGUF with ``convert_hf_to_gguf.py``
   (llama.cpp repo). For a LoRA adapter, merge first
   (``peft``: ``model.merge_and_unload()`` then ``save_pretrained``) -- quantizing
   an unmerged adapter is not supported.
2. Quantize the f16 file to a k-quant with ``llama-quantize``.

Choosing a quant type (quality vs size vs speed, 7B reference sizes):

| Type    | bpw | Size  | Notes                                              |
|---------|-----|-------|----------------------------------------------------|
| q4_0    | 4.0 | ~3.9G | legacy baseline                                    |
| q4_k_m  | 4.8 | ~4.4G | recommended default; small ppl loss vs f16         |
| q5_k_m  | 5.5 | ~5.1G | near-lossless for most tasks                       |
| q6_k    | 6.6 | ~5.9G | very small ppl delta                               |
| q8_0    | 8.5 | ~7.2G | effectively lossless, 2x smaller than f16          |

Use GGUF when serving on CPU, Apple Silicon, or via llama.cpp/Ollama. For
GPU serving with vLLM prefer AWQ/GPTQ (see ``distillery.quantize.awq``).
Perplexity-check the quantized model against a held-out slice before shipping:
quantization regressions show up on reasoning slices first.

This module only BUILDS commands; it is not executed in CI.
"""

from __future__ import annotations

from dataclasses import dataclass

KNOWN_QUANT_TYPES = (
    "q4_0",
    "q4_k_s",
    "q4_k_m",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
)


@dataclass(frozen=True)
class GGUFConfig:
    hf_model_dir: str
    gguf_f16_path: str
    quantized_path: str
    quant_type: str = "q4_k_m"
    converter: str = "convert_hf_to_gguf.py"  # path to llama.cpp converter script
    quantize_bin: str = "llama-quantize"  # path to built llama.cpp binary

    def __post_init__(self) -> None:
        if self.quant_type not in KNOWN_QUANT_TYPES:
            raise ValueError(f"quant_type must be one of {KNOWN_QUANT_TYPES}, got {self.quant_type!r}")

    def build_commands(self) -> list[list[str]]:
        """Return the two-stage convert + quantize command sequence."""
        return [
            ["python", self.converter, self.hf_model_dir, "--outfile", self.gguf_f16_path, "--outtype", "f16"],
            [self.quantize_bin, self.gguf_f16_path, self.quantized_path, self.quant_type],
        ]

    def build_docs(self) -> str:
        """Render the runbook for these exact commands."""
        commands = "\n".join("  " + " ".join(cmd) for cmd in self.build_commands())
        return (
            "GGUF quantization runbook (requires llama.cpp locally; NOT executed in CI):\n"
            f"{commands}\n"
            "Then validate: run llama-perplexity against a held-out slice and compare "
            "against the f16 baseline before deploying."
        )
