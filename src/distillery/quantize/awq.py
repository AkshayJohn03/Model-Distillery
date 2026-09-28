"""AWQ 4-bit quantization via autoawq (builder + runner; requires local tooling).

Activation-aware Weight Quantization keeps a small set of salient weights (those
with large activation magnitudes) in higher precision while quantizing the rest
to 4-bit, which preserves quality far better than naive RTN at the same bitrate.

When to use AWQ
---------------
* GPU inference with vLLM (``--quantization awq``), TGI, or any CUDA runtime.
* Calibration matters: pass representative prompts (ideally drawn from the SAME
  distribution as production traffic -- here, the distilled task mix). Poor
  calibration data is the #1 cause of AWQ quality regressions.

Recommended settings: ``w_bit=4``, ``q_group_size=128``, ``zero_point=True``,
``version="gemm"`` (fastest batched inference; use ``gemv`` for batch-1 edge
cases). Expect ~3.3x VRAM reduction vs fp16 and a modest throughput gain from
kernel efficiency.

Usage (after installing the ``torch`` extra plus ``autoawq``)::

    python -m distillery.quantize.awq \\
        --model-path merged-student/ --out-dir student-awq/ \\
        --calibration-path data/demo_prompt_pool.jsonl

This module builds and can run the command locally; it is NOT executed in CI
(no GPU, no autoawq in the test environment).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

VERSIONS = ("gemm", "gemv")


@dataclass(frozen=True)
class AWQConfig:
    model_path: str
    out_dir: str
    w_bit: int = 4
    q_group_size: int = 128
    zero_point: bool = True
    version: str = "gemm"
    calibration_path: str | None = None
    max_calib_seq_len: int = 512

    def __post_init__(self) -> None:
        if self.w_bit not in (2, 3, 4, 8):
            raise ValueError(f"w_bit must be one of (2, 3, 4, 8), got {self.w_bit}")
        if self.q_group_size not in (32, 64, 128, 256):
            raise ValueError(f"q_group_size must be one of (32, 64, 128, 256), got {self.q_group_size}")
        if self.version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}, got {self.version!r}")

    def build_command(self) -> list[str]:
        """Well-formed CLI invocation of this module's runner (``main`` below)."""
        command = [
            "python",
            "-m",
            "distillery.quantize.awq",
            "--model-path",
            self.model_path,
            "--out-dir",
            self.out_dir,
            "--w-bit",
            str(self.w_bit),
            "--q-group-size",
            str(self.q_group_size),
            "--version",
            self.version,
            "--max-calib-seq-len",
            str(self.max_calib_seq_len),
        ]
        if not self.zero_point:
            command.append("--no-zero-point")
        if self.calibration_path:
            command += ["--calibration-path", self.calibration_path]
        return command


def _quantize(config: AWQConfig) -> None:  # pragma: no cover - requires autoawq + GPU
    import json

    from awq import AutoAWQ

    texts: list[str] | None = None
    if config.calibration_path:
        with open(config.calibration_path, encoding="utf-8") as handle:
            texts = [
                f"{row.get('instruction', '')} {row.get('input', '')}".strip()
                for row in (json.loads(line) for line in handle if line.strip())
            ][:128]

    model = AutoAWQ.from_pretrained(config.model_path)
    quant_config = {
        "zero_point": config.zero_point,
        "q_group_size": config.q_group_size,
        "w_bit": config.w_bit,
        "version": config.version,
    }
    model.quantize(model.tokenizer, quant_config=quant_config, calib_data=texts)
    model.save_quantized(config.out_dir)
    model.tokenizer.save_pretrained(config.out_dir)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - requires autoawq
    parser = argparse.ArgumentParser(
        prog="python -m distillery.quantize.awq",
        description="AWQ-quantize a merged student checkpoint (requires autoawq).",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    parser.add_argument("--version", default="gemm", choices=VERSIONS)
    parser.add_argument("--max-calib-seq-len", type=int, default=512)
    parser.add_argument("--no-zero-point", action="store_true")
    parser.add_argument("--calibration-path", default=None)
    args = parser.parse_args(argv)
    config = AWQConfig(
        model_path=args.model_path,
        out_dir=args.out_dir,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
        zero_point=not args.no_zero_point,
        version=args.version,
        calibration_path=args.calibration_path,
        max_calib_seq_len=args.max_calib_seq_len,
    )
    _quantize(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
