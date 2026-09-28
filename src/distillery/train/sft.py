"""QLoRA SFT training: dataclass configs, plan estimation and the torch recipe.

Everything needed for **plan mode** (token/step/memory/GPU-hour estimates and the
markdown training plan) is pure stdlib and fully tested. The actual QLoRA run
(:func:`run_qlora_training`) lazily imports torch/transformers/peft/bitsandbytes
and raises :class:`DependencyMissingError` with an install hint when absent, so code
paths degrade gracefully and CI exercises the deterministic plan instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from distillery.data.sft import SFTExample
from distillery.errors import DependencyMissingError

# Rough architecture table (hidden size, layers) for common open-weight scales.
# Estimates only -- exact per-model numbers vary by architecture.
_ARCH_TABLE: dict[int, tuple[int, int]] = {
    1: (2048, 22),
    3: (3200, 26),
    7: (4096, 32),
    13: (5120, 40),
    70: (8192, 80),
}
# Rough seconds/optimizer-step on one A100-80GB at ~2k packed tokens and
# effective batch 16; used only for GPU-hour estimates in plan mode.
_STEP_SECONDS: dict[int, float] = {1: 0.5, 3: 1.1, 7: 2.2, 13: 4.0, 70: 16.0}


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class QuantConfig:
    """bitsandbytes 4-bit (QLoRA) settings."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    packing: bool = False
    max_seq_len: int = 2048
    bf16: bool = True
    seed: int = 42
    logging_steps: int = 10
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)

    @property
    def effective_batch(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def estimate_tokens(
    examples: Sequence[SFTExample], *, chars_per_token: float = 4.0, tokens_per_message: int = 4
) -> int:
    """Estimate total training tokens (chars/4 heuristic + per-message overhead)."""
    total = 0
    for ex in examples:
        for message in ex.messages:
            total += int(len(message.content) / chars_per_token) + tokens_per_message
    return max(1, total)


def estimate_lora_params(base_params: float, hidden: int, layers: int, lora: LoRAConfig) -> int:
    """LoRA adds ``r*(d_in + d_out)`` params per adapted matrix (all d x d here)."""
    return 2 * lora.r * hidden * layers * len(lora.target_modules)


def estimate_memory_gb(
    params_b: float, hidden: int, layers: int, config: TrainingConfig, *, batch: int | None = None
) -> float:
    """First-order QLoRA memory estimate in GiB.

    Components (documented approximations):
    * base weights: NF4 ~0.5 bytes/param + ~10% quant-constant overhead
      (double quantization trims this further);
    * adapters: bf16 weights (2B) + fp32 gradients (4B) + AdamW fp32 moments (8B);
    * activations: batch x seq x hidden x layers x 16 bytes, a gradient-
      checkpointing heuristic;
    * flat 2 GiB for CUDA context and fragmentation.
    """
    batch = config.per_device_train_batch_size if batch is None else batch
    params = params_b * 1e9
    lora_params = estimate_lora_params(params, hidden, layers, config.lora)
    weights = params * 0.55
    adapters = lora_params * (2 + 4 + 8)
    activations = batch * config.max_seq_len * hidden * layers * 16
    overhead = 2 * 2**30
    return round((weights + adapters + activations + overhead) / 2**30, 1)


@dataclass(frozen=True)
class ModelPlanRow:
    model_size_b: int
    hidden_size: int
    n_layers: int
    lora_params_m: float
    est_memory_gb: float
    est_step_seconds: float
    est_gpu_hours: float


@dataclass(frozen=True)
class TrainingPlan:
    n_train: int
    n_val: int
    epochs: int
    effective_batch: int
    steps: int
    est_tokens: int
    config: TrainingConfig
    rows: list[ModelPlanRow] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# QLoRA Training Plan",
            "",
            "| Setting | Value |",
            "| --- | ---: |",
            f"| train examples | {self.n_train} |",
            f"| val examples | {self.n_val} |",
            f"| epochs | {self.epochs} |",
            f"| effective batch (bs x grad_accum) | {self.effective_batch} "
            f"({self.config.per_device_train_batch_size} x "
            f"{self.config.gradient_accumulation_steps}) |",
            f"| optimizer steps | {self.steps} |",
            f"| estimated tokens (all epochs) | {self.est_tokens:,} |",
            f"| learning rate | {self.config.learning_rate:g} ({self.config.lr_scheduler_type}, "
            f"warmup {self.config.warmup_ratio:g}) |",
            f"| LoRA | r={self.config.lora.r}, alpha={self.config.lora.lora_alpha}, "
            f"dropout={self.config.lora.lora_dropout}, "
            f"targets={'/'.join(self.config.lora.target_modules)} |",
            f"| quantization | {self.config.quant.bnb_4bit_quant_type} "
            f"(double quant={self.config.quant.bnb_4bit_use_double_quant}, "
            f"compute={self.config.quant.bnb_4bit_compute_dtype}) |",
            f"| seed | {self.config.seed} |",
            "",
            "## Size / Cost Envelope (single A100-80GB, estimates)",
            "",
            "| Model | LoRA params | Est. VRAM | s/step | GPU-hours |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in self.rows:
            lines.append(
                f"| {row.model_size_b}B | {row.lora_params_m:.1f}M | {row.est_memory_gb:.1f} GB "
                f"| {row.est_step_seconds:.1f} | {row.est_gpu_hours:.2f} |"
            )
        lines += [
            "",
            "Memory model: NF4 base (~0.55 B/param incl. quant constants) + fp32 AdamW/gradients "
            "on the LoRA adapters + activation heuristic (grad checkpointing) + 2 GiB overhead. "
            "GPU-hours = steps x seconds/step; tune per hardware.",
            "",
            "Packing is "
            + ("ENABLED" if self.config.packing else "disabled")
            + "; with packing, step counts drop roughly by the sequence-packing factor.",
        ]
        return "\n".join(lines)


def build_training_plan(
    train: Sequence[SFTExample],
    val: Sequence[SFTExample],
    *,
    config: TrainingConfig | None = None,
    model_sizes: Sequence[int] = (1, 3, 7, 13, 70),
) -> TrainingPlan:
    """Deterministic training-plan estimation (no torch required)."""
    cfg = config or TrainingConfig()
    steps = max(1, math.ceil(len(train) * cfg.epochs / cfg.effective_batch))
    est_tokens = estimate_tokens(train) * cfg.epochs
    rows: list[ModelPlanRow] = []
    for size in model_sizes:
        if size not in _ARCH_TABLE or size not in _STEP_SECONDS:
            raise ValueError(f"no architecture estimate for {size}B")
        hidden, layers = _ARCH_TABLE[size]
        lora_params = estimate_lora_params(size * 1e9, hidden, layers, cfg.lora)
        step_seconds = _STEP_SECONDS[size]
        rows.append(
            ModelPlanRow(
                model_size_b=size,
                hidden_size=hidden,
                n_layers=layers,
                lora_params_m=lora_params / 1e6,
                est_memory_gb=estimate_memory_gb(size, hidden, layers, cfg),
                est_step_seconds=step_seconds,
                est_gpu_hours=round(steps * step_seconds / 3600.0, 3),
            )
        )
    return TrainingPlan(
        n_train=len(train),
        n_val=len(val),
        epochs=cfg.epochs,
        effective_batch=cfg.effective_batch,
        steps=steps,
        est_tokens=est_tokens,
        config=cfg,
        rows=rows,
    )


def run_qlora_training(
    train: Sequence[SFTExample],
    val: Sequence[SFTExample],
    config: TrainingConfig,
    *,
    model_name: str,
    output_dir: str | Path,
) -> None:
    """Reference QLoRA recipe (requires the ``torch`` extra; not exercised in CI).

    Prompt tokens are masked to -100 so the loss is computed on completions
    only. Note the mask assumes the chat template prefixes the completion; some
    templates interleave, so verify masking per model before real runs. Packing
    is honored by TRL's SFTTrainer rather than vanilla Trainer; keep
    ``config.packing`` False here or switch the Trainer accordingly.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise DependencyMissingError("qlora-training", "'model-distillery[torch]'") from exc

    compute_dtype = getattr(torch, config.quant.bnb_4bit_compute_dtype)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config.quant.load_in_4bit,
        bnb_4bit_quant_type=config.quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=list(config.lora.target_modules),
            task_type="CAUSAL_LM",
        ),
    )

    def encode(example: SFTExample) -> dict[str, list[int]]:
        turns = [{"role": m.role, "content": m.content} for m in example.messages]
        prompt_ids = tokenizer.apply_chat_template(
            turns[:-1], tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(turns, tokenize=True)
        labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
        return {
            "input_ids": list(full_ids[: config.max_seq_len]),
            "labels": labels[: config.max_seq_len],
        }

    train_dataset = Dataset.from_list([encode(ex) for ex in train])
    eval_dataset = Dataset.from_list([encode(ex) for ex in val]) if val else None
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        seed=config.seed,
    )
    trainer = Trainer(
        model=model, args=arguments, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    trainer.train()
    trainer.save_model(str(output_dir))
