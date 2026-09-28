"""SFT chat JSONL schema, validation and stratified splitting.

An example is an OpenAI-style chat transcript (optional system message, at least
one user/assistant exchange, ending with an assistant message) plus provenance
metadata: the generation config it came from, its quality score and capability
tags used for stratification and slice reporting.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from distillery.errors import DataValidationError
from distillery.teacher.generate import RawCandidate

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class SFTMetadata(BaseModel):
    """Provenance attached to every training example."""

    source_config: str = ""
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    capability: str = "general"
    prompt_id: str = ""
    persona: str = ""
    temperature: float = 0.0
    seed: int = 0
    cot: bool = False


class SFTExample(BaseModel):
    """One validated SFT chat example."""

    messages: list[Message] = Field(min_length=2)
    metadata: SFTMetadata = Field(default_factory=SFTMetadata)

    @model_validator(mode="after")
    def _validate_conversation(self) -> SFTExample:
        roles = [m.role for m in self.messages]
        if roles[0] not in ("system", "user"):
            raise ValueError("conversation must start with a system or user message")
        if roles[-1] != "assistant":
            raise ValueError("conversation must end with an assistant message")
        expected_user = True
        for role in roles[1:] if roles[0] == "system" else roles:
            if expected_user and role != "user":
                raise ValueError("expected a user message in alternating user/assistant order")
            if not expected_user and role != "assistant":
                raise ValueError("expected an assistant message in alternating user/assistant order")
            expected_user = not expected_user
        return self

    @property
    def capability(self) -> str:
        return self.metadata.capability

    def prompt_text(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role != "assistant")

    def response_text(self) -> str:
        return self.messages[-1].content

    def full_text(self) -> str:
        return "\n".join(m.content for m in self.messages)


def example_from_candidate(candidate: RawCandidate, *, quality_score: float) -> SFTExample:
    """Convert a filtered candidate into an SFT example (chat format preserved)."""
    messages = [
        Message(role=m.role, content=m.content)
        for m in candidate.messages
        if m.role in ("system", "user")
    ]
    messages.append(Message(role="assistant", content=candidate.response))
    metadata = SFTMetadata(
        source_config=candidate.config.label,
        quality_score=round(quality_score, 4),
        capability=candidate.capability,
        prompt_id=candidate.prompt_id,
        persona=candidate.config.persona,
        temperature=candidate.config.temperature,
        seed=candidate.config.seed,
        cot=candidate.config.cot,
    )
    return SFTExample(messages=messages, metadata=metadata)


def dump_jsonl(examples: Sequence[SFTExample], path: Path) -> None:
    path.write_text(
        "\n".join(ex.model_dump_json() for ex in examples) + ("\n" if examples else ""),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[SFTExample]:
    examples: list[SFTExample] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(SFTExample.model_validate_json(line))
        except ValueError as exc:
            raise DataValidationError(f"{path.name} line {lineno}: {exc}") from exc
    return examples


def stratified_split(
    examples: Sequence[SFTExample], *, val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Train/val split stratified by capability.

    Within each capability group the split is drawn with a per-capability RNG
    seeded by ``f"{seed}:{capability}"``, so the result is independent of input
    ordering and stable across runs. Ratios are preserved up to rounding.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    groups: dict[str, list[SFTExample]] = defaultdict(list)
    for example in examples:
        groups[example.capability].append(example)

    train: list[SFTExample] = []
    val: list[SFTExample] = []
    for capability in sorted(groups):
        rows = groups[capability]
        rng = random.Random(f"{seed}:{capability}")
        order = list(range(len(rows)))
        rng.shuffle(order)
        n_val = round(len(rows) * val_fraction)
        val_positions = set(order[:n_val])
        for i, example in enumerate(rows):
            (val if i in val_positions else train).append(example)
    return train, val


def dump_examples_json(examples: Sequence[SFTExample]) -> str:
    """Convenience for tests/logging: JSON array of example dicts."""
    return json.dumps([ex.model_dump() for ex in examples], indent=2)
