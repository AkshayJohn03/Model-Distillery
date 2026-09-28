"""Teacher-side synthetic SFT data generation.

:class:`TeacherRunner` walks a deterministic ``persona x temperature x seed``
grid and generates ``k`` candidate SFT pairs per prompt. CoT mode adds a
reasoning directive to the system message; the reasoning trace stays in the
assistant message ("Reasoning: ... Answer: ...") so the chat format is preserved.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from distillery.errors import DataValidationError
from distillery.llm import ChatMessage, LLMClient

DEFAULT_PERSONAS = (
    "a concise senior practitioner",
    "a meticulous domain analyst",
    "a pragmatic staff engineer",
)
DEFAULT_TEMPERATURES = (0.3, 0.8)
DEFAULT_SEEDS = (11, 22)

COT_DIRECTIVE = (
    "Think step by step: outline brief reasoning first, then give the final "
    "answer after an 'Answer:' heading."
)

_PROMPT_KEYS = {"id", "capability", "instruction"}


@dataclass(frozen=True)
class PromptSpec:
    """One entry of the prompt pool."""

    id: str
    capability: str
    instruction: str
    input_text: str = ""

    def render_user(self) -> str:
        if self.input_text:
            return f"{self.instruction}\n\n{self.input_text}"
        return self.instruction


@dataclass(frozen=True)
class GenerationConfig:
    """One point of the generation grid."""

    persona: str
    temperature: float
    seed: int
    cot: bool = False

    @property
    def label(self) -> str:
        return f"persona={self.persona}|temp={self.temperature}|seed={self.seed}|cot={self.cot}"


@dataclass(frozen=True)
class RawCandidate:
    """A single (prompt, config) -> response candidate pair."""

    prompt_id: str
    capability: str
    messages: tuple[ChatMessage, ...]
    response: str
    config: GenerationConfig

    def user_content(self) -> str:
        return next(m.content for m in self.messages if m.role == "user")

    def system_content(self) -> str:
        return next((m.content for m in self.messages if m.role == "system"), "")


def build_system_message(persona: str, cot: bool) -> ChatMessage:
    parts = [f"You are {persona}."]
    if cot:
        parts.append(COT_DIRECTIVE)
    parts.append("Produce a clear, correct, self-contained answer.")
    return ChatMessage("system", " ".join(parts))


def parse_prompt_pool(text: str) -> list[PromptSpec]:
    """Parse JSONL prompt-pool text into :class:`PromptSpec` objects."""
    prompts: list[PromptSpec] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"prompt pool line {lineno}: invalid JSON ({exc})") from exc
        missing = _PROMPT_KEYS - set(row)
        if missing:
            raise DataValidationError(f"prompt pool line {lineno}: missing keys {sorted(missing)}")
        prompts.append(
            PromptSpec(
                id=str(row["id"]),
                capability=str(row["capability"]),
                instruction=str(row["instruction"]),
                input_text=str(row.get("input", "")),
            )
        )
    if not prompts:
        raise DataValidationError("prompt pool is empty")
    return prompts


def load_prompt_pool(path: Path) -> list[PromptSpec]:
    return parse_prompt_pool(path.read_text(encoding="utf-8"))


class TeacherRunner:
    """Generates ``k`` candidate SFT pairs per prompt.

    The grid is ``personas x temperatures x seeds`` (CoT flag applied globally).
    Prompt ``i`` uses grid entries ``(i + j) % len(grid)`` for ``j in range(k)``:
    consecutive prompts see different configs (variety) while prompts exactly
    ``len(grid)`` apart share configs (which makes byte-identical duplicate
    prompts produce byte-identical candidates -- a realistic scenario the dedup
    stage is designed to catch).
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        k: int = 2,
        personas: Sequence[str] = DEFAULT_PERSONAS,
        temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
        seeds: Sequence[int] = DEFAULT_SEEDS,
        cot: bool = False,
        concurrency: int = 8,
        max_tokens: int = 512,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self._client = client
        self._k = k
        self._cot = cot
        self._max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._grid = [
            GenerationConfig(persona=p, temperature=t, seed=s, cot=cot)
            for p, t, s in itertools.product(personas, temperatures, seeds)
        ]
        if not self._grid:
            raise ValueError("generation grid is empty")

    @property
    def grid_size(self) -> int:
        return len(self._grid)

    async def run(self, prompts: Sequence[PromptSpec]) -> list[RawCandidate]:
        """Generate candidates for all prompts; output order is deterministic."""
        tasks = []
        for idx, prompt in enumerate(prompts):
            for j in range(self._k):
                config = self._grid[(idx + j) % len(self._grid)]
                tasks.append(self._generate(prompt, config))
        return list(await asyncio.gather(*tasks))

    async def _generate(self, prompt: PromptSpec, config: GenerationConfig) -> RawCandidate:
        messages = [build_system_message(config.persona, config.cot), ChatMessage("user", prompt.render_user())]
        async with self._semaphore:
            completion = await self._client.complete(
                messages,
                temperature=config.temperature,
                max_tokens=self._max_tokens,
                seed=config.seed,
            )
        return RawCandidate(
            prompt_id=prompt.id,
            capability=prompt.capability,
            messages=tuple(messages),
            response=completion.text,
            config=config,
        )
