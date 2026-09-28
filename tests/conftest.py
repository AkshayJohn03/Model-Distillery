"""Shared builders for the offline test suite."""

import pytest

from distillery.data.sft import Message, SFTExample, SFTMetadata
from distillery.llm import ChatMessage
from distillery.teacher.generate import GenerationConfig, RawCandidate

PROMPT_TEXT = "Summarize the quarterly revenue report for the board meeting."


def _make_candidate(
    prompt_id: str = "p1",
    response: str = "A perfectly adequate response with several words in it.",
    capability: str = "summarization",
    seed: int = 1,
    persona: str = "tester",
) -> RawCandidate:
    return RawCandidate(
        prompt_id=prompt_id,
        capability=capability,
        messages=(
            ChatMessage("system", f"You are {persona}."),
            ChatMessage("user", PROMPT_TEXT),
        ),
        response=response,
        config=GenerationConfig(persona=persona, temperature=0.5, seed=seed),
    )


def _make_example(
    capability: str = "summarization",
    index: int = 0,
    quality: float = 0.8,
    response_chars: int = 120,
) -> SFTExample:
    return SFTExample(
        messages=[
            Message(role="user", content=f"Question number {index} about {capability}?"),
            Message(
                role="assistant",
                content=(f"Answer {index} for {capability}. " + "Padding text here. " * 8)
               [:response_chars],
            ),
        ],
        metadata=SFTMetadata(capability=capability, quality_score=quality, prompt_id=f"{capability}-{index}"),
    )


@pytest.fixture
def make_candidate():
    return _make_candidate


@pytest.fixture
def make_example():
    return _make_example
