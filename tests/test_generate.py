"""TeacherRunner grid assignment, CoT mode and prompt-pool parsing."""

import asyncio

import pytest

from distillery.errors import DataValidationError
from distillery.llm import EchoMockClient
from distillery.teacher.generate import (
    COT_DIRECTIVE,
    PromptSpec,
    TeacherRunner,
    parse_prompt_pool,
)


def _pool(n: int) -> list[PromptSpec]:
    return [
        PromptSpec(id=f"p{i}", capability="summarization", instruction=f"Task {i}", input_text=f"Input {i}. " * 3)
        for i in range(n)
    ]


def test_runner_generates_k_candidates_per_prompt() -> None:
    runner = TeacherRunner(EchoMockClient(), k=2)
    candidates = asyncio.run(runner.run(_pool(3)))
    assert len(candidates) == 6
    counts = dict.fromkeys(("p0", "p1", "p2"), 0)
    for candidate in candidates:
        counts[candidate.prompt_id] += 1
    assert counts == {"p0": 2, "p1": 2, "p2": 2}


def test_runner_is_deterministic() -> None:
    a = asyncio.run(TeacherRunner(EchoMockClient(), k=2).run(_pool(3)))
    b = asyncio.run(TeacherRunner(EchoMockClient(), k=2).run(_pool(3)))
    assert [(c.prompt_id, c.response) for c in a] == [(c.prompt_id, c.response) for c in b]


def test_runner_grid_stride_repeats_after_grid_length() -> None:
    runner = TeacherRunner(EchoMockClient(), k=2)
    candidates = asyncio.run(runner.run(_pool(13)))
    by_prompt = {c.prompt_id: c.config.label for c in candidates}
    # grid = 3 personas x 2 temperatures x 2 seeds = 12 entries
    assert by_prompt["p0"] == by_prompt["p12"]
    assert by_prompt["p0"] != by_prompt["p1"]


def test_runner_cot_mode_keeps_reasoning_in_message() -> None:
    runner = TeacherRunner(EchoMockClient(), k=1, cot=True)
    candidates = asyncio.run(runner.run(_pool(1)))
    system = candidates[0].system_content()
    assert "step by step" in system
    assert COT_DIRECTIVE in system
    assert "Reasoning:" in candidates[0].response
    assert "Answer:" in candidates[0].response


def test_parse_prompt_pool() -> None:
    text = (
        '{"id": "a", "capability": "coding", "instruction": "do x", "input": "ctx"}\n'
        '{"id": "b", "capability": "reasoning", "instruction": "do y"}\n'
    )
    prompts = parse_prompt_pool(text)
    assert [p.id for p in prompts] == ["a", "b"]
    assert prompts[0].render_user() == "do x\n\nctx"
    assert prompts[1].render_user() == "do y"


def test_parse_prompt_pool_rejects_malformed() -> None:
    with pytest.raises(DataValidationError, match="missing keys"):
        parse_prompt_pool('{"id": "a"}')
    with pytest.raises(DataValidationError, match="invalid JSON"):
        parse_prompt_pool("{nope}")
    with pytest.raises(DataValidationError, match="empty"):
        parse_prompt_pool("\n\n")
