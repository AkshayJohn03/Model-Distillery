"""SFT schema validation, JSONL IO and stratified splitting."""

import pytest
from pydantic import ValidationError

from distillery.data.sft import (
    Message,
    SFTExample,
    dump_jsonl,
    example_from_candidate,
    load_jsonl,
    stratified_split,
)


def make(conversation: list[tuple[str, str]], **metadata) -> SFTExample:
    return SFTExample(
        messages=[Message(role=role, content=content) for role, content in conversation],
        metadata={"capability": "summarization", **metadata},
    )


VALID = [("user", "Question one?"), ("assistant", "Answer one.")]


def test_valid_conversations() -> None:
    assert make(VALID).messages[-1].role == "assistant"
    assert make([("system", "sys"), *VALID]).messages[0].role == "system"


@pytest.mark.parametrize(
    "conversation",
    [
        [("assistant", "starts with assistant"), ("user", "then user")],
        [("user", "q1"), ("user", "q2")],
        [("user", "q1"), ("assistant", "a1"), ("assistant", "a2")],
        [("system", "sys")],
        [("assistant", "only assistant")],
        [("user", "q1"), ("assistant", "a1"), ("user", "q2")],
    ],
)
def test_malformed_conversations_rejected(conversation) -> None:
    with pytest.raises(ValidationError):
        make(conversation)


def test_empty_content_rejected() -> None:
    with pytest.raises(ValidationError):
        make([("user", ""), ("assistant", "answer")])


def test_unknown_role_rejected() -> None:
    with pytest.raises(ValidationError):
        make([("tool", "x"), ("assistant", "y")])


def test_jsonl_roundtrip(tmp_path) -> None:
    examples = [make(VALID, capability="coding", quality_score=0.9)]
    path = tmp_path / "sft.jsonl"
    dump_jsonl(examples, path)
    loaded = load_jsonl(path)
    assert loaded == examples


def test_load_jsonl_error_has_line_context(tmp_path) -> None:
    from distillery.errors import DataValidationError

    path = tmp_path / "bad.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "only one"}]}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match=r"bad.jsonl line 1"):
        load_jsonl(path)


def test_example_from_candidate(make_candidate) -> None:
    candidate = make_candidate(prompt_id="p9", response="The answer.", seed=7)
    example = example_from_candidate(candidate, quality_score=0.75)
    roles = [m.role for m in example.messages]
    assert roles == ["system", "user", "assistant"]
    assert example.messages[-1].content == "The answer."
    assert example.metadata.prompt_id == "p9"
    assert example.metadata.quality_score == 0.75
    assert example.metadata.seed == 7
    assert "tester" in example.metadata.source_config


def test_stratified_split_preserves_capability_ratios(make_example) -> None:
    examples = (
        [make_example("summarization", i) for i in range(30)]
        + [make_example("extraction", i) for i in range(20)]
        + [make_example("coding", i) for i in range(10)]
    )
    train, val = stratified_split(examples, val_fraction=0.2, seed=42)
    assert len(train) == 48 and len(val) == 12
    val_counts = {}
    train_counts = {}
    for example in val:
        val_counts[example.capability] = val_counts.get(example.capability, 0) + 1
    for example in train:
        train_counts[example.capability] = train_counts.get(example.capability, 0) + 1
    assert val_counts == {"summarization": 6, "extraction": 4, "coding": 2}
    assert train_counts == {"summarization": 24, "extraction": 16, "coding": 8}


def test_stratified_split_is_deterministic_and_disjoint(make_example) -> None:
    examples = [make_example("summarization", i) for i in range(20)]
    train_a, val_a = stratified_split(examples, val_fraction=0.2, seed=1)
    train_b, val_b = stratified_split(examples, val_fraction=0.2, seed=1)
    assert (train_a, val_a) == (train_b, val_b)
    ids_a = {e.metadata.prompt_id for e in train_a}
    ids_b = {e.metadata.prompt_id for e in val_a}
    assert not (ids_a & ids_b)


def test_stratified_split_validates_fraction(make_example) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        stratified_split([make_example()], val_fraction=1.0)
