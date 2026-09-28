"""Bench metrics: hand-computed values, capability slices, judge column."""

import pytest

from distillery.errors import DataValidationError
from distillery.eval.bench import (
    EvalCase,
    StudentBench,
    containment,
    exact_match,
    parse_eval_set,
    token_f1,
)


def test_exact_match_normalizes() -> None:
    assert exact_match("The Cat  sat!", "the cat sat") == 1.0
    assert exact_match("the cat ran", "the cat sat") == 0.0


def test_token_f1_hand_computed() -> None:
    # pred = [a, b, c, d], ref = [a, b, e]: precision 2/4, recall 2/3 -> F1 = 4/7
    assert token_f1("a b c d", "a b e") == pytest.approx(4 / 7, abs=1e-12)
    assert token_f1("", "a b") == 0.0
    assert token_f1("a b", "") == 0.0


def test_containment_is_reference_recall() -> None:
    assert containment("a x b", "a b c") == pytest.approx(2 / 3)
    assert containment("x y", "a b") == 0.0
    assert containment("anything", "") == 0.0


CASES = [
    EvalCase(id="c1", capability="summarization", instruction="Summarize.", reference="alpha beta"),
    EvalCase(id="c2", capability="coding", instruction="Code.", reference="gamma delta"),
]


def test_student_bench_slices_and_judge() -> None:
    predictions = {
        "student": {"c1": "alpha beta", "c2": "gamma"},
        "teacher": {"c1": "alpha beta", "c2": "gamma delta"},
    }
    report = StudentBench(CASES).run(predictions)
    assert report.overall("teacher")["exact_match"] == pytest.approx(1.0)
    assert report.overall("student")["exact_match"] == pytest.approx(0.5)
    assert report.overall("student")["token_f1"] == pytest.approx(
        (1.0 + 2 / 3) / 2, abs=1e-12
    )
    slices = report.by_capability("teacher")
    assert set(slices) == {"coding", "summarization"}
    assert slices["summarization"]["exact_match"] == 1.0
    markdown = report.to_markdown()
    assert "## Capability Slices (token_f1)" in markdown
    assert "| coding |" in markdown
    assert "| teacher |" in markdown


def test_missing_prediction_counts_as_empty() -> None:
    report = StudentBench(CASES).run({"student": {"c1": "alpha beta"}})
    assert report.overall("student")["containment"] == pytest.approx(0.5)


def test_parse_eval_set_rejects_malformed() -> None:
    with pytest.raises(DataValidationError, match="missing keys"):
        parse_eval_set('{"id": "x"}')
    with pytest.raises(DataValidationError, match="invalid JSON"):
        parse_eval_set("{oops}")
    with pytest.raises(DataValidationError, match="empty"):
        parse_eval_set("")


def test_empty_case_list_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        StudentBench([])
