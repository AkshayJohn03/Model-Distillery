"""Dataset report: sections present and output deterministic."""

from distillery.data.report import build_dataset_report


def _build(make_example) -> str:
    examples = (
        [make_example("summarization", i, quality=0.9) for i in range(6)]
        + [make_example("coding", i, quality=0.5) for i in range(4)]
    )
    return build_dataset_report(
        examples,
        candidates_generated=20,
        train_size=8,
        val_size=2,
        dedup_removed=3,
        decontam_removed=1,
        discard_reasons={"rule:length": 2, "rejection_sampling": 4},
    )


def test_report_contains_all_sections(make_example) -> None:
    report = _build(make_example)
    for heading in (
        "# Dataset Report",
        "## Overview",
        "## Capability Balance",
        "## Response Length Histogram",
        "## Quality Score Distribution",
        "## Discard Summary",
    ):
        assert heading in report
    assert "| summarization | 6 | 60.0% |" in report
    assert "| coding | 4 | 40.0% |" in report
    assert "| rejection_sampling | 4 |" in report
    assert "| rule:length | 2 |" in report
    assert "| dedup near-duplicates | 3 |" in report
    assert "| decontamination | 1 |" in report


def test_report_is_deterministic(make_example) -> None:
    assert _build(make_example) == _build(make_example)
