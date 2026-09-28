"""Deterministic markdown dataset balance reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from distillery.data.sft import SFTExample

_LENGTH_BINS: tuple[tuple[int, int, str], ...] = (
    (0, 200, "<200"),
    (200, 400, "200-399"),
    (400, 800, "400-799"),
    (800, 1600, "800-1599"),
    (1600, 10**9, "1600+"),
)
_SCORE_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.4, "<0.40"),
    (0.4, 0.6, "0.40-0.59"),
    (0.6, 0.8, "0.60-0.79"),
    (0.8, 1.01, "0.80-1.00"),
)


def _bar(count: int, max_count: int, width: int = 30) -> str:
    if max_count <= 0 or count <= 0:
        return ""
    return "#" * max(1, round(width * count / max_count))


def build_dataset_report(
    examples: Sequence[SFTExample],
    *,
    candidates_generated: int,
    train_size: int,
    val_size: int,
    dedup_removed: int,
    decontam_removed: int,
    discard_reasons: Mapping[str, int],
) -> str:
    """Build the markdown dataset balance report (deterministic output)."""
    lengths = sorted(len(ex.response_text()) for ex in examples)
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    length_counts = [sum(1 for n in lengths if lo <= n < hi) for lo, hi, _ in _LENGTH_BINS]
    max_len_count = max(length_counts, default=0)

    capability_counts = Counter(ex.capability for ex in examples)
    total = len(examples)
    score_counts = Counter(
        next(label for lo, hi, label in _SCORE_BINS if lo <= ex.metadata.quality_score < hi)
        for ex in examples
    )

    lines: list[str] = [
        "# Dataset Report",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| candidates generated | {candidates_generated} |",
        f"| examples after filtering | {total} |",
        f"| dedup removed | {dedup_removed} |",
        f"| decontamination removed | {decontam_removed} |",
        f"| train size | {train_size} |",
        f"| val size | {val_size} |",
        f"| mean response length (chars) | {mean_len:.1f} |",
        "",
        "## Capability Balance",
        "",
        "| Capability | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    for capability in sorted(capability_counts):
        count = capability_counts[capability]
        share = 100.0 * count / total if total else 0.0
        lines.append(f"| {capability} | {count} | {share:.1f}% |")

    lines += [
        "",
        "## Response Length Histogram",
        "",
        "| Chars | Count | |",
        "| --- | ---: | --- |",
    ]
    for (_lo, _hi, label), count in zip(_LENGTH_BINS, length_counts, strict=True):
        lines.append(f"| {label} | {count} | {_bar(count, max_len_count)} |")

    lines += [
        "",
        "## Quality Score Distribution",
        "",
        "| Score bucket | Count | |",
        "| --- | ---: | --- |",
    ]
    max_score = max(score_counts.values(), default=0)
    for _lo, _hi, label in _SCORE_BINS:
        count = score_counts.get(label, 0)
        lines.append(f"| {label} | {count} | {_bar(count, max_score)} |")

    lines += [
        "",
        "## Discard Summary",
        "",
        "| Stage / reason | Count |",
        "| --- | ---: |",
        f"| dedup near-duplicates | {dedup_removed} |",
        f"| decontamination | {decontam_removed} |",
    ]
    for reason in sorted(discard_reasons):
        lines.append(f"| {reason} | {discard_reasons[reason]} |")
    lines.append("")
    return "\n".join(lines)
