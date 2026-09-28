"""Student-vs-teacher evaluation on a golden eval set.

Metrics per case (computed against a reference answer):

* ``exact_match`` -- normalized string equality (strict ceiling metric).
* ``token_f1`` -- bag-of-words F1 between prediction and reference (primary
  quality signal used for the cost/retention report).
* ``containment`` -- fraction of reference tokens present in the prediction
  (recall-oriented: did the student keep the key content?).
* optional judge score via the VerdictAI-compatible :class:`JudgeHook` protocol.

Results are broken down by capability slice so regressions on reasoning or
safety are visible instead of hidden in the average. All metric functions are
pure string math -- no numpy/torch -- so the bench runs anywhere.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from distillery.errors import DataValidationError
from distillery.teacher.filter import HeuristicJudge, JudgeHook, JudgeItem

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if " ".join(_tokens(prediction)) == " ".join(_tokens(reference)) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = Counter(_tokens(prediction)), Counter(_tokens(reference))
    if not pred or not ref:
        return 0.0
    common = sum((pred & ref).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def containment(prediction: str, reference: str) -> float:
    ref = set(_tokens(reference))
    if not ref:
        return 0.0
    return len(ref & set(_tokens(prediction))) / len(ref)


@dataclass(frozen=True)
class EvalCase:
    id: str
    capability: str
    instruction: str
    input_text: str = ""
    reference: str = ""

    def render_user(self) -> str:
        if self.input_text:
            return f"{self.instruction}\n\n{self.input_text}"
        return self.instruction

    def corpus_text(self) -> str:
        """Full text used when indexing this case for decontamination."""
        return f"{self.instruction}\n{self.input_text}\n{self.reference}"


def parse_eval_set(text: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"eval set line {lineno}: invalid JSON ({exc})") from exc
        missing = {"id", "capability", "instruction", "reference"} - set(row)
        if missing:
            raise DataValidationError(f"eval set line {lineno}: missing keys {sorted(missing)}")
        cases.append(
            EvalCase(
                id=str(row["id"]),
                capability=str(row["capability"]),
                instruction=str(row["instruction"]),
                input_text=str(row.get("input", "")),
                reference=str(row["reference"]),
            )
        )
    if not cases:
        raise DataValidationError("eval set is empty")
    return cases


def load_eval_set(path: Path) -> list[EvalCase]:
    return parse_eval_set(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CaseMetrics:
    exact_match: float
    token_f1: float
    containment: float
    judge_score: float | None = None


@dataclass
class BenchReport:
    models: list[str]
    rows: dict[str, dict[str, CaseMetrics]]  # model -> case_id -> metrics
    capabilities: dict[str, str]  # case_id -> capability
    metric_names: tuple[str, ...] = ("exact_match", "token_f1", "containment")

    def _mean(self, model: str, metric: str, case_ids: Sequence[str]) -> float:
        values = [getattr(self.rows[model][cid], metric) for cid in case_ids]
        return sum(values) / len(values) if values else 0.0

    def overall(self, model: str) -> dict[str, float]:
        return {m: self._mean(model, m, list(self.capabilities)) for m in self.metric_names}

    def by_capability(self, model: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for capability in sorted(set(self.capabilities.values())):
            case_ids = [c for c, cap in self.capabilities.items() if cap == capability]
            out[capability] = {m: self._mean(model, m, case_ids) for m in self.metric_names}
        return out

    def judge_overall(self, model: str) -> float | None:
        scores = [
            metrics.judge_score
            for metrics in self.rows[model].values()
            if metrics.judge_score is not None
        ]
        return sum(scores) / len(scores) if scores else None

    def to_markdown(self) -> str:
        lines = ["# Student vs Teacher Bench", "", "## Overall", ""]
        judge_available = any(self.judge_overall(model) is not None for model in self.models)
        header = "| Model | " + " | ".join(self.metric_names) + (" | judge |" if judge_available else " |")
        lines += [header, "| --- " * (len(self.metric_names) + 1) + ("| --- |" if judge_available else "|")]
        for model in self.models:
            overall = self.overall(model)
            cells = " | ".join(f"{overall[m]:.3f}" for m in self.metric_names)
            row = f"| {model} | {cells} |"
            judge = self.judge_overall(model)
            row += f" {judge:.3f} |" if judge is not None else (" -- |" if judge_available else "|")
            lines.append(row)
        lines += ["", "## Capability Slices (token_f1)", "", "| Capability | " + " | ".join(self.models) + " |", "| --- " * (len(self.models) + 1) + "|"]
        slices = {model: self.by_capability(model) for model in self.models}
        for capability in sorted({c for cap in slices.values() for c in cap}):
            cells = " | ".join(f"{slices[m][capability]['token_f1']:.3f}" for m in self.models)
            lines.append(f"| {capability} | {cells} |")
        lines.append("")
        return "\n".join(lines)


class StudentBench:
    """Runs metric computation over ``{model_name: {case_id: prediction}}``."""

    def __init__(self, cases: Sequence[EvalCase], *, judge: JudgeHook | None = None) -> None:
        self._cases = list(cases)
        self._judge: JudgeHook | None = judge or HeuristicJudge()
        if not self._cases:
            raise ValueError("eval case list is empty")

    @property
    def judge_enabled(self) -> bool:
        return self._judge is not None

    def run(self, predictions: Mapping[str, Mapping[str, str]]) -> BenchReport:
        capabilities = {case.id: case.capability for case in self._cases}
        rows: dict[str, dict[str, CaseMetrics]] = {}
        for model in sorted(predictions):
            model_preds = predictions[model]
            case_rows: dict[str, CaseMetrics] = {}
            for case in self._cases:
                pred = model_preds.get(case.id, "")
                judge_score: float | None = None
                if self._judge is not None:
                    judge_score = float(
                        self._judge.score(
                            JudgeItem(prompt=case.render_user(), response=pred, capability=case.capability)
                        )
                    )
                case_rows[case.id] = CaseMetrics(
                    exact_match=exact_match(pred, case.reference),
                    token_f1=token_f1(pred, case.reference),
                    containment=containment(pred, case.reference),
                    judge_score=judge_score,
                )
            rows[model] = case_rows
        return BenchReport(models=sorted(predictions), rows=rows, capabilities=capabilities)
