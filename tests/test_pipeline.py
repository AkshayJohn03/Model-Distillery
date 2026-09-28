"""End-to-end offline pipeline: snapshot of key stats, flow conservation,
artifact production, determinism, and the CLI entrypoint."""

import asyncio
import json

import pytest

from distillery.cli import main
from distillery.data.sft import load_jsonl
from distillery.pipeline import PipelineSettings, run_pipeline

# Snapshot: exact values for the bundled demo run (deterministic by design).
SNAPSHOT = {
    "prompts": 61,
    "candidates": 122,
    "dedup_removed": 13,
    "rule_discards": {"rule:repetition": 8},
    "rejection_discards": 42,
    "no_valid_prompts": 0,
    "filtered_kept": 59,
    "decontam_removed": 1,
    "train_size": 46,
    "val_size": 12,
    "plan_steps": 9,
    "est_tokens": 22386,
}


def _settings(tmp_path) -> PipelineSettings:
    return PipelineSettings(output_dir=tmp_path / "run", offline=True)


def test_offline_pipeline_snapshot(tmp_path) -> None:
    summary = asyncio.run(run_pipeline(_settings(tmp_path))).to_dict()
    for key, expected in SNAPSHOT.items():
        assert summary[key] == expected, f"{key}: {summary[key]} != {expected}"
    assert summary["bench_retention_pct"] == pytest.approx(68.61, abs=0.5)
    assert summary["bench_metrics"]["teacher"]["token_f1"] > summary["bench_metrics"]["student"]["token_f1"]


def test_offline_pipeline_flow_conservation(tmp_path) -> None:
    summary = asyncio.run(run_pipeline(_settings(tmp_path))).to_dict()
    discarded = summary["dedup_removed"] + sum(summary["rule_discards"].values()) + summary["rejection_discards"]
    assert summary["candidates"] - discarded == summary["filtered_kept"]
    assert summary["filtered_kept"] - summary["decontam_removed"] == summary["train_size"] + summary["val_size"]


def test_offline_pipeline_artifacts_and_reports(tmp_path) -> None:
    out = tmp_path / "run"
    asyncio.run(run_pipeline(PipelineSettings(output_dir=out, offline=True)))
    for name in (
        "01_candidates.jsonl",
        "02_discards.jsonl",
        "03_sft_train.jsonl",
        "04_sft_val.jsonl",
        "dataset_report.md",
        "training_plan.md",
        "bench_report.md",
        "cost_report.md",
        "summary.json",
    ):
        assert (out / name).exists(), name
    train = load_jsonl(out / "03_sft_train.jsonl")
    assert train, "train split must not be empty"
    capabilities = {ex.capability for ex in train}
    assert {
        "summarization",
        "extraction",
        "classification",
        "reasoning",
        "coding",
        "safety-refusal",
    } <= capabilities
    report = (out / "dataset_report.md").read_text(encoding="utf-8")
    assert "# Dataset Report" in report
    plan = (out / "training_plan.md").read_text(encoding="utf-8")
    assert "QLoRA Training Plan" in plan
    cost = (out / "cost_report.md").read_text(encoding="utf-8")
    assert "Breakeven horizon" in cost
    bench = (out / "bench_report.md").read_text(encoding="utf-8")
    assert "safety-refusal" in bench  # capability slice visible, not averaged away


def test_offline_pipeline_is_deterministic(tmp_path) -> None:
    first = asyncio.run(run_pipeline(_settings(tmp_path))).to_dict()
    second = asyncio.run(run_pipeline(_settings(tmp_path / "second"))).to_dict()
    first.pop("artifacts")
    second.pop("artifacts")
    assert first == second


def test_cli_pipeline_and_plan(tmp_path, monkeypatch) -> None:
    out = tmp_path / "cli-run"
    monkeypatch.chdir(tmp_path)
    exit_code = main(["pipeline", "--offline", "--output-dir", str(out)])
    assert exit_code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"] == SNAPSHOT["candidates"]
    exit_code = main(["plan", "--input", str(out / "03_sft_train.jsonl"), "--output", str(out / "plan.md")])
    assert exit_code == 0
    assert "QLoRA Training Plan" in (out / "plan.md").read_text(encoding="utf-8")


def test_bundled_data_is_well_formed() -> None:
    from distillery.eval.bench import parse_eval_set
    from distillery.pipeline import bundled_text
    from distillery.teacher.generate import parse_prompt_pool

    prompts = parse_prompt_pool(bundled_text("demo_prompt_pool.jsonl"))
    cases = parse_eval_set(bundled_text("demo_eval_set.jsonl"))
    assert len(prompts) == 61
    assert len(cases) == 12
    assert {p.capability for p in prompts} == {
        "summarization",
        "extraction",
        "classification",
        "reasoning",
        "coding",
        "safety-refusal",
    }
