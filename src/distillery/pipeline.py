"""End-to-end pipeline orchestration.

    generate -> dedup -> filter/rejection -> decontam -> split -> report -> plan -> bench -> cost

``python -m distillery pipeline --offline`` runs the full non-torch path on the
bundled demo data: a deterministic EchoMock teacher (fidelity 1.0) and a weaker
EchoMock student sim (fidelity 0.7) stand in for the real models, so every stage
-- including the bench quality delta and the cost report -- is exercised
reproducibly without GPU or network. With real endpoints configured, the same
orchestration runs against any OpenAI-compatible teacher/student.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from distillery.config import DistillerySettings
from distillery.data.decontam import NgramDecontaminator
from distillery.data.report import build_dataset_report
from distillery.data.sft import dump_jsonl, example_from_candidate, stratified_split
from distillery.eval.bench import EvalCase, StudentBench, parse_eval_set
from distillery.eval.cost import ModelPrice, UsageProfile, build_cost_report
from distillery.llm import ChatMessage, EchoMockClient, LLMClient, OpenAICompatClient
from distillery.teacher.dedup import DedupItem, MinHashDeduplicator
from distillery.teacher.filter import QualityFilter
from distillery.teacher.generate import TeacherRunner, parse_prompt_pool
from distillery.train.sft import TrainingConfig, build_training_plan

TEACHER_SIM_FIDELITY = 1.0
STUDENT_SIM_FIDELITY = 0.7

# Demo pricing (USD per 1M tokens): frontier-class teacher API vs the distilled
# 4-bit student on a small GPU. The shape of the argument, not exact quotes.
DEMO_TEACHER_PRICE = ModelPrice("teacher-llm (frontier API)", 3.00, 15.00)
DEMO_STUDENT_PRICE = ModelPrice("distilled-student (4-bit GPU)", 0.15, 0.60)
DEMO_USAGE = UsageProfile(avg_input_tokens=900, avg_output_tokens=250, monthly_requests=100_000)


@dataclass(frozen=True)
class PipelineSettings:
    output_dir: Path
    offline: bool = True
    k: int = 2
    seed: int = 42
    limit: int | None = None
    prompts_path: Path | None = None
    eval_path: Path | None = None
    investment_usd: float = 60.0


@dataclass(frozen=True)
class PipelineSummary:
    prompts: int
    candidates: int
    dedup_removed: int
    rule_discards: dict[str, int]
    rejection_discards: int
    no_valid_prompts: int
    filtered_kept: int
    decontam_removed: int
    train_size: int
    val_size: int
    plan_steps: int
    est_tokens: int
    bench_retention_pct: float
    bench_metrics: dict[str, dict[str, float]]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bundled_text(name: str) -> str:
    """Read a bundled data file from the installed package."""
    return (resources.files("distillery") / "data" / name).read_text(encoding="utf-8")


def _teacher_client(settings: PipelineSettings) -> LLMClient:
    if settings.offline:
        return EchoMockClient(fidelity=TEACHER_SIM_FIDELITY, model="echo-teacher-sim")
    config = DistillerySettings()
    return OpenAICompatClient(
        config.teacher_base_url,
        config.teacher_model,
        api_key=config.teacher_api_key,
        timeout=config.request_timeout,
    )


def _student_client(settings: PipelineSettings) -> LLMClient:
    if settings.offline:
        return EchoMockClient(fidelity=STUDENT_SIM_FIDELITY, model="echo-student-sim")
    config = DistillerySettings()
    base_url, api_key, model = config.student_endpoint()
    return OpenAICompatClient(base_url, model, api_key=api_key, timeout=config.request_timeout)


async def _predictions(
    client: LLMClient, cases: list[EvalCase], *, temperature: float, seed: int
) -> dict[str, str]:
    async def one(case: EvalCase) -> tuple[str, str]:
        messages = [ChatMessage("user", case.render_user())]
        completion = await client.complete(messages, temperature=temperature, max_tokens=512, seed=seed)
        return case.id, completion.text

    return dict(await asyncio.gather(*(one(case) for case in cases)))


def _write_candidates(path: Path, candidates: list) -> None:
    rows = [
        {
            "prompt_id": c.prompt_id,
            "capability": c.capability,
            "persona": c.config.persona,
            "temperature": c.config.temperature,
            "seed": c.config.seed,
            "cot": c.config.cot,
            "user": c.user_content(),
            "response": c.response,
        }
        for c in candidates
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


async def run_pipeline(settings: PipelineSettings) -> PipelineSummary:
    """Execute the full distillation pipeline and write artifacts to ``output_dir``."""
    out = settings.output_dir
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    prompt_text = (
        settings.prompts_path.read_text(encoding="utf-8")
        if settings.prompts_path
        else bundled_text("demo_prompt_pool.jsonl")
    )
    prompts = parse_prompt_pool(prompt_text)
    if settings.limit:
        prompts = prompts[: settings.limit]

    eval_text = (
        settings.eval_path.read_text(encoding="utf-8")
        if settings.eval_path
        else bundled_text("demo_eval_set.jsonl")
    )
    eval_cases = parse_eval_set(eval_text)

    # 1. Generate candidates (persona x temperature x seed grid, k per prompt).
    teacher_client = _teacher_client(settings)
    runner = TeacherRunner(teacher_client, k=settings.k)
    candidates = await runner.run(prompts)
    _write_candidates(out / "01_candidates.jsonl", candidates)
    artifacts["candidates"] = str(out / "01_candidates.jsonl")

    # 2. MinHash-LSH near-duplicate removal on candidate responses.
    dedup_result = MinHashDeduplicator().deduplicate(
        [DedupItem(id=str(i), text=c.response) for i, c in enumerate(candidates)]
    )
    kept_candidates = [candidates[int(item.id)] for item in dedup_result.kept]

    # 3. Rule gate + judge + rejection sampling (best-of-k per prompt).
    filter_result = QualityFilter().run(kept_candidates)
    examples = [
        example_from_candidate(sc.candidate, quality_score=sc.score) for sc in filter_result.kept
    ]
    discards_path = out / "02_discards.jsonl"
    discards_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "stage": d.stage,
                    "reason": d.reason,
                    "prompt_id": d.candidate.prompt_id,
                    "response": d.candidate.response,
                },
                ensure_ascii=False,
            )
            for d in filter_result.discards
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts["discards"] = str(discards_path)

    # 4. Decontaminate against the eval corpus (13-gram verbatim overlap).
    decontaminator = NgramDecontaminator(n=13)
    decontaminator.index_eval_corpus({case.id: case.corpus_text() for case in eval_cases})
    screening = decontaminator.screen({str(i): ex.full_text() for i, ex in enumerate(examples)})
    contaminated = set(screening.contaminated_ids)
    clean_examples = [ex for i, ex in enumerate(examples) if str(i) not in contaminated]

    # 5. Stratified split.
    train, val = stratified_split(clean_examples, val_fraction=0.2, seed=settings.seed)
    dump_jsonl(train, out / "03_sft_train.jsonl")
    dump_jsonl(val, out / "04_sft_val.jsonl")
    artifacts["sft_train"] = str(out / "03_sft_train.jsonl")
    artifacts["sft_val"] = str(out / "04_sft_val.jsonl")

    # 6. Dataset balance report.
    histogram = filter_result.histogram
    rule_discards = {k: v for k, v in histogram.items() if k.startswith("rule:")}
    report_md = build_dataset_report(
        clean_examples,
        candidates_generated=len(candidates),
        train_size=len(train),
        val_size=len(val),
        dedup_removed=len(candidates) - len(dedup_result.kept),
        decontam_removed=len(contaminated),
        discard_reasons=histogram,
    )
    report_path = out / "dataset_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    artifacts["dataset_report"] = str(report_path)

    # 7. Training plan (deterministic; torch-free).
    plan = build_training_plan(train, val, config=TrainingConfig(seed=settings.seed))
    plan_path = out / "training_plan.md"
    plan_path.write_text(plan.to_markdown(), encoding="utf-8")
    artifacts["training_plan"] = str(plan_path)

    # 8. Teacher-vs-student bench on the golden eval set.
    teacher_preds = await _predictions(teacher_client, eval_cases, temperature=0.3, seed=7)
    student_client = _student_client(settings)
    student_preds = await _predictions(student_client, eval_cases, temperature=0.8, seed=8)
    bench_report = StudentBench(eval_cases).run(
        {"teacher": teacher_preds, "student": student_preds}
    )
    bench_path = out / "bench_report.md"
    bench_path.write_text(bench_report.to_markdown(), encoding="utf-8")
    artifacts["bench_report"] = str(bench_path)

    teacher_f1 = bench_report.overall("teacher")["token_f1"]
    student_f1 = bench_report.overall("student")["token_f1"]
    retention = 100.0 * student_f1 / teacher_f1 if teacher_f1 else 0.0

    # 9. Cost comparison with breakeven analysis.
    cost_md = build_cost_report(
        DEMO_TEACHER_PRICE,
        DEMO_STUDENT_PRICE,
        DEMO_USAGE,
        quality_retention_pct=retention,
        investment_usd=settings.investment_usd,
    )
    cost_path = out / "cost_report.md"
    cost_path.write_text(cost_md, encoding="utf-8")
    artifacts["cost_report"] = str(cost_path)

    summary = PipelineSummary(
        prompts=len(prompts),
        candidates=len(candidates),
        dedup_removed=len(candidates) - len(dedup_result.kept),
        rule_discards=rule_discards,
        rejection_discards=histogram.get("rejection_sampling", 0),
        no_valid_prompts=histogram.get("no_valid_candidate", 0),
        filtered_kept=len(examples),
        decontam_removed=len(contaminated),
        train_size=len(train),
        val_size=len(val),
        plan_steps=plan.steps,
        est_tokens=plan.est_tokens,
        bench_retention_pct=round(retention, 2),
        bench_metrics={
            model: {k: round(v, 4) for k, v in bench_report.overall(model).items()}
            for model in bench_report.models
        },
        artifacts=artifacts,
    )
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    artifacts["summary"] = str(summary_path)
    return summary
