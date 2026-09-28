"""Command line interface for Model-Distillery."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from distillery.data.sft import load_jsonl
from distillery.errors import DistilleryError
from distillery.train.sft import TrainingConfig, build_training_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distillery", description="End-to-end model distillation pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pipeline = sub.add_parser("pipeline", help="Run the end-to-end pipeline.")
    pipeline.add_argument("--offline", action="store_true", help="Bundled data + EchoMock teacher.")
    pipeline.add_argument("--output-dir", default="artifacts/run", help="Artifact directory.")
    pipeline.add_argument("--input", default=None, help="Prompt pool JSONL (default: bundled).")
    pipeline.add_argument("--eval", default=None, help="Eval set JSONL (default: bundled).")
    pipeline.add_argument("--k", type=int, default=2, help="Candidates per prompt.")
    pipeline.add_argument("--limit", type=int, default=None, help="Cap the prompt pool size.")

    plan = sub.add_parser("plan", help="Emit a QLoRA training plan from an SFT JSONL file.")
    plan.add_argument("--input", required=True, help="SFT chat JSONL file.")
    plan.add_argument("--output", default="training_plan.md", help="Output markdown path.")
    plan.add_argument("--epochs", type=int, default=3)
    return parser


async def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "pipeline":
        from distillery.pipeline import PipelineSettings, run_pipeline

        settings = PipelineSettings(
            output_dir=Path(args.output_dir),
            offline=args.offline,
            k=args.k,
            limit=args.limit,
            prompts_path=Path(args.input) if args.input else None,
            eval_path=Path(args.eval) if args.eval else None,
        )
        summary = await run_pipeline(settings)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    if args.command == "plan":
        examples = load_jsonl(Path(args.input))
        plan = build_training_plan(examples, [], config=TrainingConfig(epochs=args.epochs))
        output = Path(args.output)
        output.write_text(plan.to_markdown(), encoding="utf-8")
        print(f"wrote {output} ({plan.steps} steps, {plan.est_tokens:,} tokens)")
        return 0

    return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; converts typed errors into a clean non-zero exit."""
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_dispatch(args))
    except DistilleryError as exc:
        # CLI boundary: any backend/data failure surfaces as a readable error
        # instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
