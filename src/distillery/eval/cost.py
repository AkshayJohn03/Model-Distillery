"""Cost comparison: teacher API vs distilled student, with breakeven analysis.

The economic case for distillation is an amortization curve: the one-time
investment (teacher generation + training + eval) is recovered through the
per-token savings of the smaller model. This module renders that curve as a
markdown table over monthly request volumes plus the breakeven horizon.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens."""

    name: str
    input_per_1m: float
    output_per_1m: float


@dataclass(frozen=True)
class UsageProfile:
    avg_input_tokens: int
    avg_output_tokens: int
    monthly_requests: int


class RequestCost(NamedTuple):
    input_cost: float
    output_cost: float

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost


def request_cost(price: ModelPrice, usage: UsageProfile) -> RequestCost:
    """USD cost of one average request."""
    return RequestCost(
        input_cost=usage.avg_input_tokens * price.input_per_1m / 1e6,
        output_cost=usage.avg_output_tokens * price.output_per_1m / 1e6,
    )


def monthly_cost(price: ModelPrice, usage: UsageProfile) -> float:
    return request_cost(price, usage).total * usage.monthly_requests


def blended_per_1m(price: ModelPrice, usage: UsageProfile) -> float:
    """USD per 1M blended tokens under the given usage mix."""
    tokens = usage.avg_input_tokens + usage.avg_output_tokens
    if tokens == 0:
        return 0.0
    return request_cost(price, usage).total / tokens * 1e6


def breakeven_months(investment_usd: float, monthly_savings_usd: float) -> float | None:
    """Months to recover the distillation investment; ``None`` when never."""
    if monthly_savings_usd <= 0:
        return None
    return investment_usd / monthly_savings_usd


_VOLUME_CURVE: tuple[int, ...] = (10_000, 50_000, 100_000, 500_000, 1_000_000)


def build_cost_report(
    teacher_price: ModelPrice,
    student_price: ModelPrice,
    usage: UsageProfile,
    quality_retention_pct: float,
    investment_usd: float,
    volumes: Sequence[int] = _VOLUME_CURVE,
) -> str:
    """Full markdown cost comparison including the volume/breakeven curve."""
    teacher_req = request_cost(teacher_price, usage)
    student_req = request_cost(student_price, usage)
    per_req_savings = teacher_req.total - student_req.total
    teacher_month = monthly_cost(teacher_price, usage)
    student_month = monthly_cost(student_price, usage)
    savings_month = teacher_month - student_month
    breakeven = breakeven_months(investment_usd, savings_month)

    lines = [
        "# Cost Comparison: Teacher vs Distilled Student",
        "",
        f"Quality retention (student token_f1 / teacher token_f1): **{quality_retention_pct:.1f}%**",
        "",
        "## Per-Request Economics",
        "",
        "| Model | $/1M in | $/1M out | $/request (blended) |",
        "| --- | ---: | ---: | ---: |",
        f"| {teacher_price.name} | {teacher_price.input_per_1m:.2f} | {teacher_price.output_per_1m:.2f} "
        f"| {teacher_req.total:.6f} |",
        f"| {student_price.name} | {student_price.input_per_1m:.2f} | {student_price.output_per_1m:.2f} "
        f"| {student_req.total:.6f} |",
        f"| **savings** |  |  | **{per_req_savings:.6f}** |",
        "",
        f"Usage profile: {usage.monthly_requests:,} requests/month, "
        f"{usage.avg_input_tokens} in + {usage.avg_output_tokens} out tokens/request.",
        "",
        "## Monthly Cost at Current Volume",
        "",
        "| Model | $/month |",
        "| --- | ---: |",
        f"| {teacher_price.name} | {teacher_month:,.2f} |",
        f"| {student_price.name} | {student_month:,.2f} |",
        f"| **monthly savings** | **{savings_month:,.2f}** |",
        "",
        f"One-time distillation investment: ${investment_usd:,.2f} "
        "(data generation + training + eval).",
    ]
    if breakeven is None:
        lines.append("Breakeven: **never** at this volume (no savings).")
    else:
        lines.append(
            f"Breakeven horizon: **{breakeven:.2f} months** "
            f"(~{breakeven * 30.44:.0f} days)."
        )

    lines += [
        "",
        "## Cost Curve Across Volumes",
        "",
        "| Requests/month | Teacher $/mo | Student $/mo | Savings $/mo | Breakeven (mo) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for volume in volumes:
        scaled = UsageProfile(usage.avg_input_tokens, usage.avg_output_tokens, volume)
        t = monthly_cost(teacher_price, scaled)
        s = monthly_cost(student_price, scaled)
        be = breakeven_months(investment_usd, t - s)
        be_text = f"{be:.2f}" if be is not None else "--"
        lines.append(f"| {volume:,} | {t:,.0f} | {s:,.0f} | {t - s:,.0f} | {be_text} |")
    lines += [
        "",
        "Takeaway: savings scale linearly with volume while the investment is fixed, "
        "so higher traffic only shortens payback; the decision hinge is whether "
        f"quality retention ({quality_retention_pct:.1f}%) is acceptable for the workload.",
        "",
    ]
    return "\n".join(lines)
