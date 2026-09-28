"""Cost model: hand-computed economics and report rendering."""

import pytest

from distillery.eval.cost import (
    ModelPrice,
    UsageProfile,
    blended_per_1m,
    breakeven_months,
    build_cost_report,
    monthly_cost,
    request_cost,
)

TEACHER = ModelPrice("teacher", 3.00, 15.00)
STUDENT = ModelPrice("student", 0.15, 0.60)
USAGE = UsageProfile(avg_input_tokens=900, avg_output_tokens=250, monthly_requests=100_000)


def test_request_cost_hand_computed() -> None:
    teacher = request_cost(TEACHER, USAGE)
    assert teacher.input_cost == pytest.approx(900 * 3.0 / 1e6)
    assert teacher.total == pytest.approx((900 * 3.0 + 250 * 15.0) / 1e6)  # 0.00645


def test_monthly_cost_and_savings() -> None:
    teacher_month = monthly_cost(TEACHER, USAGE)  # 0.00645 * 100k = 645.0
    student_month = monthly_cost(STUDENT, USAGE)  # 0.000285 * 100k = 28.5
    assert teacher_month == pytest.approx(645.0, abs=1e-6)
    assert student_month == pytest.approx(28.5, abs=1e-6)
    assert teacher_month - student_month == pytest.approx(616.5, abs=1e-6)


def test_blended_per_1m() -> None:
    assert blended_per_1m(TEACHER, USAGE) == pytest.approx(0.00645 / 1150 * 1e6)
    assert blended_per_1m(TEACHER, UsageProfile(0, 0, 10)) == 0.0


def test_breakeven() -> None:
    assert breakeven_months(60.0, 616.5) == pytest.approx(60.0 / 616.5, abs=1e-9)
    assert breakeven_months(60.0, 0.0) is None
    assert breakeven_months(60.0, -5.0) is None


def test_cost_report_renders_curve_and_breakeven() -> None:
    report = build_cost_report(
        TEACHER,
        STUDENT,
        USAGE,
        quality_retention_pct=68.61,
        investment_usd=60.0,
    )
    assert "Quality retention" in report
    assert "68.6%" in report
    assert "Breakeven horizon" in report
    assert "| 1,000,000 |" in report  # volume curve rows
    assert "teacher" in report and "student" in report
