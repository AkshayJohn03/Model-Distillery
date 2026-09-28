"""QLoRA training plan: deterministic estimation and markdown output."""

import importlib.util
import math

import pytest

from distillery.errors import DependencyMissingError
from distillery.train.sft import (
    TrainingConfig,
    build_training_plan,
    estimate_lora_params,
    estimate_memory_gb,
    run_qlora_training,
)


def _examples(make_example, n: int):
    return [make_example("summarization", i) for i in range(n)]


def test_step_formula(make_example) -> None:
    train = _examples(make_example, 40)
    config = TrainingConfig()  # bs 2 x accum 8 = 16, 3 epochs
    plan = build_training_plan(train, [], config=config)
    assert plan.effective_batch == 16
    assert plan.steps == math.ceil(40 * 3 / 16) == 8


def test_token_estimate_matches_manual_computation(make_example) -> None:
    train = _examples(make_example, 5)
    plan = build_training_plan(train, [])
    manual = 0
    for example in train:
        for message in example.messages:
            manual += int(len(message.content) / 4.0) + 4
    assert plan.est_tokens == manual * 3


def test_memory_and_gpu_hours_monotonic_in_model_size(make_example) -> None:
    plan = build_training_plan(_examples(make_example, 40), [])
    memories = [row.est_memory_gb for row in plan.rows]
    hours = [row.est_gpu_hours for row in plan.rows]
    assert memories == sorted(memories)
    assert hours == sorted(hours)
    by_size = {row.model_size_b: row for row in plan.rows}
    assert by_size[7].est_memory_gb < 16.0  # QLoRA keeps 7B well under 16 GB
    assert by_size[70].est_memory_gb > 40.0


def test_lora_param_estimate_7b() -> None:
    config = TrainingConfig()
    params = estimate_lora_params(7e9, hidden=4096, layers=32, lora=config.lora)
    assert params == 2 * 16 * 4096 * 32 * 4  # r(d+d) per matrix, 4 targets
    assert params / 7e9 < 0.005  # adapters stay under 0.5% of the base model


def test_memory_estimate_includes_flat_overhead() -> None:
    config = TrainingConfig()
    assert estimate_memory_gb(7, 4096, 32, config) > 7 * 0.55 + 2


def test_plan_markdown_contents(make_example) -> None:
    plan = build_training_plan(_examples(make_example, 40), _examples(make_example, 4))
    markdown = plan.to_markdown()
    assert "# QLoRA Training Plan" in markdown
    assert "| optimizer steps | 8 |" in markdown
    assert "NF4" in markdown or "nf4" in markdown
    assert "| 70B |" in markdown
    assert "GPU-hours" in markdown


def test_plan_is_deterministic(make_example) -> None:
    a = build_training_plan(_examples(make_example, 40), [])
    b = build_training_plan(_examples(make_example, 40), [])
    assert a == b


def test_unknown_model_size_rejected(make_example) -> None:
    with pytest.raises(ValueError, match="architecture estimate"):
        build_training_plan(_examples(make_example, 4), [], model_sizes=(2,))


def test_torch_recipe_is_guarded(make_example) -> None:
    train = _examples(make_example, 4)
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(DependencyMissingError, match="model-distillery\\[torch\\]"):
            run_qlora_training(train, [], TrainingConfig(), model_name="dummy", output_dir="out")
    else:  # pragma: no cover - only when the torch extra is installed
        pytest.skip("torch installed: full recipe exercised manually, not in CI")
