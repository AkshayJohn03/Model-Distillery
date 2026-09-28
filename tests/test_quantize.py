"""Quantize command builders: well-formed commands, dry-run only."""

import pytest

from distillery.quantize.awq import AWQConfig
from distillery.quantize.gguf import GGUFConfig


def test_gguf_build_commands() -> None:
    config = GGUFConfig(
        hf_model_dir="merged-student/",
        gguf_f16_path="student-f16.gguf",
        quantized_path="student-q4.gguf",
        quant_type="q4_k_m",
    )
    commands = config.build_commands()
    assert len(commands) == 2
    convert, quantize = commands
    assert convert == [
        "python",
        "convert_hf_to_gguf.py",
        "merged-student/",
        "--outfile",
        "student-f16.gguf",
        "--outtype",
        "f16",
    ]
    assert quantize == ["llama-quantize", "student-f16.gguf", "student-q4.gguf", "q4_k_m"]
    for command in commands:
        assert all(isinstance(arg, str) and arg for arg in command)


def test_gguf_rejects_unknown_quant_type() -> None:
    with pytest.raises(ValueError, match="quant_type"):
        GGUFConfig("m/", "a.gguf", "b.gguf", quant_type="q99_z")


def test_gguf_docs_render_commands() -> None:
    config = GGUFConfig("m/", "a.gguf", "b.gguf")
    docs = config.build_docs()
    assert "llama-quantize a.gguf b.gguf q4_k_m" in docs
    assert "NOT executed in CI" in docs


def test_awq_command_flags_are_paired() -> None:
    config = AWQConfig(
        model_path="merged-student/",
        out_dir="student-awq/",
        calibration_path="data/prompts.jsonl",
    )
    command = config.build_command()
    assert command[:3] == ["python", "-m", "distillery.quantize.awq"]
    for i, arg in enumerate(command):
        if arg.startswith("--"):
            assert i + 1 < len(command), f"flag {arg} has no value"
            assert not command[i + 1].startswith("--") or arg == "--no-zero-point"
    assert command[command.index("--w-bit") : command.index("--w-bit") + 2] == ["--w-bit", "4"]
    assert command[
        command.index("--calibration-path") :
        command.index("--calibration-path") + 2
    ] == ["--calibration-path", "data/prompts.jsonl"]


def test_awq_no_zero_point_flag() -> None:
    config = AWQConfig(model_path="m/", out_dir="o/", zero_point=False)
    assert "--no-zero-point" in config.build_command()
    assert AWQConfig(model_path="m/", out_dir="o/").build_command().count("--no-zero-point") == 0


@pytest.mark.parametrize("field, value", [("w_bit", 5), ("q_group_size", 100), ("version", "turbo")])
def test_awq_validates_settings(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        AWQConfig(model_path="m/", out_dir="o/", **{field: value})
