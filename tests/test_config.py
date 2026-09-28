"""pydantic-settings configuration: defaults and env overrides."""

from distillery.config import DistillerySettings


def test_defaults() -> None:
    settings = DistillerySettings()
    assert settings.teacher_base_url == "http://localhost:8000/v1"
    assert settings.concurrency == 8
    assert settings.student_base_url is None


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("DISTILLERY_TEACHER_MODEL", "Qwen2.5-32B")
    monkeypatch.setenv("DISTILLERY_CONCURRENCY", "16")
    settings = DistillerySettings()
    assert settings.teacher_model == "Qwen2.5-32B"
    assert settings.concurrency == 16


def test_student_endpoint_falls_back_to_teacher(monkeypatch) -> None:
    settings = DistillerySettings()
    assert settings.student_endpoint() == (
        settings.teacher_base_url,
        settings.teacher_api_key,
        settings.teacher_model,
    )
    monkeypatch.setenv("DISTILLERY_STUDENT_BASE_URL", "http://student:8001/v1")
    monkeypatch.setenv("DISTILLERY_STUDENT_MODEL", "Qwen2.5-1.5B")
    overridden = DistillerySettings()
    assert overridden.student_endpoint()[0] == "http://student:8001/v1"
    assert overridden.student_endpoint()[2] == "Qwen2.5-1.5B"


def test_unknown_env_vars_ignored(monkeypatch) -> None:
    monkeypatch.setenv("DISTILLERY_NOT_A_THING", "x")
    DistillerySettings()  # extra="ignore" must not raise
