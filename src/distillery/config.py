"""Environment-driven configuration via pydantic-settings.

Every variable uses the ``DISTILLERY_`` prefix; see ``.env.example``.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DistillerySettings(BaseSettings):
    """Teacher/student endpoints and client behaviour."""

    model_config = SettingsConfigDict(env_prefix="DISTILLERY_", env_file=".env", extra="ignore")

    teacher_base_url: str = "http://localhost:8000/v1"
    teacher_api_key: str = "sk-local"
    teacher_model: str = "demo-teacher"

    # When unset, bench step falls back to the teacher endpoint settings.
    student_base_url: str | None = None
    student_api_key: str | None = None
    student_model: str | None = None

    request_timeout: float = 60.0
    concurrency: int = 8

    def student_endpoint(self) -> tuple[str, str, str]:
        """Return ``(base_url, api_key, model)`` for the student, defaulting to teacher."""
        return (
            self.student_base_url or self.teacher_base_url,
            self.student_api_key or self.teacher_api_key,
            self.student_model or self.teacher_model,
        )
