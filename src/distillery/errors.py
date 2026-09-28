"""Exception hierarchy for the distillery package."""

from __future__ import annotations


class DistilleryError(Exception):
    """Base class for all distillery errors."""


class DependencyMissingError(DistilleryError):
    """Raised when an optional heavy dependency (torch/peft/fastapi) is unavailable."""

    def __init__(self, feature: str, package_hint: str) -> None:
        super().__init__(
            f"{feature} requires optional dependencies that are not installed. "
            f"Install them with: pip install {package_hint}"
        )
        self.feature = feature
        self.package_hint = package_hint


class TeacherClientError(DistilleryError):
    """Raised when the teacher LLM client fails (HTTP error, malformed payload)."""


class DataValidationError(DistilleryError):
    """Raised when bundled or user-provided data files are malformed."""
