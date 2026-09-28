"""Quality filtering, judge hooks and rejection sampling.

Two-stage quality gate:

1. **Rules** (cheap, deterministic, short-circuit in a fixed order):
   ``length -> format -> repetition -> language``. Rules catch degenerate
   generations (rambling, placeholders, non-English output) without a model.
2. **Judge + rejection sampling**: a :class:`JudgeHook` scores the survivors and
   the best candidate per prompt is kept; losers are recorded with their discard
   reason, producing a full discard-reason histogram.

The :class:`JudgeHook` protocol is VerdictAI-compatible: any object exposing
``score(item: JudgeItem) -> float`` (in ``[0, 1]``) plugs in -- an offline
heuristic judge ships here, an LLM judge can be dropped in for API runs.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from distillery.teacher.generate import RawCandidate

_WORD_RE = re.compile(r"[a-z0-9']+")
_NUMBERED_RE = re.compile(r"^\d+[.)]")

# Broad English stopword sample used by the language heuristic.
_COMMON_WORDS = frozenset(
    ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"]
)

_MIN_RESPONSE_CHARS = 40
_MAX_RESPONSE_CHARS = 4000
_MIN_ALPHA_RATIO = 0.30
_UNIQUE_TOKEN_RATIO = 0.35
_MIN_TOKENS_FOR_REPETITION = 12
_REPEAT_SENTENCE_THRESHOLD = 3
_MIN_ASCII_LETTER_RATIO = 0.5
_MIN_COMMON_WORD_HITS = 2


@dataclass(frozen=True)
class JudgeItem:
    """Inputs a judge may look at when scoring one candidate."""

    prompt: str
    response: str
    capability: str


@runtime_checkable
class JudgeHook(Protocol):
    """VerdictAI-compatible scoring hook: ``score(item) -> float`` in ``[0, 1]``."""

    def score(self, item: JudgeItem) -> float: ...


class HeuristicJudge:
    """Offline reference judge.

    Blends four transparent signals: lexical diversity (0.35), length adequacy
    (0.25), structural formatting -- bullets/numbered lists (0.15) and prompt
    content-word overlap (0.25). Deterministic, no model call.
    """

    def score(self, item: JudgeItem) -> float:
        tokens = _WORD_RE.findall(item.response.lower())
        if not tokens:
            return 0.0
        diversity = len(set(tokens)) / len(tokens)
        length_adequacy = min(1.0, len(item.response) / 300.0)
        lines = [ln.strip() for ln in item.response.splitlines() if ln.strip()]
        structured = sum(
            1 for ln in lines if ln.startswith(("-", "*", ">")) or _NUMBERED_RE.match(ln)
        )
        structure = structured / len(lines) if lines else 0.0
        prompt_words = set(_WORD_RE.findall(item.prompt.lower())) - _COMMON_WORDS
        overlap = len(prompt_words & set(tokens)) / max(1, len(prompt_words))
        score = 0.35 * diversity + 0.25 * length_adequacy + 0.15 * structure + 0.25 * overlap
        return min(1.0, max(0.0, score))


@dataclass(frozen=True)
class RuleVerdict:
    rule: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: RawCandidate
    score: float


@dataclass(frozen=True)
class Discard:
    candidate: RawCandidate
    stage: str  # "rule" | "rejection"
    reason: str


@dataclass
class FilterResult:
    kept: list[ScoredCandidate] = field(default_factory=list)
    discards: list[Discard] = field(default_factory=list)
    reasons: Counter[str] = field(default_factory=Counter)

    @property
    def histogram(self) -> dict[str, int]:
        return dict(sorted(self.reasons.items()))

    def reason_total(self, *prefixes: str) -> int:
        return sum(n for key, n in self.reasons.items() if key.startswith(tuple(prefixes)))


class QualityFilter:
    """Rule stage + judge + rejection sampling over candidate groups."""

    def __init__(
        self,
        *,
        judge: JudgeHook | None = None,
        min_chars: int = _MIN_RESPONSE_CHARS,
        max_chars: int = _MAX_RESPONSE_CHARS,
    ) -> None:
        self._judge: JudgeHook = judge or HeuristicJudge()
        self._min_chars = min_chars
        self._max_chars = max_chars

    def check_rules(self, response: str) -> list[RuleVerdict]:
        """All rule verdicts (for introspection; ``run`` short-circuits)."""
        return [
            self._rule_length(response),
            self._rule_format(response),
            self._rule_repetition(response),
            self._rule_language(response),
        ]

    def first_failure(self, response: str) -> str | None:
        """Name of the first failing rule, or ``None`` when the response passes."""
        for verdict in self.check_rules(response):
            if not verdict.passed:
                return verdict.rule
        return None

    def run(self, candidates: Sequence[RawCandidate]) -> FilterResult:
        by_prompt: dict[str, list[RawCandidate]] = {}
        for candidate in candidates:
            by_prompt.setdefault(candidate.prompt_id, []).append(candidate)

        result = FilterResult()
        for prompt_id in sorted(by_prompt):
            group = by_prompt[prompt_id]
            valid: list[RawCandidate] = []
            for candidate in group:
                failure = self.first_failure(candidate.response)
                if failure is None:
                    valid.append(candidate)
                else:
                    result.discards.append(Discard(candidate, "rule", failure))
                    result.reasons[f"rule:{failure}"] += 1
            if not valid:
                result.reasons["no_valid_candidate"] += 1
                continue
            scored = [(self._score(candidate), candidate) for candidate in valid]
            # max() returns the FIRST maximal element, and group order is the
            # deterministic TeacherRunner order -- ties are broken stably.
            best_score, best = max(scored, key=lambda pair: pair[0])
            result.kept.append(ScoredCandidate(candidate=best, score=best_score))
            for _score, candidate in scored:
                if candidate is not best:
                    result.discards.append(Discard(candidate, "rejection", "rejection_sampling"))
                    result.reasons["rejection_sampling"] += 1
        return result

    def _score(self, candidate: RawCandidate) -> float:
        item = JudgeItem(
            prompt=candidate.user_content(), response=candidate.response, capability=candidate.capability
        )
        return float(self._judge.score(item))

    def _rule_length(self, response: str) -> RuleVerdict:
        n = len(response.strip())
        ok = self._min_chars <= n <= self._max_chars
        return RuleVerdict("length", ok, f"{n} chars (allowed {self._min_chars}..{self._max_chars})")

    def _rule_format(self, response: str) -> RuleVerdict:
        total = len(response)
        alpha = sum(ch.isalpha() for ch in response)
        ratio = alpha / total if total else 0.0
        ok = bool(response.strip()) and ratio >= _MIN_ALPHA_RATIO
        return RuleVerdict("format", ok, f"alphabetic ratio {ratio:.2f}")

    def _rule_repetition(self, response: str) -> RuleVerdict:
        tokens = _WORD_RE.findall(response.lower())
        if len(tokens) >= _MIN_TOKENS_FOR_REPETITION:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < _UNIQUE_TOKEN_RATIO:
                return RuleVerdict("repetition", False, f"unique token ratio {unique_ratio:.2f}")
        lines = [
            ln.strip().lstrip("-*").strip().lower()
            for ln in response.splitlines()
            if len(ln.strip()) > 20
        ]
        counts = Counter(lines)
        if counts:
            sentence, repeats = counts.most_common(1)[0]
            if repeats >= _REPEAT_SENTENCE_THRESHOLD:
                detail = f"sentence repeated {repeats}x: {sentence[:40]!r}"
                return RuleVerdict("repetition", False, detail)
        return RuleVerdict("repetition", True, "ok")

    def _rule_language(self, response: str) -> RuleVerdict:
        tokens = _WORD_RE.findall(response.lower())
        alpha = sum(ch.isalpha() for ch in response)
        if not tokens or alpha == 0:
            return RuleVerdict("language", False, "no alphabetic tokens")
        ascii_ratio = sum(ch.isascii() and ch.isalpha() for ch in response) / alpha
        if ascii_ratio < _MIN_ASCII_LETTER_RATIO:
            return RuleVerdict("language", False, f"ascii letter ratio {ascii_ratio:.2f}")
        common_hits = len(set(tokens) & _COMMON_WORDS)
        if common_hits < _MIN_COMMON_WORD_HITS:
            return RuleVerdict("language", False, f"only {common_hits} common-word hits")
        return RuleVerdict("language", True, "ok")
