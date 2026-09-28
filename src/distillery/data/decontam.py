"""N-gram decontamination against an evaluation corpus.

Training examples that share a long verbatim n-gram (default 13 tokens, the
standard from the GPT-3/PaLM lineage) with an eval document are dropped, with
the overlapping n-gram retained as evidence. 13 tokens is long enough that a
match is almost never a coincidence of fluent English, yet short enough to
catch partial paraphrases that reuse a distinctive span -- exactly the failure
mode that inflates benchmark scores via memorization.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization (punctuation dropped)."""
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


@dataclass(frozen=True)
class ContaminationHit:
    item_id: str
    eval_doc_id: str
    ngram: str
    position: int


@dataclass
class ContaminationResult:
    n: int
    evaluated: int
    contaminated_ids: list[str] = field(default_factory=list)
    hits: list[ContaminationHit] = field(default_factory=list)
    all_ids: list[str] = field(default_factory=list)

    @property
    def clean_ids(self) -> list[str]:
        contaminated = set(self.contaminated_ids)
        return [i for i in self.all_ids if i not in contaminated]


class NgramDecontaminator:
    """Index an eval corpus once, then screen any number of training items."""

    def __init__(self, *, n: int = 13, max_evidence_per_item: int = 5) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n
        self._max_evidence = max_evidence_per_item
        self._index: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self._n_docs = 0

    @property
    def n(self) -> int:
        return self._n

    def index_eval_corpus(self, docs: Mapping[str, str]) -> None:
        for doc_id, text in docs.items():
            self._n_docs += 1
            for gram in ngrams(tokenize(text), self._n):
                self._index[gram].add(doc_id)

    def check(self, item_id: str, text: str) -> list[ContaminationHit]:
        """Return evidence hits for one item (capped at ``max_evidence_per_item``)."""
        hits: list[ContaminationHit] = []
        for position, gram in enumerate(ngrams(tokenize(text), self._n)):
            doc_ids = self._index.get(gram)
            if doc_ids:
                hits.append(
                    ContaminationHit(
                        item_id=item_id,
                        eval_doc_id=sorted(doc_ids)[0],
                        ngram=" ".join(gram),
                        position=position,
                    )
                )
                if len(hits) >= self._max_evidence:
                    break
        return hits

    def screen(self, items: Mapping[str, str]) -> ContaminationResult:
        contaminated: list[str] = []
        hits: list[ContaminationHit] = []
        for item_id in sorted(items):
            item_hits = self.check(item_id, items[item_id])
            if item_hits:
                contaminated.append(item_id)
                hits.extend(item_hits)
        return ContaminationResult(
            n=self._n,
            evaluated=len(items),
            contaminated_ids=contaminated,
            hits=hits,
            all_ids=sorted(items),
        )
