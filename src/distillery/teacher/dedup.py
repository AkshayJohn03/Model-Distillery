"""MinHash + banded LSH near-duplicate detection.

Pure python/numpy-free implementation:

1. Texts are converted to word n-gram ("shingle") sets.
2. Each set is reduced to a ``num_perm``-dimensional MinHash signature built
   from universal hash functions ``h(x) = (a*x + b) mod p`` with ``p = 2^61 - 1``.
   Shingles are hashed with keyed BLAKE2b so signatures are stable across
   processes (unlike builtin ``hash``).
3. Banded LSH (``bands`` bands of ``num_perm / bands`` rows) proposes candidate
   pairs that share any band signature -- probability of a candidate at Jaccard
   ``J`` is ``1 - (1 - J^r)^b``.
4. Candidates are verified with exact Jaccard and merged into clusters via
   union-find; the earliest item in input order is the canonical survivor.

With the defaults (128 perms, 16 bands of 8 rows) the S-curve midpoint sits at
``J ~ 0.71`` and a pair with ``J = 0.8`` is caught with ~95% probability per
pass; verification makes the reported precision exact.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

_MERSENNE = (1 << 61) - 1


def stable_hash(text: str) -> int:
    """Process-stable 64-bit hash (builtin ``hash`` is salted per process)."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def shingles(text: str, size: int = 3) -> set[int]:
    """Word n-gram shingle set of a text."""
    tokens = text.lower().split()
    if not tokens:
        return set()
    if len(tokens) < size:
        return {stable_hash(" ".join(tokens))}
    return {stable_hash(" ".join(tokens[i : i + size])) for i in range(len(tokens) - size + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MinHasher:
    """``num_perm`` universal hash functions ``(a, b)`` over shingle IDs."""

    def __init__(self, num_perm: int = 128, seed: int = 1234) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        rng = random.Random(seed)
        self._num_perm = num_perm
        self._a = [rng.randrange(1, _MERSENNE) for _ in range(num_perm)]
        self._b = [rng.randrange(0, _MERSENNE) for _ in range(num_perm)]

    def signature(self, hashes: set[int]) -> tuple[int, ...]:
        if not hashes:
            return (_MERSENNE,) * self._num_perm
        return tuple(
            min((a * h + b) % _MERSENNE for h in hashes) for a, b in zip(self._a, self._b, strict=True)
        )


def lsh_candidate_pairs(
    signatures: Mapping[str, tuple[int, ...]], bands: int
) -> set[tuple[str, str]]:
    """Banded LSH: return candidate pairs sharing at least one band signature."""
    if not signatures:
        return set()
    num_perm = len(next(iter(signatures.values())))
    if bands < 1 or num_perm % bands != 0:
        raise ValueError("bands must divide num_perm")
    rows = num_perm // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for item_id, sig in signatures.items():
        for band in range(bands):
            key = (band, sig[band * rows : (band + 1) * rows])
            buckets.setdefault(key, []).append(item_id)
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(tuple(sorted((members[i], members[j]))))
    return pairs


class _UnionFind:
    def __init__(self, ids: Sequence[str]) -> None:
        self._parent: dict[str, str] = {i: i for i in ids}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union_earlier_wins(self, a: str, b: str, order: Mapping[str, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if order[ra] <= order[rb]:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb


@dataclass(frozen=True)
class DedupItem:
    id: str
    text: str


class DuplicatePair(NamedTuple):
    removed_id: str
    canonical_id: str
    jaccard: float


@dataclass
class DedupResult:
    kept: list[DedupItem] = field(default_factory=list)
    duplicates: list[DuplicatePair] = field(default_factory=list)
    clusters: list[tuple[str, ...]] = field(default_factory=list)


class MinHashDeduplicator:
    """Near-duplicate clustering over :class:`DedupItem` sequences."""

    def __init__(
        self,
        *,
        threshold: float = 0.8,
        shingle_size: int = 3,
        num_perm: int = 128,
        bands: int = 16,
        seed: int = 1234,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._threshold = threshold
        self._shingle_size = shingle_size
        self._num_perm = num_perm
        self._bands = bands
        self._seed = seed

    def deduplicate(self, items: Sequence[DedupItem]) -> DedupResult:
        shingle_sets = [shingles(it.text, self._shingle_size) for it in items]
        hasher = MinHasher(self._num_perm, seed=self._seed)
        signatures = {it.id: hasher.signature(s) for it, s in zip(items, shingle_sets, strict=True)}
        order = {it.id: i for i, it in enumerate(items)}

        union_find = _UnionFind([it.id for it in items])
        verified: dict[tuple[str, str], float] = {}
        for a, b in sorted(lsh_candidate_pairs(signatures, self._bands)):
            score = jaccard(shingle_sets[order[a]], shingle_sets[order[b]])
            if score < self._threshold:
                continue
            verified[(a, b)] = score
            union_find.union_earlier_wins(a, b, order)

        roots = {it.id: union_find.find(it.id) for it in items}
        kept = [it for it in items if roots[it.id] == it.id]
        # One entry per removed item (not per candidate pair, which would
        # over-count cliques of identical texts); canonical = cluster root,
        # which union_earlier_wins guarantees is the earliest input item.
        duplicates = [
            DuplicatePair(
                removed_id=it.id,
                canonical_id=roots[it.id],
                jaccard=max(s for (a, b), s in verified.items() if it.id in (a, b)),
            )
            for it in items
            if roots[it.id] != it.id
        ]
        clusters: dict[str, list[str]] = {}
        for it in items:
            clusters.setdefault(roots[it.id], []).append(it.id)
        return DedupResult(
            kept=kept,
            duplicates=duplicates,
            clusters=[tuple(members) for members in clusters.values() if len(members) > 1],
        )
