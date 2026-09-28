"""MinHash dedup: planted near-dupe caught, distinct item kept, canonical wins."""

from distillery.teacher.dedup import DedupItem, MinHashDeduplicator, jaccard, shingles

BASE = (
    "The incident began when the primary database cluster failed over to the secondary "
    "region during the nightly backup window. On call engineers noticed replication lag "
    "climbing within four minutes and paged the platform rotation immediately. Customer "
    "facing writes were queued while the cache layer served stale reads for eleven "
    "minutes. After failback the team ran a full consistency audit and found no lost "
    "transactions. The postmortem recommended faster DNS propagation and a quarterly "
    "failover drill."
)
# Two word substitutions in a ~75-word text: <=6 of ~73 shingles broken.
NEAR_DUPE = BASE.replace("database cluster failed", "database array failed").replace(
    "eleven minutes", "twelve minutes"
)
DISTINCT = (
    "The recipe calls for slowly caramelized onions, smoked paprika, and a splash of "
    "sherry vinegar added at the end. Brown the short ribs first, then braise them for "
    "three hours in the oven. Serve the stew with crusty bread and a sharp green salad."
)


def test_planted_near_duplicate_is_caught() -> None:
    result = MinHashDeduplicator().deduplicate(
        [
            DedupItem(id="a", text=BASE),
            DedupItem(id="b", text=NEAR_DUPE),
            DedupItem(id="c", text=DISTINCT),
        ]
    )
    assert [(d.removed_id, d.canonical_id) for d in result.duplicates] == [("b", "a")]
    assert result.duplicates[0].jaccard >= 0.8
    assert [item.id for item in result.kept] == ["a", "c"]
    assert result.clusters == [("a", "b")]


def test_first_occurrence_is_canonical() -> None:
    result = MinHashDeduplicator().deduplicate(
        [DedupItem(id="x2", text=BASE), DedupItem(id="x1", text=BASE), DedupItem(id="x3", text=BASE)]
    )
    assert [item.id for item in result.kept] == ["x2"]
    assert len(result.duplicates) == 2
    assert all(d.canonical_id == "x2" for d in result.duplicates)


def test_below_threshold_pairs_are_kept() -> None:
    heavily_edited = " ".join(
        BASE.split()[:20] + ["totally", "unrelated", "replacement", "words", "appear", "here", "now", "in", "force"] * 2
    )
    result = MinHashDeduplicator().deduplicate(
        [DedupItem(id="a", text=BASE), DedupItem(id="b", text=heavily_edited)]
    )
    assert result.duplicates == []
    assert [item.id for item in result.kept] == ["a", "b"]


def test_dedup_is_deterministic() -> None:
    items = [DedupItem(id="a", text=BASE), DedupItem(id="b", text=NEAR_DUPE), DedupItem(id="c", text=DISTINCT)]
    one = MinHashDeduplicator().deduplicate(items)
    two = MinHashDeduplicator().deduplicate(items)
    assert one.kept == two.kept
    assert one.duplicates == two.duplicates


def test_jaccard_and_shingle_edges() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, set()) == 0.0
    assert shingles("", 3) == set()
    assert shingles("one two", 3)  # short text still yields a single shingle
    assert shingles("one two", 3) == shingles("ONE  TWO", 3)  # case/whitespace normalized
