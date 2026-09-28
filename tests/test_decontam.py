"""13-gram decontamination: planted overlap caught, near-miss passes."""

from distillery.data.decontam import NgramDecontaminator, ngrams, tokenize

# 14-token span; any 13-window of it must match.
PHRASE = (
    "migratory patterns of arctic terns shift dramatically during unusually warm "
    "pacific decadal oscillation years"
)
EVAL_DOC = f"Research notes: {PHRASE} tracking continues."
CONTAMINATED_ITEM = f"The team observed that {PHRASE} this season."
# Shares only the first 12 tokens -> no 13-gram can match.
NEAR_MISS_ITEM = (
    "The team observed that migratory patterns of arctic terns shift dramatically "
    "during unusually warm pacific decadal this season."
)
CLEAN_ITEM = "Completely unrelated notes about the museum bakery and its wholesale counter."


def test_tokenize_and_ngrams() -> None:
    tokens = tokenize("The Bay-Fishery reopened!")
    assert tokens == ["the", "bay", "fishery", "reopened"]
    assert len(ngrams(tokens, 13)) == 0
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]


def test_planted_overlap_is_caught_with_evidence() -> None:
    decon = NgramDecontaminator(n=13)
    decon.index_eval_corpus({"eval-1": EVAL_DOC})
    hits = decon.check("item-1", CONTAMINATED_ITEM)
    assert hits, "13-gram overlap must be flagged"
    assert hits[0].eval_doc_id == "eval-1"
    assert "arctic terns shift" in hits[0].ngram
    assert len(hits) <= 5  # evidence is capped


def test_twelve_token_overlap_is_not_flagged() -> None:
    decon = NgramDecontaminator(n=13)
    decon.index_eval_corpus({"eval-1": EVAL_DOC})
    assert decon.check("near-miss", NEAR_MISS_ITEM) == []


def test_clean_item_passes() -> None:
    decon = NgramDecontaminator(n=13)
    decon.index_eval_corpus({"eval-1": EVAL_DOC})
    assert decon.check("clean", CLEAN_ITEM) == []


def test_short_items_are_handled() -> None:
    decon = NgramDecontaminator(n=13)
    decon.index_eval_corpus({"eval-1": EVAL_DOC})
    assert decon.check("tiny", "two words") == []


def test_screen_partitions_items() -> None:
    decon = NgramDecontaminator(n=13)
    decon.index_eval_corpus({"eval-1": EVAL_DOC})
    result = decon.screen(
        {"a": CONTAMINATED_ITEM, "b": CLEAN_ITEM, "c": NEAR_MISS_ITEM}
    )
    assert result.contaminated_ids == ["a"]
    assert result.clean_ids == ["b", "c"]
    assert result.evaluated == 3
    assert result.n == 13


def test_default_n_is_thirteen() -> None:
    assert NgramDecontaminator().n == 13
