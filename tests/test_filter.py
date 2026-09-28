"""Quality filter rules, judge hook and rejection sampling."""

from distillery.teacher.filter import HeuristicJudge, JudgeItem, QualityFilter

GOOD = (
    "Key points:\n"
    "- The quarterly revenue report shows steady growth for the board meeting.\n"
    "- Costs stayed flat while the board meeting approved new hiring.\n"
    "- The meeting closed with a revenue forecast for next quarter."
)


def test_rule_length_fires(make_candidate) -> None:
    assert QualityFilter().first_failure("Too short.") == "length"


def test_rule_format_fires_on_non_text(make_candidate) -> None:
    junk = "!!!! #### %%%% #### !!!! #### %%%% #### !!!! ####"
    assert len(junk) >= 40
    assert QualityFilter().first_failure(junk) == "format"


def test_rule_repetition_fires(make_candidate) -> None:
    repeated = "The system failed to restart the cache worker after the outage. " * 4
    assert QualityFilter().first_failure(repeated) == "repetition"


def test_rule_language_fires_on_gibberish(make_candidate) -> None:
    gibberish = "zx qv jj kk pp ww tt yy uu oo pp aa ss dd ff gg hh ii uu"
    assert QualityFilter().first_failure(gibberish) == "language"


def test_good_response_passes_all_rules() -> None:
    assert QualityFilter().first_failure(GOOD) is None


def test_rules_short_circuit_in_fixed_order() -> None:
    verdicts = QualityFilter().check_rules("Hi.")
    assert [v.rule for v in verdicts] == ["length", "format", "repetition", "language"]
    assert verdicts[0].passed is False
    assert verdicts[1].passed is True


def test_rejection_sampling_keeps_best_by_judge_score(make_candidate) -> None:
    rich = make_candidate(
        prompt_id="p1",
        seed=1,
        response=(
            "Key points:\n"
            "- Quarterly revenue grew in the report prepared for the board meeting.\n"
            "- The board meeting reviewed the revenue report line by line.\n"
            "- Meeting notes attached for the quarterly revenue review."
        ),
    )
    mid = make_candidate(
        prompt_id="p1",
        seed=2,
        response=(
            "The quarterly revenue report was presented at the board meeting and discussed "
            "at length by everyone present there today."
        ),
    )
    short = make_candidate(
        prompt_id="p1", seed=3, response="The report is fine and the board has seen it all before."
    )
    result = QualityFilter().run([short, rich, mid])
    assert len(result.kept) == 1
    assert result.kept[0].candidate.response == rich.response
    assert result.histogram["rejection_sampling"] == 2
    assert result.kept[0].score > 0.6


def test_rule_discards_and_no_valid_prompt(make_candidate) -> None:
    bad = make_candidate(prompt_id="p1", response="Too short.")
    worse = make_candidate(prompt_id="p1", response="No.")
    other = make_candidate(prompt_id="p2", response=GOOD)
    result = QualityFilter().run([bad, worse, other])
    assert [sc.candidate.prompt_id for sc in result.kept] == ["p2"]
    assert result.histogram["rule:length"] == 2
    assert result.histogram["no_valid_candidate"] == 1


def test_heuristic_judge_bounds_and_ranking() -> None:
    judge = HeuristicJudge()
    item = JudgeItem(prompt="Summarize the quarterly revenue report.", response=GOOD, capability="x")
    score = judge.score(item)
    assert 0.0 <= score <= 1.0
    assert score > judge.score(
        JudgeItem(prompt=item.prompt, response="Fine.", capability="x")
    )


def test_judge_returns_zero_for_empty_response() -> None:
    assert HeuristicJudge().score(JudgeItem(prompt="p", response="", capability="x")) == 0.0
