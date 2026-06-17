"""Tests for rulechef.ranking — per-rule ranking, ablation, pruning."""

from rulechef.core import Dataset, Example, Rule, RuleFormat, Task, TaskType
from rulechef.executor import RuleExecutor
from rulechef.ranking import prune_harmful_rules, rank_rules, wilson_lower_bound


class TestWilsonLowerBound:
    def test_zero_support_is_untrusted(self):
        assert wilson_lower_bound(1.0, 0) == 0.0

    def test_low_support_discounted(self):
        # 1/1 correct should be trusted far less than 88/88
        assert wilson_lower_bound(1.0, 1) < 0.3
        assert wilson_lower_bound(1.0, 88) > 0.95

    def test_monotonic_in_support(self):
        bounds = [wilson_lower_bound(0.9, n) for n in (5, 20, 100, 1000)]
        assert bounds == sorted(bounds)

    def test_bounded_by_precision(self):
        assert wilson_lower_bound(0.8, 50) < 0.8
        assert wilson_lower_bound(0.8, 50) > 0.0


def _task():
    return Task(
        name="clf",
        description="classify",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )


def _dataset():
    dataset = Dataset(name="clf-data", task=_task())
    rows = [
        ("how is the exchange rate", "exchange_rate"),
        ("exchange rate for euros", "exchange_rate"),
        ("my card has not arrived", "card_arrival"),
        ("when does my card arrive", "card_arrival"),
        ("track my card delivery", "card_arrival"),
    ]
    for i, (text, label) in enumerate(rows):
        dataset.examples.append(
            Example(
                id=f"e{i}",
                input={"text": text},
                expected_output={"label": label},
                source="test",
            )
        )
    return dataset


def _rule(rule_id, label, pattern, priority=5):
    return Rule(
        id=rule_id,
        name=rule_id,
        description="",
        format=RuleFormat.REGEX,
        content=pattern,
        priority=priority,
        output_template={"label": label},
        output_key="label",
    )


def _apply_rules_fn():
    executor = RuleExecutor()
    return lambda rules, input_data, task_type=None, text_field=None: executor.apply_rules(
        rules, input_data, task_type, text_field
    )


class TestRankRules:
    def test_solo_precision_computed(self):
        rules = [
            _rule("good", "exchange_rate", r"exchange rate"),
            _rule("bad", "card_arrival", r"."),  # matches everything
        ]
        report = rank_rules(rules, _dataset(), _apply_rules_fn(), compute_marginal=False)
        by_id = {r.rule_id: r for r in report.rankings}
        assert by_id["good"].solo_precision == 1.0
        assert by_id["bad"].solo_precision < 1.0

    def test_stamps_validated_stats(self):
        rules = [_rule("good", "exchange_rate", r"exchange rate")]
        rank_rules(rules, _dataset(), _apply_rules_fn(), compute_marginal=False)
        assert rules[0].validated_precision == 1.0
        assert rules[0].validated_support == 2

    def test_marginal_contribution(self):
        rules = [
            _rule("exchange", "exchange_rate", r"exchange rate"),
            _rule("card", "card_arrival", r"card"),
        ]
        report = rank_rules(rules, _dataset(), _apply_rules_fn(), compute_marginal=True)
        by_id = {r.rule_id: r for r in report.rankings}
        # Each rule covers its own class — removing either drops F1
        assert by_id["exchange"].marginal_f1 > 0
        assert by_id["card"].marginal_f1 > 0

    def test_harmful_rule_has_negative_marginal(self):
        rules = [
            _rule("exchange", "exchange_rate", r"exchange rate", priority=5),
            _rule("card", "card_arrival", r"card", priority=5),
            # Always fires with the wrong label and outranks others via priority
            _rule("noise", "exchange_rate", r".", priority=10),
        ]
        report = rank_rules(rules, _dataset(), _apply_rules_fn(), compute_marginal=True)
        by_id = {r.rule_id: r for r in report.rankings}
        assert by_id["noise"].marginal_f1 < 0

    def test_empty_rules(self):
        report = rank_rules([], _dataset(), _apply_rules_fn())
        assert report.rankings == []

    def test_rankings_sorted_most_valuable_first(self):
        rules = [
            _rule("noise", "exchange_rate", r".", priority=10),
            _rule("exchange", "exchange_rate", r"exchange rate"),
        ]
        report = rank_rules(rules, _dataset(), _apply_rules_fn(), compute_marginal=True)
        assert report.rankings[0].rule_id == "exchange"
        assert report.rankings[-1].rule_id == "noise"


class TestPruneHarmfulRules:
    def test_prunes_negative_marginal(self):
        rules = [
            _rule("exchange", "exchange_rate", r"exchange rate", priority=5),
            _rule("card", "card_arrival", r"card", priority=5),
            _rule("noise", "exchange_rate", r".", priority=10),
        ]
        dataset = _dataset()
        apply_fn = _apply_rules_fn()
        report = rank_rules(rules, dataset, apply_fn, compute_marginal=True)
        kept, dropped = prune_harmful_rules(rules, report)
        assert [r.id for r in dropped] == ["noise"]
        assert {r.id for r in kept} == {"exchange", "card"}

    def test_keeps_rules_without_ablation_data(self):
        rules = [_rule("a", "exchange_rate", r"exchange")]
        report = rank_rules(rules, _dataset(), _apply_rules_fn(), compute_marginal=False)
        kept, dropped = prune_harmful_rules(rules, report)
        assert len(kept) == 1
        assert not dropped
