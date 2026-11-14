"""Tests for conflict resolution: which rule wins when several match."""

from unittest.mock import MagicMock

import pytest

from rulechef import RuleChef
from rulechef.core import Rule, RuleFormat, Task, TaskType
from rulechef.executor import RuleExecutor


def _rule(rule_id, label, pattern="text", priority=5, validated_precision=None, confidence=0.5):
    return Rule(
        id=rule_id,
        name=rule_id,
        description="",
        format=RuleFormat.REGEX,
        content=pattern,
        priority=priority,
        confidence=confidence,
        validated_precision=validated_precision,
        output_template={"label": label},
        output_key="label",
    )


class TestClassificationConflicts:
    def setup_method(self):
        self.executor = RuleExecutor()

    def test_higher_priority_wins(self):
        rules = [
            _rule("low", "label_low", priority=3),
            _rule("high", "label_high", priority=8),
        ]
        output = self.executor.apply_rules(rules, {"text": "some text"}, TaskType.CLASSIFICATION)
        assert output["label"] == "label_high"

    def test_validated_precision_breaks_priority_ties(self):
        rules = [
            _rule("imprecise", "label_a", priority=5, validated_precision=0.4),
            _rule("precise", "label_b", priority=5, validated_precision=0.95),
        ]
        output = self.executor.apply_rules(rules, {"text": "some text"}, TaskType.CLASSIFICATION)
        assert output["label"] == "label_b"

    def test_confidence_used_when_no_validated_precision(self):
        rules = [
            _rule("low_conf", "label_a", priority=5, confidence=0.3),
            _rule("high_conf", "label_b", priority=5, confidence=0.9),
        ]
        output = self.executor.apply_rules(rules, {"text": "some text"}, TaskType.CLASSIFICATION)
        assert output["label"] == "label_b"

    def test_winner_attribution_recorded(self):
        rules = [_rule("winner", "label_a", priority=5, validated_precision=0.9)]
        output = self.executor.apply_rules(rules, {"text": "some text"}, TaskType.CLASSIFICATION)
        assert output["rule_id"] == "winner"
        assert output["rule_name"] == "winner"

    def test_deterministic_on_full_tie(self):
        rules = [
            _rule("b_rule", "label_b", priority=5),
            _rule("a_rule", "label_a", priority=5),
        ]
        outputs = {
            self.executor.apply_rules(rules, {"text": "some text"}, TaskType.CLASSIFICATION)[
                "label"
            ]
            for _ in range(5)
        }
        # Same input, same rules → same winner every time (name order)
        assert outputs == {"label_a"}

    def test_non_matching_high_precision_rule_does_not_block(self):
        rules = [
            _rule("no_match", "label_a", pattern="zzz_never", priority=9, validated_precision=1.0),
            _rule("match", "label_b", priority=5, validated_precision=0.8),
        ]
        output = self.executor.apply_rules(rules, {"text": "some text"}, TaskType.CLASSIFICATION)
        assert output["label"] == "label_b"


# =========================================================================
# Span overlaps (overlap_strategy / cross_type_overlaps)
# =========================================================================


def _span_rule(rule_id, pattern, type_, priority=5):
    return Rule(
        id=rule_id,
        name=rule_id,
        description="",
        format=RuleFormat.REGEX,
        content=pattern,
        priority=priority,
        output_template={"text": "$0", "start": "$start", "end": "$end", "type": type_},
        output_key="entities",
    )


SPAN_TEXT = "Contact John Doe Consulting LLC today"
ORG = _span_rule("org", r"John Doe Consulting LLC", "ORG", priority=1)
ORG_SHORT = _span_rule("org_short", r"John Doe", "ORG", priority=9)
PER = _span_rule("per", r"John Doe", "PER", priority=9)
ORG_NEAR = _span_rule("org_near", r"John Doe Consulting", "ORG", priority=9)
TIE_A = _span_rule("tie_a", r"John Doe Consu", "ORG", priority=9)
TIE_B = _span_rule("tie_b", r"Doe Consulting", "ORG", priority=1)
CHAIN_A = _span_rule("chain_a", r"Contact John Doe", "ORG")
CHAIN_B = _span_rule("chain_b", r"Doe Consulting", "ORG")
CHAIN_C = _span_rule("chain_c", r"Consulting LLC today", "ORG")


def _span_texts(rules, **kwargs):
    out = RuleExecutor(**kwargs).apply_rules(rules, {"text": SPAN_TEXT}, TaskType.NER, "text")
    return sorted(e["text"] for e in out.get("entities", []))


class TestSpanOverlapConflicts:
    @pytest.mark.parametrize(
        "rules, kwargs, expected",
        [
            # default: partial overlap (IoU 0.35 < 0.7) -> both kept
            ([ORG, ORG_SHORT], {}, ["John Doe", "John Doe Consulting LLC"]),
            # longest wins, even against higher priority
            ([ORG, ORG_SHORT], {"overlap_strategy": "longest"}, ["John Doe Consulting LLC"]),
            # IoU = 0.83 > 0.7, if strategy is longest then longest wins
            ([ORG, ORG_NEAR], {"overlap_strategy": "longest"}, ["John Doe Consulting LLC"]),
            # IoU = 0.83 > 0.7, without a strategy the higher priority wins
            ([ORG, ORG_NEAR], {}, ["John Doe Consulting"]),
            # equal length (14 chars): higher-priority rule wins
            ([TIE_A, TIE_B], {"overlap_strategy": "longest"}, ["John Doe Consu"]),
            (
                [CHAIN_A, CHAIN_B, CHAIN_C],
                {"overlap_strategy": "longest"},
                ["Consulting LLC today", "Contact John Doe"],
            ),
            # different types don't conflict by default
            (
                [ORG, PER],
                {"overlap_strategy": "longest"},
                ["John Doe", "John Doe Consulting LLC"],
            ),
            # unless cross_type_overlaps=True
            (
                [ORG, PER],
                {"overlap_strategy": "longest", "cross_type_overlaps": True},
                ["John Doe Consulting LLC"],
            ),
        ],
    )
    def test_overlap_strategy(self, rules, kwargs, expected):
        assert _span_texts(rules, **kwargs) == expected

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            RuleExecutor(overlap_strategy="shortest")

    def test_invalid_strategy_raises_in_rulechef_without_task(self):
        with pytest.raises(ValueError):
            RuleChef(task=None, overlap_strategy="bogus")

    def test_rulechef_passes_overlap_options_to_executor(self, tmp_path):
        task = Task(
            name="ner",
            description="",
            input_schema={"text": "str"},
            output_schema={"entities": "list[dict]"},
        )
        chef = RuleChef(
            task=task,
            client=MagicMock(),
            storage_path=str(tmp_path),
            overlap_strategy="longest",
            cross_type_overlaps=True,
        )
        assert chef.learner.executor.overlap_strategy == "longest"
        assert chef.learner.executor.cross_type_overlaps is True
