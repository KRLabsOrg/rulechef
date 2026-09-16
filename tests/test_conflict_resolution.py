"""Tests for conflict resolution: which rule wins when several match."""

from rulechef.core import Rule, RuleFormat, TaskType
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
