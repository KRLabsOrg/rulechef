"""Tests for failure-mode clustering in the patch sampling path."""

from unittest.mock import MagicMock

from rulechef.core import RuleFormat, TaskType
from rulechef.learner import RuleLearner


def _learner():
    return RuleLearner(MagicMock(), allowed_formats=[RuleFormat.REGEX], model="test-model")


def _clf_failure(expected, got, is_correction=False):
    return {
        "input": {"text": f"{expected}-{got}"},
        "expected": {"label": expected},
        "got": {"label": got} if got else {},
        "is_correction": is_correction,
    }


class TestFailureSignature:
    def test_classification_no_prediction(self):
        learner = _learner()
        cls, kind = learner._failure_signature(
            _clf_failure("card_arrival", ""), TaskType.CLASSIFICATION
        )
        assert cls == "card_arrival"
        assert kind == "no_prediction"

    def test_classification_wrong_label(self):
        learner = _learner()
        cls, kind = learner._failure_signature(
            _clf_failure("card_arrival", "exchange_rate"), TaskType.CLASSIFICATION
        )
        assert cls == "card_arrival"
        assert kind == "predicted_as:exchange_rate"

    def test_ner_missed_entity(self):
        learner = _learner()
        failure = {
            "expected": {"entities": [{"text": "May", "type": "DATE"}]},
            "got": {"entities": []},
        }
        cls, kind = learner._failure_signature(failure, TaskType.NER)
        assert cls == "DATE"
        assert kind == "missed:DATE"

    def test_ner_spurious_entity(self):
        learner = _learner()
        failure = {
            "expected": {"entities": []},
            "got": {"entities": [{"text": "5", "type": "CARDINAL"}]},
        }
        cls, kind = learner._failure_signature(failure, TaskType.NER)
        assert kind == "spurious:CARDINAL"


class TestSampleFailures:
    def test_round_robin_covers_all_modes(self):
        learner = _learner()
        failures = (
            [_clf_failure("a", "") for _ in range(50)]
            + [_clf_failure("b", "a") for _ in range(50)]
            + [_clf_failure("c", "") for _ in range(2)]
        )
        sampled = learner._sample_failures(
            failures, max_samples=12, task_type=TaskType.CLASSIFICATION
        )
        sampled_classes = {f["expected"]["label"] for f in sampled}
        # The rare mode (class c) must be represented, not drowned out
        assert sampled_classes == {"a", "b", "c"}

    def test_corrections_always_included(self):
        learner = _learner()
        failures = [_clf_failure("a", "") for _ in range(30)] + [
            _clf_failure("z", "", is_correction=True)
        ]
        sampled = learner._sample_failures(
            failures, max_samples=5, task_type=TaskType.CLASSIFICATION
        )
        assert any(f.get("is_correction") for f in sampled)

    def test_respects_max_samples(self):
        learner = _learner()
        failures = [_clf_failure("a", "") for _ in range(100)]
        sampled = learner._sample_failures(
            failures, max_samples=10, task_type=TaskType.CLASSIFICATION
        )
        assert len(sampled) == 10


class TestFailureModeSummary:
    def test_summary_counts_all_failures(self):
        learner = _learner()
        failures = [_clf_failure("a", "") for _ in range(120)] + [
            _clf_failure("b", "a") for _ in range(40)
        ]
        summary = learner._failure_mode_summary(failures, TaskType.CLASSIFICATION)
        assert "all 160 failures" in summary
        assert "a — no_prediction: 120" in summary
        assert "b — predicted_as:a: 40" in summary

    def test_empty_failures(self):
        learner = _learner()
        assert learner._failure_mode_summary([], TaskType.CLASSIFICATION) == ""


class TestCatchAllDetection:
    def test_dot_star_is_catch_all(self):
        learner = _learner()
        assert learner._is_catch_all_regex(r".*")
        assert learner._is_catch_all_regex(r"(?i)^[\s\S]{0,1000}$")
        assert learner._is_catch_all_regex(r".+|^$")
        assert learner._is_catch_all_regex(r"(?s)^.{1,}$")  # matches any non-empty input

    def test_specific_pattern_not_catch_all(self):
        learner = _learner()
        assert not learner._is_catch_all_regex(r"\bkill\b")
        assert not learner._is_catch_all_regex(r"(?i)how (do|can) i make a bomb")

    def test_pure_negative_lookahead_is_catch_all(self):
        # Matches everything except niche blocked terms = near-constant predictor
        learner = _learner()
        assert learner._is_catch_all_regex(r"(?i)^(?!.*\bkill\b).*$")

    def test_lookahead_with_positive_anchor_not_catch_all(self):
        # A required positive term means it only matches relevant inputs
        learner = _learner()
        assert not learner._is_catch_all_regex(r"(?i)(?!.*\bkill\b).*\bweather\b")

    def test_catch_all_rule_rejected_by_validate(self):
        from rulechef.core import Rule

        learner = _learner()
        rule = Rule(
            id="x",
            name="fallback",
            description="",
            format=RuleFormat.REGEX,
            content=r"(?i)^[\s\S]{0,1000}$",
            output_template={"label": "safe"},
            output_key="label",
        )
        assert not learner._validate_rule(rule)


class TestTruncatedJSONRecovery:
    def test_recovers_complete_objects(self):
        learner = _learner()
        truncated = (
            '{"analysis": "x", "rules": ['
            '{"name": "r1", "format": "regex", "content": "abc"},'
            '{"name": "r2", "format": "regex", "content": "def"},'
            '{"name": "r3", "format": "regex", "content": "ghi'  # cut off mid-string
        )
        recovered = learner._recover_truncated_rules(truncated)
        assert recovered is not None
        assert [r["name"] for r in recovered["rules"]] == ["r1", "r2"]

    def test_handles_braces_inside_strings(self):
        learner = _learner()
        truncated = (
            '{"rules": [{"name": "r1", "content": "a{0,5}b", "format": "regex"},'
            '{"name": "r2", "content": "incomplete'
        )
        recovered = learner._recover_truncated_rules(truncated)
        assert [r["name"] for r in recovered["rules"]] == ["r1"]
        assert recovered["rules"][0]["content"] == "a{0,5}b"

    def test_returns_none_without_rules_array(self):
        learner = _learner()
        assert learner._recover_truncated_rules('{"analysis": "no rules here"') is None

    def test_parse_json_falls_back_to_recovery(self):
        learner = _learner()
        truncated = (
            '{"rules": [{"name": "r1", "format": "regex", "content": "abc"},'
            '{"name": "r2", "format": "regex", "content": "trunc'
        )
        result = learner._parse_json(truncated)
        assert len(result["rules"]) == 1
