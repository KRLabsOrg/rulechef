"""Tests for rulechef.evaluation — metrics and evaluation functions."""

import pytest

from rulechef.core import (
    Dataset,
    Example,
    Rule,
    RuleFormat,
    Task,
    TaskType,
)
from rulechef.evaluation import (
    ClassMetrics,
    _match_entities,
    evaluate_dataset,
    evaluate_rules_individually,
)
from rulechef.executor import RuleExecutor

# =========================================================================
# ClassMetrics
# =========================================================================


class TestClassMetrics:
    def test_precision_recall_f1(self):
        m = ClassMetrics(label="PERSON", tp=8, fp=2, fn=1)
        # precision = 8/10 = 0.8
        assert m.precision == pytest.approx(0.8)
        # recall = 8/9
        assert m.recall == pytest.approx(8 / 9)
        # f1 = 2 * 0.8 * (8/9) / (0.8 + 8/9)
        expected_f1 = 2 * 0.8 * (8 / 9) / (0.8 + 8 / 9)
        assert m.f1 == pytest.approx(expected_f1)

    def test_zero_division_returns_zero(self):
        m = ClassMetrics(label="EMPTY", tp=0, fp=0, fn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_all_false_positives(self):
        m = ClassMetrics(label="FP_ONLY", tp=0, fp=5, fn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_all_false_negatives(self):
        m = ClassMetrics(label="FN_ONLY", tp=0, fp=0, fn=5)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_perfect_metrics(self):
        m = ClassMetrics(label="PERFECT", tp=10, fp=0, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_to_dict(self):
        m = ClassMetrics(label="X", tp=5, fp=1, fn=2)
        d = m.to_dict()
        assert d["label"] == "X"
        assert d["tp"] == 5
        assert d["fp"] == 1
        assert d["fn"] == 2
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d


# =========================================================================
# _match_entities
# =========================================================================


class TestMatchEntities:
    def test_text_mode_matching(self):
        predicted = [
            {"text": "Alice", "start": 0, "end": 5, "type": "PERSON"},
            {"text": "Bob", "start": 10, "end": 13, "type": "PERSON"},
        ]
        expected = [
            {"text": "Alice", "start": 99, "end": 104, "type": "PERSON"},
            {"text": "Bob", "start": 200, "end": 203, "type": "PERSON"},
        ]
        matched, fps, fns = _match_entities(predicted, expected, TaskType.NER, mode="text")
        assert len(matched) == 2
        assert len(fps) == 0
        assert len(fns) == 0

    def test_exact_mode_matching(self):
        predicted = [
            {"text": "Alice", "start": 0, "end": 5, "type": "PERSON"},
        ]
        expected = [
            {"text": "Alice", "start": 0, "end": 5, "type": "PERSON"},
        ]
        matched, fps, fns = _match_entities(predicted, expected, TaskType.NER, mode="exact")
        assert len(matched) == 1
        assert len(fps) == 0
        assert len(fns) == 0

    def test_exact_mode_mismatch_position(self):
        """Same text but different position should NOT match in exact mode."""
        predicted = [
            {"text": "Alice", "start": 0, "end": 5, "type": "PERSON"},
        ]
        expected = [
            {"text": "Alice", "start": 99, "end": 104, "type": "PERSON"},
        ]
        matched, fps, fns = _match_entities(predicted, expected, TaskType.NER, mode="exact")
        assert len(matched) == 0
        assert len(fps) == 1
        assert len(fns) == 1

    def test_partial_matches(self):
        predicted = [
            {"text": "Alice", "start": 0, "end": 5, "type": "PERSON"},
            {"text": "Eve", "start": 20, "end": 23, "type": "PERSON"},
        ]
        expected = [
            {"text": "Alice", "start": 0, "end": 5, "type": "PERSON"},
            {"text": "Bob", "start": 10, "end": 13, "type": "PERSON"},
        ]
        matched, fps, fns = _match_entities(predicted, expected, TaskType.NER, mode="text")
        assert len(matched) == 1  # Alice
        assert len(fps) == 1  # Eve (FP)
        assert len(fns) == 1  # Bob (FN)

    def test_classification_label_matching(self):
        predicted = [{"label": "positive"}]
        expected = [{"label": "POSITIVE"}]
        matched, fps, fns = _match_entities(
            predicted, expected, TaskType.CLASSIFICATION, mode="text"
        )
        assert len(matched) == 1
        assert len(fps) == 0
        assert len(fns) == 0

    def test_classification_label_mismatch(self):
        predicted = [{"label": "positive"}]
        expected = [{"label": "negative"}]
        matched, fps, fns = _match_entities(
            predicted, expected, TaskType.CLASSIFICATION, mode="text"
        )
        assert len(matched) == 0
        assert len(fps) == 1
        assert len(fns) == 1


# =========================================================================
# evaluate_dataset
# =========================================================================
def _make_dataset(task_type, examples):
    """Helper to build a dataset with the given examples."""
    task = Task(
        name="test",
        description="test",
        input_schema={"text": "string"},
        output_schema={"spans": "List[Span]"},
        type=task_type,
    )
    ex_objs = [
        Example(
            id=f"ex-{i}",
            input=ex["input"],
            expected_output=ex["expected"],
            source="human_labeled",
        )
        for i, ex in enumerate(examples)
    ]
    return Dataset(name="test_ds", task=task, examples=ex_objs)


class TestEvaluateDataset:
    def test_perfect_score(self):
        """Mock apply_rules_fn returns exactly the expected output => F1 = 1.0."""
        ds = _make_dataset(
            TaskType.EXTRACTION,
            [
                {
                    "input": {"text": "hello 42"},
                    "expected": {"spans": [{"text": "42", "start": 6, "end": 8}]},
                },
                {
                    "input": {"text": "no numbers"},
                    "expected": {"spans": []},
                },
            ],
        )

        expected_outputs = {
            "hello 42": {"spans": [{"text": "42", "start": 6, "end": 8}]},
            "no numbers": {"spans": []},
        }

        def mock_apply(rules, input_data, task_type, text_field):
            return expected_outputs[input_data["text"]]

        result = evaluate_dataset([], ds, mock_apply)
        assert result.micro_f1 == pytest.approx(1.0)
        assert result.exact_match == pytest.approx(1.0)
        assert result.total_tp == 1
        assert result.total_fp == 0
        assert result.total_fn == 0
        assert result.total_docs == 2
        assert len(result.failures) == 0

    def test_no_matches(self):
        """apply_rules_fn returns empty => F1 = 0.0."""
        ds = _make_dataset(
            TaskType.EXTRACTION,
            [
                {
                    "input": {"text": "hello 42"},
                    "expected": {"spans": [{"text": "42", "start": 6, "end": 8}]},
                },
            ],
        )

        def mock_apply(rules, input_data, task_type, text_field):
            return {"spans": []}

        result = evaluate_dataset([], ds, mock_apply)
        assert result.micro_f1 == pytest.approx(0.0)
        assert result.micro_precision == pytest.approx(0.0)
        assert result.micro_recall == pytest.approx(0.0)
        assert result.total_tp == 0
        assert result.total_fn == 1

    def test_partial_matches(self):
        """Some matches, some misses => check exact TP/FP/FN counts."""
        ds = _make_dataset(
            TaskType.NER,
            [
                {
                    "input": {"text": "Alice met Bob"},
                    "expected": {
                        "entities": [
                            {"text": "Alice", "type": "PERSON"},
                            {"text": "Bob", "type": "PERSON"},
                        ]
                    },
                },
            ],
        )

        def mock_apply(rules, input_data, task_type, text_field):
            return {
                "entities": [
                    {"text": "Alice", "type": "PERSON"},  # TP
                    {"text": "Eve", "type": "PERSON"},  # FP
                ]
            }

        result = evaluate_dataset([], ds, mock_apply)
        assert result.total_tp == 1  # Alice
        assert result.total_fp == 1  # Eve
        assert result.total_fn == 1  # Bob
        # precision = 1/2 = 0.5, recall = 1/2 = 0.5, f1 = 0.5
        assert result.micro_precision == pytest.approx(0.5)
        assert result.micro_recall == pytest.approx(0.5)
        assert result.micro_f1 == pytest.approx(0.5)
        assert result.total_docs == 1
        assert len(result.failures) == 1

    def test_classification_evaluation(self):
        """Classification task evaluation via apply_rules_fn."""
        ds = _make_dataset(
            TaskType.CLASSIFICATION,
            [
                {"input": {"text": "great"}, "expected": {"label": "positive"}},
                {"input": {"text": "bad"}, "expected": {"label": "negative"}},
            ],
        )

        def mock_apply(rules, input_data, task_type, text_field):
            if "great" in input_data["text"]:
                return {"label": "positive"}
            return {"label": "positive"}  # Wrong for "bad"

        result = evaluate_dataset([], ds, mock_apply)
        assert result.total_tp == 1
        assert result.total_fp == 1
        assert result.total_fn == 1


# =========================================================================
# evaluate_rules_individually
# =========================================================================


class TestEvaluateRulesIndividually:
    def test_two_rules_separate_metrics(self):
        """Each rule is evaluated in isolation."""
        task = Task(
            name="ner",
            description="test ner",
            input_schema={"text": "string"},
            output_schema={"entities": "List[Entity]"},
            type=TaskType.NER,
        )
        ds = Dataset(
            name="test",
            task=task,
            examples=[
                Example(
                    id="ex-1",
                    input={"text": "Alice and Bob went to NYC"},
                    expected_output={
                        "entities": [
                            {"text": "Alice", "type": "PERSON"},
                            {"text": "Bob", "type": "PERSON"},
                            {"text": "NYC", "type": "LOC"},
                        ]
                    },
                    source="human_labeled",
                ),
            ],
        )

        rule_a = Rule(
            id="rA",
            name="persons",
            description="Find persons",
            format=RuleFormat.REGEX,
            content=r"dummy",
        )
        rule_b = Rule(
            id="rB",
            name="locations",
            description="Find locations",
            format=RuleFormat.REGEX,
            content=r"dummy",
        )

        def mock_apply(rules, input_data, task_type, text_field):
            """Returns different results depending on which rule is passed."""
            rule = rules[0]
            if rule.id == "rA":
                return {
                    "entities": [
                        {"text": "Alice", "type": "PERSON"},
                        {"text": "Bob", "type": "PERSON"},
                    ]
                }
            elif rule.id == "rB":
                return {
                    "entities": [
                        {"text": "NYC", "type": "LOC"},
                        {"text": "Paris", "type": "LOC"},  # FP
                    ]
                }
            return {"entities": []}

        results = evaluate_rules_individually([rule_a, rule_b], ds, mock_apply)

        assert len(results) == 2

        # Rule A: 2 TP (Alice, Bob), 0 FP, covers 2 of 3 expected
        ra = next(r for r in results if r.rule_id == "rA")
        assert ra.true_positives == 2
        assert ra.false_positives == 0
        assert ra.precision == pytest.approx(1.0)

        # Rule B: 1 TP (NYC), 1 FP (Paris), covers 1 of 3 expected
        rb = next(r for r in results if r.rule_id == "rB")
        assert rb.true_positives == 1
        assert rb.false_positives == 1
        assert rb.precision == pytest.approx(0.5)
        assert rb.covered_expected == 1
        assert rb.total_expected == 3

    def test_in_context_ner(self):
        ds = _make_dataset(
            TaskType.NER,
            [
                {
                    "input": {"text": "Contact John Doe Consulting LLC today"},
                    "expected": {"entities": [{"text": "John Doe Consulting LLC", "type": "ORG"}]},
                }
            ],
        )
        template = {"text": "$0", "start": "$start", "end": "$end", "type": "ORG"}
        rules = [
            Rule(
                id="rLong",
                name="long_org",
                description="",
                format=RuleFormat.REGEX,
                content=r"John Doe Consulting LLC",
                priority=1,
                output_template=template,
                output_key="entities",
            ),
            Rule(
                id="rNear",
                name="near_org",
                description="",
                format=RuleFormat.REGEX,
                content=r"John Doe Consulting",
                priority=9,
                output_template=template,
                output_key="entities",
            ),
        ]
        apply = RuleExecutor().apply_rules
        isolated = evaluate_rules_individually(rules, ds, apply)
        assert next(r for r in isolated if r.rule_id == "rLong").true_positives == 1
        in_ctx = evaluate_rules_individually(rules, ds, apply, in_context=True)
        assert next(r for r in in_ctx if r.rule_id == "rLong").true_positives == 0
        assert next(r for r in in_ctx if r.rule_id == "rNear").false_positives == 1

    def test_in_context_classification(self):
        ds = _make_dataset(
            TaskType.CLASSIFICATION,
            [{"input": {"text": "not great"}, "expected": {"label": "negative"}}],
        )
        rules = [
            # matches "great" -> positive, lower priority
            Rule(
                id="rPos",
                name="positive",
                description="",
                format=RuleFormat.REGEX,
                content="great",
                priority=3,
                output_template={"label": "positive"},
                output_key="label",
            ),
            # matches "not" -> negative, higher priority
            Rule(
                id="rNeg",
                name="negative",
                description="",
                format=RuleFormat.REGEX,
                content="not",
                priority=8,
                output_template={"label": "negative"},
                output_key="label",
            ),
        ]

        # Alone: rPos says "positive" -> wrong
        apply = RuleExecutor().apply_rules
        isolated = evaluate_rules_individually(rules, ds, apply)
        assert next(r for r in isolated if r.rule_id == "rPos").false_positives == 1

        # In context: rNeg wins (priority 8), the system says "negative"
        in_ctx = evaluate_rules_individually(rules, ds, apply, in_context=True)
        assert next(r for r in in_ctx if r.rule_id == "rNeg").true_positives == 1
        assert next(r for r in in_ctx if r.rule_id == "rPos").false_positives == 0
