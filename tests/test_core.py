"""Tests for rulechef.core data structures."""

from typing import Literal

import pytest
from pydantic import BaseModel

from rulechef.core import (
    Correction,
    Dataset,
    Example,
    Feedback,
    Span,
    Task,
    TaskType,
    get_labels_from_model,
)

# =========================================================================
# Span
# =========================================================================


class TestSpan:
    def test_overlaps_true(self):
        a = Span(text="foo", start=0, end=5)
        b = Span(text="bar", start=3, end=8)
        assert a.overlaps(b) is True

    def test_overlaps_false_adjacent(self):
        a = Span(text="foo", start=0, end=5)
        b = Span(text="bar", start=5, end=10)
        assert a.overlaps(b) is False

    def test_overlaps_false_disjoint(self):
        a = Span(text="foo", start=0, end=3)
        b = Span(text="bar", start=10, end=15)
        assert a.overlaps(b) is False

    def test_overlap_ratio_full_overlap(self):
        a = Span(text="hello", start=0, end=5)
        b = Span(text="hello", start=0, end=5)
        assert a.overlap_ratio(b) == pytest.approx(1.0)

    def test_overlap_ratio_partial(self):
        a = Span(text="abc", start=0, end=6)
        b = Span(text="def", start=4, end=10)
        # overlap = 6-4 = 2, union = 10-0 = 10, ratio = 0.2
        assert a.overlap_ratio(b) == pytest.approx(0.2)

    def test_overlap_ratio_no_overlap(self):
        a = Span(text="x", start=0, end=5)
        b = Span(text="y", start=10, end=15)
        assert a.overlap_ratio(b) == pytest.approx(0.0)

    def test_to_dict(self):
        s = Span(text="hello", start=0, end=5, score=0.9)
        d = s.to_dict()
        assert d == {"text": "hello", "start": 0, "end": 5, "score": 0.9}

    def test_default_score(self):
        s = Span(text="x", start=0, end=1)
        assert s.score == 1.0


# =========================================================================
# Task
# =========================================================================


class TestTask:
    def test_get_labels_dict_schema_returns_empty(self, extraction_task):
        """Dict schemas have no Literal types, so get_labels returns []."""
        assert extraction_task.get_labels() == []

    def test_get_labels_pydantic_schema(self, ner_task):
        """Pydantic schema with Literal['PERSON','ORG','DRUG'] extracts labels."""
        labels = ner_task.get_labels()
        assert sorted(labels) == ["DRUG", "ORG", "PERSON"]

    def test_validate_output_dict_schema_always_valid(self, extraction_task):
        is_valid, errors = extraction_task.validate_output({"anything": True})
        assert is_valid is True
        assert errors == []

    def test_validate_output_pydantic_valid(self, ner_task):
        valid_output = {"entities": [{"text": "Aspirin", "start": 0, "end": 7, "type": "DRUG"}]}
        is_valid, errors = ner_task.validate_output(valid_output)
        assert is_valid is True
        assert errors == []

    def test_validate_output_pydantic_invalid(self, ner_task):
        bad_output = {"entities": [{"text": "Aspirin", "start": 0}]}
        is_valid, errors = ner_task.validate_output(bad_output)
        assert is_valid is False
        assert len(errors) > 0

    def test_get_schema_for_prompt_dict(self, extraction_task):
        prompt_str = extraction_task.get_schema_for_prompt()
        assert "spans" in prompt_str

    def test_get_schema_for_prompt_pydantic(self, ner_task):
        prompt_str = ner_task.get_schema_for_prompt()
        # Should mention 'entities' somewhere in the formatted output
        assert "entities" in prompt_str.lower() or "Entity" in prompt_str

    def test_to_dict(self, extraction_task):
        d = extraction_task.to_dict()
        assert d["name"] == "extract_emails"
        assert d["type"] == "extraction"
        assert d["matching_mode"] == "text"
        assert d["text_field"] is None
        assert "input_schema" in d
        assert "output_schema" in d

    def test_to_dict_pydantic_schema(self, ner_task):
        d = ner_task.to_dict()
        # Pydantic schema is converted to JSON schema dict
        assert isinstance(d["output_schema"], dict)
        assert d["type"] == "ner"

    def test_text_field_stored(self):
        t = Task(
            name="t",
            description="d",
            input_schema={"body": "str"},
            output_schema={"spans": "List[Span]"},
            type=TaskType.EXTRACTION,
            text_field="body",
        )
        assert t.text_field == "body"
        assert t.to_dict()["text_field"] == "body"


# =========================================================================
# Rule
# =========================================================================


class TestRule:
    def test_pattern_property_alias(self, sample_regex_rule):
        assert sample_regex_rule.pattern == sample_regex_rule.content

    def test_pattern_setter(self, sample_regex_rule):
        sample_regex_rule.pattern = r"\d+"
        assert sample_regex_rule.content == r"\d+"
        assert sample_regex_rule.pattern == r"\d+"

    def test_update_stats_success(self, sample_regex_rule):
        sample_regex_rule.update_stats(success=True)
        assert sample_regex_rule.times_applied == 1
        assert sample_regex_rule.successes == 1
        assert sample_regex_rule.failures == 0
        # confidence = 0.3 + (1/1 * 0.7) = 1.0
        assert sample_regex_rule.confidence == pytest.approx(1.0)

    def test_update_stats_failure(self, sample_regex_rule):
        sample_regex_rule.update_stats(success=False)
        assert sample_regex_rule.times_applied == 1
        assert sample_regex_rule.successes == 0
        assert sample_regex_rule.failures == 1
        # confidence = 0.3 + (0/1 * 0.7) = 0.3
        assert sample_regex_rule.confidence == pytest.approx(0.3)

    def test_update_stats_mixed(self, sample_regex_rule):
        sample_regex_rule.update_stats(success=True)
        sample_regex_rule.update_stats(success=True)
        sample_regex_rule.update_stats(success=False)
        assert sample_regex_rule.times_applied == 3
        assert sample_regex_rule.successes == 2
        assert sample_regex_rule.failures == 1
        # confidence = 0.3 + (2/3 * 0.7) ≈ 0.7667
        expected_conf = 0.3 + (2.0 / 3.0) * 0.7
        assert sample_regex_rule.confidence == pytest.approx(expected_conf)

    def test_to_dict_required_fields(self, sample_regex_rule):
        d = sample_regex_rule.to_dict()
        assert d["id"] == "rule-email-1"
        assert d["name"] == "email_pattern"
        assert d["format"] == "regex"
        assert "content" in d
        assert "created_at" in d
        # output_template and output_key should NOT be in dict when None
        assert "output_template" not in d
        assert "output_key" not in d

    def test_to_dict_with_optional_fields(self, sample_ner_regex_rule):
        d = sample_ner_regex_rule.to_dict()
        assert "output_template" in d
        assert d["output_template"]["type"] == "DRUG"
        assert d["output_key"] == "entities"


# =========================================================================
# Example / Correction / Feedback
# =========================================================================


class TestExampleCorrectionFeedback:
    def test_example_to_dict(self):
        ex = Example(
            id="e1",
            input={"text": "hello"},
            expected_output={"spans": []},
            source="human_labeled",
            confidence=0.9,
        )
        d = ex.to_dict()
        assert d["id"] == "e1"
        assert d["source"] == "human_labeled"
        assert d["confidence"] == 0.9
        assert d["input"] == {"text": "hello"}
        assert d["expected_output"] == {"spans": []}

    def test_correction_to_dict(self):
        c = Correction(
            id="c1",
            input={"text": "abc"},
            model_output={"spans": []},
            expected_output={"spans": [{"text": "abc"}]},
            feedback="missed",
        )
        d = c.to_dict()
        assert d["id"] == "c1"
        assert d["feedback"] == "missed"
        assert d["model_output"] == {"spans": []}
        assert d["expected_output"] == {"spans": [{"text": "abc"}]}

    def test_correction_to_dict_no_feedback(self):
        c = Correction(
            id="c2",
            input={"text": "x"},
            model_output={},
            expected_output={},
        )
        d = c.to_dict()
        assert d["feedback"] is None

    def test_feedback_to_dict(self):
        f = Feedback(
            id="f1",
            text="Too broad",
            level="rule",
            target_id="rule-1",
        )
        d = f.to_dict()
        assert d == {
            "id": "f1",
            "text": "Too broad",
            "level": "rule",
            "target_id": "rule-1",
        }

    def test_feedback_task_level_empty_target(self):
        f = Feedback(id="f2", text="general advice", level="task")
        assert f.target_id == ""
        assert f.to_dict()["target_id"] == ""


# =========================================================================
# Dataset
# =========================================================================


class TestDataset:
    def test_get_all_training_data_combines(self, sample_dataset):
        """Corrections come first, then examples."""
        all_data = sample_dataset.get_all_training_data()
        assert len(all_data) == 3
        # First item should be the correction
        assert isinstance(all_data[0], Correction)
        assert isinstance(all_data[1], Example)

    def test_get_feedback_for_level(self, extraction_task):
        fb_task = Feedback(id="f1", text="hint", level="task")
        fb_rule = Feedback(id="f2", text="too broad", level="rule", target_id="r1")
        fb_rule2 = Feedback(id="f3", text="too specific", level="rule", target_id="r2")

        ds = Dataset(
            name="ds",
            task=extraction_task,
            structured_feedback=[fb_task, fb_rule, fb_rule2],
        )

        assert len(ds.get_feedback_for("task")) == 1
        assert len(ds.get_feedback_for("rule")) == 2
        assert len(ds.get_feedback_for("rule", "r1")) == 1
        assert len(ds.get_feedback_for("example")) == 0

    def test_to_dict(self, sample_dataset):
        d = sample_dataset.to_dict()
        assert d["name"] == "email_dataset"
        assert len(d["examples"]) == 2
        assert len(d["corrections"]) == 1
        assert d["version"] == 1
        assert "task" in d

    def test_empty_dataset(self, extraction_task):
        ds = Dataset(name="empty", task=extraction_task)
        assert ds.get_all_training_data() == []
        assert ds.to_dict()["examples"] == []
        assert ds.to_dict()["corrections"] == []


# =========================================================================
# get_labels_from_model
# =========================================================================


class TestGetLabelsFromModel:
    def test_nested_pydantic_with_list_entity(self):
        """Model with List[Entity] where Entity has type: Literal[...]."""

        class Entity(BaseModel):
            text: str
            type: Literal["PERSON", "ORG", "LOC"]

        class Output(BaseModel):
            entities: list[Entity]

        labels = get_labels_from_model(Output)
        assert sorted(labels) == ["LOC", "ORG", "PERSON"]

    def test_flat_model_with_literal(self):
        class SingleEntity(BaseModel):
            text: str
            type: Literal["DRUG", "SYMPTOM"]

        labels = get_labels_from_model(SingleEntity)
        assert sorted(labels) == ["DRUG", "SYMPTOM"]

    def test_no_literal_field(self):
        class NoLiteral(BaseModel):
            text: str
            value: int

        assert get_labels_from_model(NoLiteral) == []

    def test_custom_field_name(self):
        class TaggedItem(BaseModel):
            text: str
            category: Literal["A", "B", "C"]

        labels = get_labels_from_model(TaggedItem, field_name="category")
        assert sorted(labels) == ["A", "B", "C"]

        # Asking for wrong field_name returns []
        assert get_labels_from_model(TaggedItem, field_name="type") == []
