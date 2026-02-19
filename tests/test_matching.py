"""Tests for rulechef.matching — output comparison logic."""

from rulechef.core import TaskType
from rulechef.matching import outputs_match


# =========================================================================
# EXTRACTION matching
# =========================================================================


class TestExtractionMatching:
    def test_text_match_same_texts_different_positions(self):
        """Text mode: same texts, different positions => match."""
        expected = {"spans": [{"text": "hello", "start": 0, "end": 5}]}
        actual = {"spans": [{"text": "hello", "start": 100, "end": 105}]}
        assert outputs_match(expected, actual, TaskType.EXTRACTION) is True

    def test_exact_match(self):
        expected = {"spans": [{"text": "hello", "start": 0, "end": 5}]}
        actual = {"spans": [{"text": "hello", "start": 0, "end": 5}]}
        assert (
            outputs_match(expected, actual, TaskType.EXTRACTION, matching_mode="exact")
            is True
        )

    def test_exact_mismatch_different_position(self):
        """Same text but different position => fails in exact mode."""
        expected = {"spans": [{"text": "hello", "start": 0, "end": 5}]}
        actual = {"spans": [{"text": "hello", "start": 10, "end": 15}]}
        assert (
            outputs_match(expected, actual, TaskType.EXTRACTION, matching_mode="exact")
            is False
        )

    def test_different_span_count(self):
        expected = {"spans": [{"text": "a"}, {"text": "b"}]}
        actual = {"spans": [{"text": "a"}]}
        assert outputs_match(expected, actual, TaskType.EXTRACTION) is False

    def test_order_independent(self):
        expected = {"spans": [{"text": "b"}, {"text": "a"}]}
        actual = {"spans": [{"text": "a"}, {"text": "b"}]}
        assert outputs_match(expected, actual, TaskType.EXTRACTION) is True

    def test_empty_spans_match(self):
        assert outputs_match({"spans": []}, {"spans": []}, TaskType.EXTRACTION) is True

    def test_text_mismatch(self):
        expected = {"spans": [{"text": "hello"}]}
        actual = {"spans": [{"text": "world"}]}
        assert outputs_match(expected, actual, TaskType.EXTRACTION) is False


# =========================================================================
# NER matching
# =========================================================================


class TestNERMatching:
    def test_same_entities_match(self):
        expected = {
            "entities": [
                {"text": "Alice", "type": "PERSON"},
                {"text": "NYC", "type": "LOC"},
            ]
        }
        actual = {
            "entities": [
                {"text": "Alice", "type": "PERSON"},
                {"text": "NYC", "type": "LOC"},
            ]
        }
        assert outputs_match(expected, actual, TaskType.NER) is True

    def test_type_mismatch_fails(self):
        expected = {"entities": [{"text": "Apple", "type": "ORG"}]}
        actual = {"entities": [{"text": "Apple", "type": "FOOD"}]}
        assert outputs_match(expected, actual, TaskType.NER) is False

    def test_legacy_spans_key(self):
        """Legacy 'spans' key should still work for NER."""
        expected = {"spans": [{"text": "Alice", "type": "PERSON"}]}
        actual = {"spans": [{"text": "Alice", "type": "PERSON"}]}
        assert outputs_match(expected, actual, TaskType.NER) is True

    def test_entities_key_vs_spans_key(self):
        """'entities' key in one dict, 'spans' in another."""
        expected = {"entities": [{"text": "Alice", "type": "PERSON"}]}
        actual = {"spans": [{"text": "Alice", "type": "PERSON"}]}
        assert outputs_match(expected, actual, TaskType.NER) is True

    def test_different_entity_count(self):
        expected = {"entities": [{"text": "Alice", "type": "PERSON"}]}
        actual = {
            "entities": [
                {"text": "Alice", "type": "PERSON"},
                {"text": "Bob", "type": "PERSON"},
            ]
        }
        assert outputs_match(expected, actual, TaskType.NER) is False

    def test_order_independent_ner(self):
        expected = {
            "entities": [
                {"text": "NYC", "type": "LOC"},
                {"text": "Alice", "type": "PERSON"},
            ]
        }
        actual = {
            "entities": [
                {"text": "Alice", "type": "PERSON"},
                {"text": "NYC", "type": "LOC"},
            ]
        }
        assert outputs_match(expected, actual, TaskType.NER) is True


# =========================================================================
# CLASSIFICATION matching
# =========================================================================


class TestClassificationMatching:
    def test_same_label(self):
        expected = {"label": "positive"}
        actual = {"label": "positive"}
        assert outputs_match(expected, actual, TaskType.CLASSIFICATION) is True

    def test_case_insensitive(self):
        expected = {"label": "POSITIVE"}
        actual = {"label": "positive"}
        assert outputs_match(expected, actual, TaskType.CLASSIFICATION) is True

    def test_mismatch(self):
        expected = {"label": "positive"}
        actual = {"label": "negative"}
        assert outputs_match(expected, actual, TaskType.CLASSIFICATION) is False

    def test_whitespace_stripped(self):
        expected = {"label": "  positive  "}
        actual = {"label": "positive"}
        assert outputs_match(expected, actual, TaskType.CLASSIFICATION) is True

    def test_empty_labels_match(self):
        expected = {"label": ""}
        actual = {"label": ""}
        assert outputs_match(expected, actual, TaskType.CLASSIFICATION) is True


# =========================================================================
# TRANSFORMATION matching
# =========================================================================


class TestTransformationMatching:
    def test_same_dict(self):
        expected = {"company": "Acme", "amount": "$100"}
        actual = {"company": "Acme", "amount": "$100"}
        assert outputs_match(expected, actual, TaskType.TRANSFORMATION) is True

    def test_array_order_independence(self):
        expected = {"items": [{"name": "b"}, {"name": "a"}]}
        actual = {"items": [{"name": "a"}, {"name": "b"}]}
        assert outputs_match(expected, actual, TaskType.TRANSFORMATION) is True

    def test_mismatch_value(self):
        expected = {"company": "Acme"}
        actual = {"company": "Beta"}
        assert outputs_match(expected, actual, TaskType.TRANSFORMATION) is False

    def test_missing_key(self):
        expected = {"company": "Acme", "amount": "$100"}
        actual = {"company": "Acme"}
        assert outputs_match(expected, actual, TaskType.TRANSFORMATION) is False

    def test_extra_key(self):
        expected = {"company": "Acme"}
        actual = {"company": "Acme", "extra": "value"}
        assert outputs_match(expected, actual, TaskType.TRANSFORMATION) is False

    def test_array_length_mismatch(self):
        expected = {"items": [{"a": 1}]}
        actual = {"items": [{"a": 1}, {"a": 2}]}
        assert outputs_match(expected, actual, TaskType.TRANSFORMATION) is False


# =========================================================================
# Custom output_matcher override
# =========================================================================


class TestCustomMatcher:
    def test_custom_matcher_overrides_default(self):
        """Custom matcher should be called instead of default logic."""

        def always_true(expected, actual):
            return True

        # These would normally not match
        expected = {"spans": [{"text": "A"}]}
        actual = {"spans": [{"text": "B"}]}
        assert (
            outputs_match(
                expected, actual, TaskType.EXTRACTION, custom_matcher=always_true
            )
            is True
        )

    def test_custom_matcher_returns_false(self):
        def always_false(expected, actual):
            return False

        expected = {"spans": [{"text": "A"}]}
        actual = {"spans": [{"text": "A"}]}
        assert (
            outputs_match(
                expected, actual, TaskType.EXTRACTION, custom_matcher=always_false
            )
            is False
        )

    def test_custom_matcher_receives_dicts(self):
        """Verify the custom matcher receives the actual arguments."""
        received = {}

        def spy_matcher(expected, actual):
            received["expected"] = expected
            received["actual"] = actual
            return True

        e = {"label": "A"}
        a = {"label": "B"}
        outputs_match(e, a, TaskType.CLASSIFICATION, custom_matcher=spy_matcher)
        assert received["expected"] is e
        assert received["actual"] is a
