"""Tests for rulechef.executor — rule execution logic."""

from rulechef.core import Rule, RuleFormat, Span, TaskType
from rulechef.executor import RuleExecutor, substitute_template

# =========================================================================
# substitute_template
# =========================================================================


class TestSubstituteTemplate:
    def test_full_match_variable(self):
        tpl = {"text": "$0"}
        result = substitute_template(tpl, "hello", 5, 10)
        assert result == {"text": "hello"}

    def test_capture_groups(self):
        tpl = {"first": "$1", "second": "$2"}
        result = substitute_template(tpl, "John Doe", 0, 8, groups=("John", "Doe"))
        assert result == {"first": "John", "second": "Doe"}

    def test_start_end_offsets(self):
        tpl = {"start": "$start", "end": "$end"}
        result = substitute_template(tpl, "word", 10, 14)
        assert result == {"start": 10, "end": 14}

    def test_nested_dict_recursion(self):
        tpl = {"outer": {"text": "$0", "pos": "$start"}}
        result = substitute_template(tpl, "abc", 3, 6)
        assert result == {"outer": {"text": "abc", "pos": 3}}

    def test_inline_substitution_in_strings(self):
        tpl = {"label": "found: $0 at $start"}
        result = substitute_template(tpl, "hello", 5, 10)
        assert result == {"label": "found: hello at 5"}

    def test_literal_passthrough(self):
        tpl = {"type": "DRUG", "score": 0.9, "active": True}
        result = substitute_template(tpl, "aspirin", 0, 7)
        assert result == {"type": "DRUG", "score": 0.9, "active": True}

    def test_missing_capture_group_returns_empty(self):
        tpl = {"val": "$3"}
        result = substitute_template(tpl, "x", 0, 1, groups=("a",))
        assert result == {"val": ""}

    def test_group_zero_via_dollar_zero(self):
        tpl = {"match": "$0"}
        result = substitute_template(tpl, "full_match", 0, 10, groups=())
        assert result == {"match": "full_match"}


# =========================================================================
# RuleExecutor._execute_regex_rule
# =========================================================================


class TestExecuteRegexRule:
    def setup_method(self):
        self.executor = RuleExecutor()

    def test_basic_span_output_no_template(self):
        rule = Rule(
            id="r1",
            name="digits",
            description="Find digits",
            format=RuleFormat.REGEX,
            content=r"\d+",
            confidence=0.8,
        )
        results = self.executor._execute_regex_rule(rule, {"text": "abc 123 def 456"})
        assert len(results) == 2
        assert isinstance(results[0], Span)
        assert results[0].text == "123"
        assert results[0].start == 4
        assert results[0].end == 7
        assert results[0].score == 0.8
        assert results[1].text == "456"

    def test_with_output_template(self):
        rule = Rule(
            id="r2",
            name="entities",
            description="Find entities",
            format=RuleFormat.REGEX,
            content=r"\b[A-Z][a-z]+\b",
            output_template={
                "text": "$0",
                "start": "$start",
                "end": "$end",
                "type": "NAME",
            },
        )
        results = self.executor._execute_regex_rule(rule, {"text": "Hello World"})
        assert len(results) == 2
        assert results[0] == {"text": "Hello", "start": 0, "end": 5, "type": "NAME"}
        assert results[1] == {"text": "World", "start": 6, "end": 11, "type": "NAME"}

    def test_capture_groups_with_template(self):
        rule = Rule(
            id="r3",
            name="dosage",
            description="Match dosage",
            format=RuleFormat.REGEX,
            content=r"(\d+)\s*mg\s+(\w+)",
            output_template={"dose": "$1", "drug": "$2", "full": "$0"},
        )
        results = self.executor._execute_regex_rule(rule, {"text": "Take 100 mg Aspirin daily"})
        assert len(results) == 1
        assert results[0]["dose"] == "100"
        assert results[0]["drug"] == "Aspirin"
        assert results[0]["full"] == "100 mg Aspirin"

    def test_no_match_returns_empty(self):
        rule = Rule(
            id="r4",
            name="nope",
            description="No match",
            format=RuleFormat.REGEX,
            content=r"ZZZZZ",
        )
        results = self.executor._execute_regex_rule(rule, {"text": "nothing here"})
        assert results == []


# =========================================================================
# RuleExecutor._execute_code_rule
# =========================================================================


class TestExecuteCodeRule:
    def setup_method(self):
        self.executor = RuleExecutor()

    def test_working_extract_function(self):
        code = """
def extract(input_data):
    text = input_data.get("text", "")
    return [{"text": w, "start": text.index(w), "end": text.index(w) + len(w)}
            for w in text.split() if w.startswith("#")]
"""
        rule = Rule(
            id="r1",
            name="hashtag",
            description="Find hashtags",
            format=RuleFormat.CODE,
            content=code,
        )
        results = self.executor._execute_code_rule(rule, {"text": "Hello #world and #python"})
        assert len(results) == 2
        assert results[0]["text"] == "#world"
        assert results[1]["text"] == "#python"

    def test_broken_code_returns_none(self):
        code = """
def extract(input_data):
    raise ValueError("boom")
"""
        rule = Rule(
            id="r2",
            name="broken",
            description="Broken rule",
            format=RuleFormat.CODE,
            content=code,
        )
        result = self.executor._execute_code_rule(rule, {"text": "anything"})
        assert result is None

    def test_code_with_syntax_error_returns_none(self):
        code = "def extract(input_data):\n    return ]["
        rule = Rule(
            id="r3",
            name="syntax_err",
            description="Syntax error",
            format=RuleFormat.CODE,
            content=code,
        )
        result = self.executor._execute_code_rule(rule, {"text": "x"})
        assert result is None

    def test_code_without_extract_function(self):
        code = "x = 42"
        rule = Rule(
            id="r4",
            name="no_func",
            description="No extract func",
            format=RuleFormat.CODE,
            content=code,
        )
        result = self.executor._execute_code_rule(rule, {"text": "x"})
        assert result is None


# =========================================================================
# RuleExecutor.apply_rules
# =========================================================================


class TestApplyRules:
    def setup_method(self):
        self.executor = RuleExecutor()

    def test_extraction_task_type(self):
        rule = Rule(
            id="r1",
            name="digits",
            description="Find digits",
            format=RuleFormat.REGEX,
            content=r"\d+",
        )
        output = self.executor.apply_rules(
            [rule], {"text": "abc 42 def"}, task_type=TaskType.EXTRACTION
        )
        assert "spans" in output
        assert len(output["spans"]) == 1
        assert output["spans"][0]["text"] == "42"

    def test_classification_task_type(self):
        code = """
def extract(input_data):
    text = input_data.get("text", "")
    if "happy" in text.lower():
        return {"label": "positive"}
    return {"label": "negative"}
"""
        rule = Rule(
            id="r1",
            name="sentiment",
            description="Classify sentiment",
            format=RuleFormat.CODE,
            content=code,
        )
        output = self.executor.apply_rules(
            [rule], {"text": "I am happy today"}, task_type=TaskType.CLASSIFICATION
        )
        assert output.get("label") == "positive"

    def test_ner_with_output_template(self):
        rule = Rule(
            id="r1",
            name="ner_caps",
            description="Capitalized words as entities",
            format=RuleFormat.REGEX,
            content=r"\b[A-Z][a-z]{2,}\b",
            output_template={
                "text": "$0",
                "start": "$start",
                "end": "$end",
                "type": "NAME",
            },
            output_key="entities",
        )
        output = self.executor.apply_rules(
            [rule], {"text": "Alice met Bob"}, task_type=TaskType.NER
        )
        assert "entities" in output
        texts = [e["text"] for e in output["entities"]]
        assert "Alice" in texts
        assert "Bob" in texts

    def test_deduplication_overlapping_spans(self):
        """Two rules that match the same position should be deduped."""
        rule1 = Rule(
            id="r1",
            name="find_hello",
            description="Match hello",
            format=RuleFormat.REGEX,
            content=r"hello",
            priority=10,
        )
        rule2 = Rule(
            id="r2",
            name="find_hello_too",
            description="Also match hello",
            format=RuleFormat.REGEX,
            content=r"hello",
            priority=5,
        )
        output = self.executor.apply_rules(
            [rule1, rule2], {"text": "say hello world"}, task_type=TaskType.EXTRACTION
        )
        # Same start/end position should be deduped
        assert len(output["spans"]) == 1

    def test_priority_ordering(self):
        """Higher priority rule's results come first."""
        rule_low = Rule(
            id="r_low",
            name="a_low",
            description="Low priority",
            format=RuleFormat.REGEX,
            content=r"\d+",
            priority=1,
        )
        rule_high = Rule(
            id="r_high",
            name="b_high",
            description="High priority",
            format=RuleFormat.REGEX,
            content=r"[A-Z]+",
            priority=10,
        )
        output = self.executor.apply_rules(
            [rule_low, rule_high], {"text": "ABC 123"}, task_type=TaskType.EXTRACTION
        )
        assert len(output["spans"]) == 2
        # High priority rule processed first
        assert output["spans"][0]["text"] == "ABC"
        assert output["spans"][1]["text"] == "123"

    def test_empty_rules_returns_empty(self):
        output = self.executor.apply_rules([], {"text": "anything"}, task_type=TaskType.EXTRACTION)
        assert output == {}

    def test_classification_first_label_wins(self):
        """For classification, only the first matching label is kept."""
        rule1 = Rule(
            id="r1",
            name="positive",
            description="Pos",
            format=RuleFormat.CODE,
            priority=10,
            content='def extract(input_data):\n    return {"label": "positive"}',
        )
        rule2 = Rule(
            id="r2",
            name="negative",
            description="Neg",
            format=RuleFormat.CODE,
            priority=5,
            content='def extract(input_data):\n    return {"label": "negative"}',
        )
        output = self.executor.apply_rules(
            [rule1, rule2], {"text": "hello"}, task_type=TaskType.CLASSIFICATION
        )
        # Higher-priority rule processed first, so "positive" wins
        assert output["label"] == "positive"


# =========================================================================
# RuleExecutor._select_text
# =========================================================================


class TestSelectText:
    def setup_method(self):
        self.executor = RuleExecutor()

    def test_explicit_text_field(self):
        data = {"body": "short", "content": "this is much longer text"}
        result = self.executor._select_text(data, text_field="body")
        assert result == "short"

    def test_no_text_field_picks_longest(self):
        data = {"a": "hi", "b": "hello world this is longer"}
        result = self.executor._select_text(data, text_field=None)
        assert result == "hello world this is longer"

    def test_empty_dict(self):
        result = self.executor._select_text({}, text_field=None)
        assert result == ""

    def test_no_string_fields(self):
        result = self.executor._select_text({"num": 42, "flag": True}, text_field=None)
        assert result == ""

    def test_text_field_not_found_falls_back(self):
        data = {"text": "hello world"}
        result = self.executor._select_text(data, text_field="missing_field")
        # Falls back to longest string
        assert result == "hello world"


# =========================================================================
# RuleExecutor._normalize_label
# =========================================================================


class TestNormalizeLabel:
    def setup_method(self):
        self.executor = RuleExecutor()

    def test_string_input(self):
        assert self.executor._normalize_label("positive") == "positive"

    def test_string_strips_whitespace(self):
        assert self.executor._normalize_label("  negative  ") == "negative"

    def test_empty_string_returns_none(self):
        assert self.executor._normalize_label("   ") is None

    def test_dict_with_label_key(self):
        assert self.executor._normalize_label({"label": "spam"}) == "spam"

    def test_dict_with_type_key(self):
        assert self.executor._normalize_label({"type": "urgent"}) == "urgent"

    def test_dict_with_text_key(self):
        assert self.executor._normalize_label({"text": "hello"}) == "hello"

    def test_list_of_dicts(self):
        result = self.executor._normalize_label([{"label": "first"}, {"label": "second"}])
        assert result == "first"

    def test_span_object(self):
        s = Span(text="positive", start=0, end=8)
        assert self.executor._normalize_label(s) == "positive"

    def test_none_returns_none(self):
        assert self.executor._normalize_label(None) is None

    def test_empty_list_returns_none(self):
        assert self.executor._normalize_label([]) is None


# =========================================================================
# RuleExecutor._deduplicate_dicts
# =========================================================================


class TestDeduplicateDicts:
    def setup_method(self):
        self.executor = RuleExecutor()

    def test_identical_dicts_deduplicated(self):
        items = [
            {"text": "hello", "start": 0, "end": 5, "type": "NAME"},
            {"text": "hello", "start": 0, "end": 5, "type": "NAME"},
        ]
        result = self.executor._deduplicate_dicts(items)
        assert len(result) == 1

    def test_overlapping_spans_same_type_deduped(self):
        """IoU > 0.7 with same type should be deduped."""
        items = [
            {"text": "hello world", "start": 0, "end": 11, "type": "NAME"},
            {"text": "hello worl", "start": 0, "end": 10, "type": "NAME"},
        ]
        # overlap = 10, union = 11, IoU = 10/11 ≈ 0.909 > 0.7
        result = self.executor._deduplicate_dicts(items)
        assert len(result) == 1

    def test_different_types_kept(self):
        """Same span but different types should NOT be deduped."""
        items = [
            {"text": "Apple", "start": 0, "end": 5, "type": "ORG"},
            {"text": "Apple", "start": 0, "end": 5, "type": "FOOD"},
        ]
        result = self.executor._deduplicate_dicts(items)
        assert len(result) == 2

    def test_non_span_dicts_exact_dedup(self):
        """Dicts without start/end should use exact dedup."""
        items = [
            {"label": "positive"},
            {"label": "positive"},
            {"label": "negative"},
        ]
        result = self.executor._deduplicate_dicts(items)
        assert len(result) == 2

    def test_empty_list(self):
        assert self.executor._deduplicate_dicts([]) == []

    def test_non_overlapping_spans_kept(self):
        items = [
            {"text": "hello", "start": 0, "end": 5, "type": "NAME"},
            {"text": "world", "start": 100, "end": 105, "type": "NAME"},
        ]
        result = self.executor._deduplicate_dicts(items)
        assert len(result) == 2
