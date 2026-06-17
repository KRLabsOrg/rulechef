"""Tests for Rule.from_dict and RuleChef.load_rules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rulechef import RuleChef
from rulechef.core import Rule, RuleFormat, Task, TaskType


def _ner_task():
    return Task(
        name="tab",
        description="extract entities",
        input_schema={"text": "str"},
        output_schema={"entities": "List[{text,start,end,type}]"},
        type=TaskType.NER,
        text_field="text",
    )


def _chef():
    # MagicMock client avoids constructing a real OpenAI client (no API key needed).
    return RuleChef(
        task=_ner_task(),
        client=MagicMock(),
        dataset_name="t",
        storage_path=tempfile.mkdtemp(),
    )


class TestRuleFromDict:
    def test_full_roundtrip(self):
        original = Rule(
            id="abc123",
            name="single_date",
            description="dates",
            format=RuleFormat.REGEX,
            content=r"\d{4}",
            priority=6,
            validated_precision=0.95,
            validated_support=362,
            output_template={"text": "$0", "type": "DATETIME"},
            output_key="entities",
        )
        r = Rule.from_dict(original.to_dict())
        assert r.id == "abc123"
        assert r.name == "single_date"
        assert r.description == "dates"
        assert r.format == RuleFormat.REGEX
        assert r.content == r"\d{4}"
        assert r.priority == 6
        assert r.validated_precision == 0.95
        assert r.validated_support == 362
        assert r.output_template == {"text": "$0", "type": "DATETIME"}
        assert r.output_key == "entities"

    def test_trimmed_dict_without_id_or_description(self):
        # The shape the benchmark checkpoints store (no id/description/confidence).
        d = {
            "name": "case_and_echr_numbers",
            "format": "regex",
            "content": r"\d{4,6}/\d{2,4}",
            "priority": 5,
            "validated_precision": 0.86,
            "validated_support": 22,
            "output_template": {"text": "$1", "type": "CODE"},
            "output_key": "entities",
        }
        r = Rule.from_dict(d)
        assert r.id  # auto-generated
        assert r.description == ""
        assert r.name == "case_and_echr_numbers"
        assert r.format == RuleFormat.REGEX
        assert r.validated_precision == 0.86
        assert r.validated_support == 22

    def test_minimal_dict_defaults(self):
        r = Rule.from_dict({"name": "x", "content": "a"})
        assert r.format == RuleFormat.REGEX
        assert r.priority == 5
        assert r.confidence == 0.5
        assert r.validated_precision is None
        assert r.validated_support == 0


class TestLoadRules:
    def test_load_from_list(self):
        chef = _chef()
        rules = chef.load_rules([{"name": "a", "content": r"\d+"}])
        assert len(rules) == 1
        assert chef.dataset.rules[0].name == "a"

    def test_load_from_checkpoint_shape(self):
        chef = _chef()
        data = {"fingerprint": {}, "result": {"rules": [{"name": "a", "content": "x"}]}}
        rules = chef.load_rules(data)
        assert len(rules) == 1
        assert chef.dataset.rules[0].name == "a"

    def test_load_from_dataset_shape(self):
        chef = _chef()
        rules = chef.load_rules(
            {"rules": [{"name": "a", "content": "x"}, {"name": "b", "content": "y"}]}
        )
        assert len(rules) == 2

    def test_load_from_comparison_meta_shape(self):
        # The shape of results_extract_tab.json: rules under meta.rulechef.rules
        chef = _chef()
        data = {"meta": {"rulechef": {"rules": [{"name": "a", "content": "x"}]}, "gliner2": {}}}
        rules = chef.load_rules(data)
        assert len(rules) == 1

    def test_load_from_file(self):
        chef = _chef()
        path = Path(tempfile.mkdtemp()) / "rules.json"
        path.write_text(json.dumps({"result": {"rules": [{"name": "a", "content": "x"}]}}))
        rules = chef.load_rules(str(path))
        assert len(rules) == 1

    def test_loaded_rules_are_executable(self):
        chef = _chef()
        chef.load_rules(
            [
                {
                    "name": "year",
                    "format": "regex",
                    "content": r"\b\d{4}\b",
                    "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "DATETIME"},
                    "output_key": "entities",
                }
            ]
        )
        out = chef.extract({"text": "filed in 2006 before the court"})
        assert any(e.get("text") == "2006" for e in out.get("entities", []))

    def test_bad_source_raises(self):
        chef = _chef()
        with pytest.raises(ValueError):
            chef.load_rules({"no_rules_here": 1})
