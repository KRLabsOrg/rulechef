import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from rulechef import RuleChef
from rulechef.core import Rule, RuleFormat, Task, TaskType
from rulechef.learner import RuleLearner

live = pytest.mark.skipif("LIVE_LLM_BASE_URL" not in os.environ, reason="needs a live LLM endpoint")


def _learner(**kwargs):
    return RuleLearner(
        MagicMock(), allowed_formats=[RuleFormat.REGEX], model="test-model", **kwargs
    )


def test_same_input_gives_same_id():
    learner = _learner()
    assert learner._generate_id(
        name="a", content="b", format=RuleFormat.REGEX, output_template=None, output_key=None
    ) == learner._generate_id(
        name="a", content="b", format=RuleFormat.REGEX, output_template=None, output_key=None
    )


def test_id_same_across_learners():
    first = _learner()
    second = _learner()
    assert first._generate_id(
        name="a", content="b", format=RuleFormat.REGEX, output_template=None, output_key=None
    ) == second._generate_id(
        name="a", content="b", format=RuleFormat.REGEX, output_template=None, output_key=None
    )


def test_same_name_and_pattern_different_type_get_different_ids():
    learner = _learner()
    rule = {"name": "capitalized_pair", "format": "regex", "pattern": r"[A-Z]\w+ [A-Z]\w+"}
    result = {
        "rules": [
            {**rule, "output_template": {"text": "$0", "type": "PER"}},
            {**rule, "output_template": {"text": "$0", "type": "ORG"}},
        ]
    }
    rules = learner._parse_rules_from_response(result, max_rules=10)
    assert len(rules) == 2

    assert rules[0].id != rules[1].id
    remaining = [r for r in rules if r.id != rules[0].id]
    assert [r.output_template["type"] for r in remaining] == ["ORG"]


def test_identical_rules_get_suffix():
    learner = _learner()
    rule = {"name": "capitalized_pair", "format": "regex", "pattern": r"[A-Z]\w+ [A-Z]\w+"}
    result = {
        "rules": [
            {**rule, "output_template": {"text": "$0", "type": "PER"}},
            {**rule, "output_template": {"text": "$0", "type": "PER"}},
            {**rule, "output_template": {"text": "$0", "type": "PER"}},
        ]
    }
    rules = learner._parse_rules_from_response(result, max_rules=10)
    assert len(rules) == 3
    assert rules[1].id == f"{rules[0].id}_2"
    assert rules[2].id == f"{rules[0].id}_3"


def test_temp_kwargs_include_seed_and_temperature():
    learner = _learner(temperature=0, seed=42)
    assert learner._temp_kwargs() == {"temperature": 0, "seed": 42}


NER_DATA = Path(__file__).parent.parent / "examples" / "data" / "demo_ner_bulk.jsonl"


def _learn_with_live_llm(storage_path):
    task = Task(
        name="intent",
        description="Extract drugs, dosages, frequencies and conditions from text",
        input_schema={"text": "str"},
        output_schema={"entities": "list[dict]"},
        type=TaskType.NER,
    )

    chef = RuleChef(
        task=task,
        client=OpenAI(
            base_url=os.environ.get(
                "LIVE_LLM_BASE_URL",
            ),
            api_key=os.environ.get("LIVE_LLM_API_KEY", "EMPTY"),
        ),
        model=os.environ.get("LIVE_LLM_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        dataset_name="repro",
        storage_path=str(storage_path),
        allowed_formats=[RuleFormat.REGEX],
        temperature=0,
        seed=42,
    )
    for line in NER_DATA.read_text().splitlines():
        row = json.loads(line)
        chef.add_example(row["input"], row["output"])
    chef.learn_rules()
    return sorted((r.id, r.name, r.content) for r in chef.dataset.rules)


@pytest.mark.integration
@live
def test_real_llm_gives_same_rules(tmp_path):
    first = _learn_with_live_llm(tmp_path / "a")
    second = _learn_with_live_llm(tmp_path / "b")
    assert first
    assert first == second
