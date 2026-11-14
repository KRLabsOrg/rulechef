import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from rulechef import RuleChef
from rulechef.coordinator import AgenticCoordinator
from rulechef.core import RuleFormat, Task, TaskType
from rulechef.learner import RuleLearner

NER_DATA = Path(__file__).parent.parent / "examples" / "data" / "demo_ner_bulk.jsonl"
FAKE_RESPONSE = json.dumps(
    {
        "rules": [
            {
                "name": "drug_name",
                "format": "regex",
                "pattern": r"\b[A-Z][a-z]+(?:in|ol|il)\b",
                "output_template": {"text": "$0", "start": "$start", "end": "$end", "type": "DRUG"},
            },
            {
                "name": "dosage",
                "format": "regex",
                "pattern": r"\b\d+mg\b",
                "output_template": {
                    "text": "$0",
                    "start": "$start",
                    "end": "$end",
                    "type": "DOSAGE",
                },
            },
        ],
        "analysis": "ok",
        "actions": [],
    }
)


def _mock_client(response_text: str = FAKE_RESPONSE):
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = response
    return client


def _learner(**kwargs):
    return RuleLearner(
        MagicMock(), allowed_formats=[RuleFormat.REGEX], model="test-model", **kwargs
    )


def _ner_task():
    return Task(
        name="ner",
        description="Extract drugs, dosages, frequencies and conditions from text",
        input_schema={"text": "str"},
        output_schema={"entities": "list[dict]"},
        type=TaskType.NER,
    )


def _learn(storage_path, learner_client, coordinator_client):

    chef = RuleChef(
        task=_ner_task(),
        client=learner_client,
        coordinator=AgenticCoordinator(coordinator_client, prune_after_learn=True),
        dataset_name="repro",
        storage_path=str(storage_path),
        allowed_formats=[RuleFormat.REGEX],
        temperature=0,
        seed=42,
    )
    for line in NER_DATA.read_text().splitlines()[:5]:
        row = json.loads(line)
        chef.add_example(row["input"], row["output"])
    chef.learn_rules(max_refinement_iterations=1)
    return sorted((r.id, r.name) for r in chef.dataset.rules)


def test_same_input_gives_same_id():
    learner = _learner()
    kwargs = dict(
        name="a", content="b", format=RuleFormat.REGEX, output_template=None, output_key=None
    )
    assert RuleLearner._generate_id(**kwargs) == RuleLearner._generate_id(**kwargs)


def test_id_same_across_learners():
    first = _learner()
    second = _learner()
    kwargs = dict(
        name="a", content="b", format=RuleFormat.REGEX, output_template=None, output_key=None
    )
    assert first._generate_id(**kwargs) == second._generate_id(**kwargs)


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


def test_rules_with_same_id_get_suffix():
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


def test_learner_and_coordinator_calls_send_seed(tmp_path):
    learner_client, coordinator_client = _mock_client(), _mock_client()
    _learn(tmp_path, learner_client, coordinator_client)
    for client in (learner_client, coordinator_client):
        calls = client.chat.completions.create.call_args_list
        assert calls
        assert all(c.kwargs.get("seed") == 42 for c in calls)


def test_temp_kwargs_include_seed_and_temperature():
    learner = _learner(temperature=0, seed=42)
    assert learner._temp_kwargs() == {"temperature": 0, "seed": 42}


def test_two_runs_give_same_ids(tmp_path):
    first = _learn(tmp_path / "a", _mock_client(), _mock_client())
    second = _learn(tmp_path / "b", _mock_client(), _mock_client())
    assert first
    assert first == second


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
        client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        model=os.environ.get("LIVE_LLM_MODEL", "gpt-4o-mini"),
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
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("RUN_LIVE_TESTS"),
    reason="Requires OPENAI_API_KEY and RUN_LIVE_TESTS=1 for live LLM calls",
)
def test_real_llm_gives_same_rules(tmp_path):
    first = _learn_with_live_llm(tmp_path / "a")
    second = _learn_with_live_llm(tmp_path / "b")
    assert first
    assert first == second
