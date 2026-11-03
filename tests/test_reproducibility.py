import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from rulechef import RuleChef
from rulechef.core import RuleFormat, Task, TaskType
from rulechef.learner import RuleLearner

live = pytest.mark.skipif("LIVE_LLM_BASE_URL" not in os.environ, reason="needs a live LLM endpoint")


def _learner(**kwargs):
    return RuleLearner(
        MagicMock(), allowed_formats=[RuleFormat.REGEX], model="test-model", **kwargs
    )


def test_same_input_gives_same_id():
    learner = _learner()
    assert learner._generate_id("a", "b") == learner._generate_id("a", "b")


def test_id_same_across_learners():
    first = _learner()
    second = _learner()
    assert first._generate_id("a", "b") == second._generate_id("a", "b")


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
