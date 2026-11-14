import sys
from unittest.mock import MagicMock

from rulechef import RuleChef
from rulechef.core import Rule, RuleFormat, Task, TaskType
from rulechef.executor import RuleExecutor


def test_rulechef_passes_spacy_model_to_executor(tmp_path):
    task = Task(
        name="ner",
        type=TaskType.NER,
        description="",
        input_schema={"text": "str"},
        output_schema={"entities": "list[dict]"},
    )
    chef = RuleChef(
        task=task, client=MagicMock(), storage_path=str(tmp_path), spacy_model="de_core_news_sm"
    )
    assert chef.learner.executor.spacy_model == "de_core_news_sm"


def test_executor_loads_configured_spacy_model(monkeypatch):
    fake_spacy = MagicMock()
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    monkeypatch.setitem(sys.modules, "spacy.matcher", fake_spacy.matcher)
    rule = Rule(
        id="r",
        name="r",
        description="",
        format=RuleFormat.SPACY,
        content='[{"LOWER": "test"}]',
    )
    RuleExecutor(spacy_model="de_core_news_sm")._execute_spacy_rule(rule, {"text": "test"})
    fake_spacy.load.assert_called_once_with("de_core_news_sm")
