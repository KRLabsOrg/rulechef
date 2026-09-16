import json
from unittest.mock import MagicMock

from rulechef.core import Dataset, Example, Rule, RuleFormat, Task, TaskType
from rulechef.learner import RuleLearner
from rulechef.llm_calls import LLMCallConfig


def _mock_client(response_text: str):
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = response
    return client


def _classification_dataset():
    task = Task(
        name="intent",
        description="Classify intent",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
    )
    return Dataset(
        name="intent-data",
        task=task,
        examples=[
            Example(
                id="ex1",
                input={"text": "what is the exchange rate"},
                expected_output={"label": "exchange_rate"},
                source="test",
            )
        ],
    )


def _rule(name: str, label: str, content: str = "exchange") -> Rule:
    return Rule(
        id=name,
        name=name,
        description=f"Rule for {label}",
        format=RuleFormat.REGEX,
        content=content,
        priority=5,
        output_template={"label": label},
        output_key="label",
    )


def test_patch_synthesis_small_prompt_uses_full_variant():
    response_text = json.dumps(
        {
            "analysis": "add rule",
            "rules": [
                {
                    "name": "exchange_rule",
                    "description": "match exchange",
                    "format": "regex",
                    "content": "exchange",
                    "priority": 5,
                    "output_template": {"label": "exchange_rate"},
                }
            ],
            "deleted_rules": [],
        }
    )
    client = _mock_client(response_text)
    learner = RuleLearner(
        client,
        allowed_formats=[RuleFormat.REGEX],
        model="test-model",
        llm_config=LLMCallConfig(context_window=20000, patch_output_tokens=100),
    )
    dataset = _classification_dataset()

    rules, deleted = learner.synthesize_patch_ruleset(
        [_rule("old_exchange", "exchange_rate")],
        [
            {
                "input": {"text": "exchange rate today"},
                "expected": {"label": "exchange_rate"},
                "got": {},
            }
        ],
        dataset=dataset,
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert [r.name for r in rules] == ["exchange_rule"]
    assert deleted == set()
    assert kwargs["max_completion_tokens"] == 100
    assert "CURRENT RULES" in kwargs["messages"][0]["content"]


def test_patch_synthesis_uses_smaller_variant_when_full_prompt_exceeds_budget():
    response_text = json.dumps({"analysis": "no-op", "rules": [], "deleted_rules": []})
    client = _mock_client(response_text)
    learner = RuleLearner(
        client,
        allowed_formats=[RuleFormat.REGEX],
        model="test-model",
        llm_config=LLMCallConfig(
            context_window=5000, patch_output_tokens=100, safety_margin_tokens=0
        ),
    )
    dataset = _classification_dataset()
    huge = "x" * 12000
    current_rules = [
        _rule("exchange_rule", "exchange_rate", huge),
        _rule("cash_rule", "cash_withdrawal", huge),
        _rule("card_rule", "card_arrival", huge),
    ]

    learner.synthesize_patch_ruleset(
        current_rules,
        [
            {
                "input": {"text": "exchange rate today"},
                "expected": {"label": "exchange_rate"},
                "got": {},
            }
        ],
        dataset=dataset,
    )

    sent_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert client.chat.completions.create.call_count == 1
    assert "OTHER CURRENT RULES (compact index" in sent_prompt or len(sent_prompt) < huge.__len__()


def test_patch_synthesis_tiny_context_window_skips_provider_call():
    client = _mock_client('{"rules": []}')
    learner = RuleLearner(
        client,
        allowed_formats=[RuleFormat.REGEX],
        model="test-model",
        llm_config=LLMCallConfig(
            context_window=10, patch_output_tokens=100, safety_margin_tokens=0
        ),
    )
    dataset = _classification_dataset()

    rules, deleted = learner.synthesize_patch_ruleset(
        [_rule("old_exchange", "exchange_rate")],
        [
            {
                "input": {"text": "exchange rate today"},
                "expected": {"label": "exchange_rate"},
                "got": {},
            }
        ],
        dataset=dataset,
    )

    assert rules == []
    assert deleted == set()
    client.chat.completions.create.assert_not_called()


def test_patch_synthesis_supports_max_tokens_and_no_output_token_param():
    response_text = json.dumps({"analysis": "no-op", "rules": [], "deleted_rules": []})
    dataset = _classification_dataset()

    client = _mock_client(response_text)
    learner = RuleLearner(
        client,
        allowed_formats=[RuleFormat.REGEX],
        model="test-model",
        llm_config=LLMCallConfig(output_token_param="max_tokens", patch_output_tokens=123),
    )
    learner.synthesize_patch_ruleset(
        [_rule("old_exchange", "exchange_rate")],
        [{"input": {"text": "x"}, "expected": {"label": "exchange_rate"}, "got": {}}],
        dataset=dataset,
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 123
    assert "max_completion_tokens" not in kwargs

    client = _mock_client(response_text)
    learner = RuleLearner(
        client,
        allowed_formats=[RuleFormat.REGEX],
        model="test-model",
        llm_config=LLMCallConfig(output_token_param=None, patch_output_tokens=123),
    )
    learner.synthesize_patch_ruleset(
        [_rule("old_exchange", "exchange_rate")],
        [{"input": {"text": "x"}, "expected": {"label": "exchange_rate"}, "got": {}}],
        dataset=dataset,
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs
