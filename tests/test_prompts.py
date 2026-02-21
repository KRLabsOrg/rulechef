from rulechef.core import Dataset, Example, RuleFormat, Task, TaskType
from rulechef.prompts import PromptBuilder


def test_prompt_derives_labels_and_includes_evidence_for_dict_schema():
    task = Task(
        name="NER",
        description="Extract entities from text",
        input_schema={"text": "str"},
        output_schema={"entities": "list[dict]"},
        type=TaskType.NER,
    )
    dataset = Dataset(
        name="test",
        task=task,
        examples=[
            Example(
                id="ex1",
                input={"text": "KR Labs is a company"},
                expected_output={
                    "entities": [{"text": "KR Labs", "start": 0, "end": 7, "type": "ORG"}]
                },
                source="human_labeled",
            )
        ],
    )

    builder = PromptBuilder(allowed_formats=[RuleFormat.REGEX], use_spacy_ner=False)
    prompt = builder.build_synthesis_prompt(dataset, max_rules=5)

    # Output schema is always present.
    assert "Output schema:" in prompt
    assert str(task.output_schema) in prompt

    # Labels are derived from training data when the schema doesn't encode them.
    assert "AVAILABLE ENTITY TYPES: ['ORG']" in prompt
    assert "Rules MUST use one of these types in output_template." in prompt

    # Evidence section is present and bounded.
    assert "DATA EVIDENCE FROM TRAINING:" in prompt
    assert "- ORG" in prompt
    assert '"KR Labs"' in prompt

    # Hardcoded PER/ORG examples should not be present.
    assert "NER RULE EXAMPLES" not in prompt
    assert "corporate_suffixes" not in prompt

    # Regex guidance should appear; spaCy guidance should not.
    assert "REGEX TECHNIQUES:" in prompt
    assert "SPACY TECHNIQUES:" not in prompt


def test_prompt_includes_grex_patterns_when_available():
    import importlib.util

    has_grex = bool(importlib.util.find_spec("grex"))

    task = Task(
        name="NER",
        description="Extract entities from text",
        input_schema={"text": "str"},
        output_schema={"entities": "list[dict]"},
        type=TaskType.NER,
    )
    dataset = Dataset(
        name="test",
        task=task,
        examples=[
            Example(
                id="ex1",
                input={"text": "KR Labs is a company"},
                expected_output={
                    "entities": [{"text": "KR Labs", "start": 0, "end": 7, "type": "ORG"}]
                },
                source="human_labeled",
            ),
            Example(
                id="ex2",
                input={"text": "Acme Corp is hiring"},
                expected_output={
                    "entities": [{"text": "Acme Corp", "start": 0, "end": 9, "type": "ORG"}]
                },
                source="human_labeled",
            ),
        ],
    )

    builder = PromptBuilder(allowed_formats=[RuleFormat.REGEX], use_spacy_ner=False, use_grex=True)
    prompt = builder.build_synthesis_prompt(dataset, max_rules=5)

    # grex output should only appear if the dependency is installed.
    if has_grex:
        assert "Exact pattern:" in prompt
    else:
        assert "Exact pattern:" not in prompt
