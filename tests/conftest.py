"""Shared fixtures for RuleChef tests."""

from typing import Literal

import pytest
from pydantic import BaseModel

from rulechef.core import (
    Correction,
    Dataset,
    Example,
    Rule,
    RuleFormat,
    Task,
    TaskType,
)

# ---------------------------------------------------------------------------
# Pydantic models used in fixtures
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    text: str
    start: int
    end: int
    type: Literal["PERSON", "ORG", "DRUG"]


class NEROutput(BaseModel):
    entities: list[Entity]


# ---------------------------------------------------------------------------
# Task fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extraction_task():
    """Simple extraction task with dict schema."""
    return Task(
        name="extract_emails",
        description="Extract email addresses from text",
        input_schema={"text": "string"},
        output_schema={"spans": "List[Span]"},
        type=TaskType.EXTRACTION,
    )


@pytest.fixture
def ner_task():
    """NER task with Pydantic schema carrying Literal labels."""
    return Task(
        name="ner_entities",
        description="Extract named entities",
        input_schema={"text": "string"},
        output_schema=NEROutput,
        type=TaskType.NER,
    )


@pytest.fixture
def classification_task():
    """Classification task with dict schema."""
    return Task(
        name="sentiment",
        description="Classify sentiment of text",
        input_schema={"text": "string"},
        output_schema={"label": "string"},
        type=TaskType.CLASSIFICATION,
    )


@pytest.fixture
def transformation_task():
    """Transformation task."""
    return Task(
        name="extract_structured",
        description="Extract structured fields",
        input_schema={"text": "string"},
        output_schema={"company": "str", "amount": "str"},
        type=TaskType.TRANSFORMATION,
    )


# ---------------------------------------------------------------------------
# Rule fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_regex_rule():
    """A simple regex rule for email extraction."""
    return Rule(
        id="rule-email-1",
        name="email_pattern",
        description="Match email addresses",
        format=RuleFormat.REGEX,
        content=r"[\w.+-]+@[\w-]+\.[\w.]+",
        priority=5,
        confidence=0.8,
    )


@pytest.fixture
def sample_ner_regex_rule():
    """Regex rule with output_template for NER."""
    return Rule(
        id="rule-ner-drug",
        name="drug_pattern",
        description="Match drug names after dosage",
        format=RuleFormat.REGEX,
        content=r"\b(\d+\s*mg)\s+(\w+)\b",
        priority=5,
        confidence=0.7,
        output_template={
            "text": "$0",
            "start": "$start",
            "end": "$end",
            "type": "DRUG",
        },
        output_key="entities",
    )


@pytest.fixture
def sample_code_rule():
    """A simple code rule."""
    return Rule(
        id="rule-code-1",
        name="code_extractor",
        description="Extract via Python function",
        format=RuleFormat.CODE,
        content="""
def extract(input_data):
    text = input_data.get("text", "")
    spans = []
    for word in text.split():
        if "@" in word:
            start = text.index(word)
            spans.append({"text": word, "start": start, "end": start + len(word)})
    return spans
""",
        priority=3,
        confidence=0.6,
    )


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset(extraction_task):
    """A small dataset with examples and corrections for extraction."""
    examples = [
        Example(
            id="ex-1",
            input={"text": "Contact us at hello@example.com for details."},
            expected_output={
                "spans": [{"text": "hello@example.com", "start": 14, "end": 31}]
            },
            source="human_labeled",
        ),
        Example(
            id="ex-2",
            input={"text": "No emails here."},
            expected_output={"spans": []},
            source="llm_generated",
        ),
    ]
    corrections = [
        Correction(
            id="cor-1",
            input={"text": "Send to admin@test.org please."},
            model_output={"spans": []},
            expected_output={
                "spans": [{"text": "admin@test.org", "start": 8, "end": 22}]
            },
            feedback="Missed email address",
        ),
    ]
    return Dataset(
        name="email_dataset",
        task=extraction_task,
        description="Test email dataset",
        examples=examples,
        corrections=corrections,
    )
