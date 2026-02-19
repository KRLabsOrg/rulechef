# Task Types

RuleChef supports four task types, each with a canonical output format.

## Overview

| Type | Output Key | Output Format | Use Case |
|------|-----------|---------------|----------|
| `EXTRACTION` | `spans` | `List[Span]` | Find text spans (untyped) |
| `NER` | `entities` | `List[Entity]` | Find typed entities with labels |
| `CLASSIFICATION` | `label` | `str` | Classify text into categories |
| `TRANSFORMATION` | Custom | `Dict` | Extract structured fields |

## Extraction

Extraction finds text spans without type labels. Each span has `text`, `start`, and `end` fields.

```python
task = Task(
    name="Date Extraction",
    description="Extract date mentions from text",
    input_schema={"text": "str"},
    output_schema={"spans": "List[Span]"},
    type=TaskType.EXTRACTION,
)
```

**Output format:**
```json
{
  "spans": [
    {"text": "January 2024", "start": 10, "end": 22}
  ]
}
```

### Matching Modes

For extraction evaluation, you can choose how spans are compared:

```python
task = Task(
    ...,
    type=TaskType.EXTRACTION,
    matching_mode="text",   # Compare by span text only (default)
    # matching_mode="exact",  # Compare by text + start/end offsets
)
```

## NER (Named Entity Recognition)

NER extracts typed entities. Each entity has `text`, `start`, `end`, and `type` fields.

```python
from pydantic import BaseModel
from typing import List, Literal

class Entity(BaseModel):
    text: str
    start: int
    end: int
    type: Literal["PERSON", "ORG", "LOCATION"]

class NEROutput(BaseModel):
    entities: List[Entity]

task = Task(
    name="NER",
    description="Extract named entities",
    input_schema={"text": "str"},
    output_schema=NEROutput,
    type=TaskType.NER,
)
```

**Output format:**
```json
{
  "entities": [
    {"text": "Alice", "start": 0, "end": 5, "type": "PERSON"},
    {"text": "Acme Corp", "start": 15, "end": 24, "type": "ORG"}
  ]
}
```

!!! tip "Pydantic schemas"
    Using a Pydantic model with `Literal` type fields lets RuleChef automatically discover valid labels and validate outputs at runtime.

## Classification

Classification assigns a single label to each input.

```python
task = Task(
    name="Sentiment",
    description="Classify text sentiment",
    input_schema={"text": "str"},
    output_schema={"label": "str"},
    type=TaskType.CLASSIFICATION,
    text_field="text",
)
```

**Output format:**
```json
{"label": "positive"}
```

Classification matching is case-insensitive and strips whitespace.

## Transformation

Transformation extracts arbitrary structured fields. The output schema defines the target shape.

```python
task = Task(
    name="Contact Parser",
    description="Extract name and email from text",
    input_schema={"text": "str"},
    output_schema={"name": "str", "email": "str"},
    type=TaskType.TRANSFORMATION,
)
```

**Output format:**
```json
{"name": "Alice Smith", "email": "alice@example.com"}
```

## Input Schema and Text Field

### Multi-Field Inputs

Tasks can have multiple input fields:

```python
task = Task(
    name="Q&A",
    input_schema={"question": "str", "context": "str"},
    output_schema={"spans": "List[Span]"},
    type=TaskType.EXTRACTION,
    text_field="context",  # Regex/spaCy rules match against this field
)
```

### Text Field Selection

By default, regex and spaCy rules match against the longest string input field. Use `text_field` to specify which field to use:

```python
task = Task(
    ...,
    text_field="context",  # Explicit: use "context" field
)
```

## Rule Formats

Rules can be generated in three formats:

| Format | Best For | Speed |
|--------|----------|-------|
| `RuleFormat.REGEX` | Keyword patterns, structured text | Fastest |
| `RuleFormat.CODE` | Complex logic, multi-field extraction | Fast |
| `RuleFormat.SPACY` | Linguistic patterns (POS, dependency) | Moderate |

```python
from rulechef import RuleFormat

# Restrict to regex only (fastest, most portable)
chef = RuleChef(task, client, allowed_formats=[RuleFormat.REGEX])

# Allow code rules for complex logic
chef = RuleChef(task, client, allowed_formats=[RuleFormat.CODE])

# All formats
chef = RuleChef(task, client, allowed_formats=[RuleFormat.REGEX, RuleFormat.CODE, RuleFormat.SPACY])
```
