# Quick Start

This guide shows complete examples for all four task types.

## Setup

```python
from openai import OpenAI
from rulechef import RuleChef, Task, TaskType

client = OpenAI()  # Uses OPENAI_API_KEY env var
```

## Extraction

Extract untyped text spans from input:

```python
task = Task(
    name="Q&A Extraction",
    description="Extract answer spans from context",
    input_schema={"question": "str", "context": "str"},
    output_schema={"spans": "List[Span]"},
    type=TaskType.EXTRACTION,
)

chef = RuleChef(task, client)

chef.add_example(
    {"question": "When?", "context": "Built in 1991"},
    {"spans": [{"text": "1991", "start": 9, "end": 13}]}
)
chef.add_example(
    {"question": "When?", "context": "Released in 2005"},
    {"spans": [{"text": "2005", "start": 12, "end": 16}]}
)

rules, eval_result = chef.learn_rules()
print(f"Learned {len(rules)} rules")

result = chef.extract({"question": "When?", "context": "Founded in 1997"})
print(result)  # {"spans": [{"text": "1997", ...}]}
```

## Named Entity Recognition (NER)

Extract typed entities with Pydantic schema validation:

```python
from pydantic import BaseModel
from typing import List, Literal

class Entity(BaseModel):
    text: str
    start: int
    end: int
    type: Literal["DRUG", "DOSAGE", "CONDITION"]

class NEROutput(BaseModel):
    entities: List[Entity]

task = Task(
    name="Medical NER",
    description="Extract drugs, dosages, and conditions",
    input_schema={"text": "str"},
    output_schema=NEROutput,
    type=TaskType.NER,
)

chef = RuleChef(task, client)
chef.add_example(
    {"text": "Take Aspirin 500mg for headache"},
    {"entities": [
        {"text": "Aspirin", "start": 5, "end": 12, "type": "DRUG"},
        {"text": "500mg", "start": 13, "end": 18, "type": "DOSAGE"},
        {"text": "headache", "start": 23, "end": 31, "type": "CONDITION"},
    ]}
)
chef.learn_rules()
```

## Classification

Classify text into categories:

```python
task = Task(
    name="Intent Classification",
    description="Classify banking customer queries",
    input_schema={"text": "str"},
    output_schema={"label": "str"},
    type=TaskType.CLASSIFICATION,
    text_field="text",
)

chef = RuleChef(task, client)
chef.add_example({"text": "what is the exchange rate?"}, {"label": "exchange_rate"})
chef.add_example({"text": "I want to know the rates"}, {"label": "exchange_rate"})
chef.add_example({"text": "my card hasn't arrived"}, {"label": "card_arrival"})

chef.learn_rules()
result = chef.extract({"text": "current exchange rate please"})
print(result)  # {"label": "exchange_rate"}
```

## Transformation

Extract structured fields from text:

```python
task = Task(
    name="Invoice Parser",
    description="Extract company and amount from invoices",
    input_schema={"text": "str"},
    output_schema={"company": "str", "amount": "str"},
    type=TaskType.TRANSFORMATION,
)

chef = RuleChef(task, client)
chef.add_example(
    {"text": "Invoice from Acme Corp for $1,500.00"},
    {"company": "Acme Corp", "amount": "$1,500.00"}
)
chef.learn_rules()

result = chef.extract({"text": "Invoice from Globex Inc for $2,300.00"})
print(result)  # {"company": "Globex Inc", "amount": "$2,300.00"}
```

## What's Next?

- [Task Types](../guide/task-types.md) — detailed guide for each type
- [Learning & Refinement](../guide/learning.md) — buffer architecture and incremental patching
- [Evaluation & Feedback](../guide/evaluation.md) — measure and improve rule quality
