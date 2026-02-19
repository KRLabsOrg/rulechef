# Advanced Features

## Observation Mode

Passively observe an existing OpenAI client to collect training data:

```python
wrapped_client = chef.start_observing(openai_client, auto_learn=True)

# Use wrapped_client as normal — RuleChef collects examples
response = wrapped_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
)

chef.stop_observing()
```

When `auto_learn=True`, learning triggers automatically based on the coordinator's decision.

## Pydantic Output Schemas

Use Pydantic models for type-safe, validated outputs:

```python
from pydantic import BaseModel
from typing import List, Literal

class Entity(BaseModel):
    text: str
    start: int
    end: int
    type: Literal["PERSON", "ORG", "LOCATION"]

class Output(BaseModel):
    entities: List[Entity]

task = Task(
    name="NER",
    description="Extract entities",
    input_schema={"text": "str"},
    output_schema=Output,
    type=TaskType.NER,
)
```

RuleChef automatically:

- Discovers valid labels from `Literal` type annotations
- Validates rule outputs against the model at runtime
- Generates readable schema fragments for synthesis prompts

## Output Templates

Rules can emit structured JSON using template variables:

### Regex Templates

| Variable | Meaning |
|----------|---------|
| `$0` | Full match text |
| `$1`, `$2`, ... | Capture groups |
| `$start`, `$end` | Match offsets |

```json
{
  "output_template": {
    "text": "$1",
    "type": "DRUG",
    "start": "$start",
    "end": "$end"
  }
}
```

### spaCy Templates

| Variable | Meaning |
|----------|---------|
| `$1.text`, `$2.text` | Token text |
| `$1.start`, `$1.end` | Token character offsets |

## spaCy Patterns

### Token Matcher

Use token attributes for linguistic patterns:

```json
[
  {"POS": "PROPN", "OP": "+"},
  {"POS": "NOUN"}
]
```

Available attributes: `TEXT`, `LOWER`, `POS`, `TAG`, `DEP`, `LEMMA`, `SHAPE`, `IS_ALPHA`, `IS_DIGIT`, `OP`.

### Dependency Matcher

Match syntactic structure:

```json
[
  {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
  {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"DEP": "nsubj"}}
]
```

!!! note "spaCy NER"
    By default, `use_spacy_ner=False` — spaCy's NER pipe is disabled and patterns relying on `ENT_TYPE` are rejected. Set `use_spacy_ner=True` to enable.

## LLM Fallback

When rules produce no results, optionally fall back to direct LLM extraction:

```python
chef = RuleChef(task, client, llm_fallback=True)

result = chef.extract({"text": "unusual input"})
# If no rule matches → calls LLM directly
```

## Using grex for Regex Suggestions

[grex](https://github.com/pemistahl/grex) generates regex patterns from example strings. When enabled, RuleChef includes grex-suggested patterns in synthesis prompts:

```python
chef = RuleChef(task, client, use_grex=True)  # Default
```

This helps the LLM generate more accurate regex rules, especially for structured patterns.

## CLI

Interactive CLI for quick experimentation:

```bash
export OPENAI_API_KEY=your_key
rulechef
```
