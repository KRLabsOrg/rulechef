# Advanced Features

## Observation Mode

RuleChef can learn from your existing LLM pipeline. Collect observations from **any LLM provider** — no task definition needed upfront.

### Structured observations (`add_observation`)

When you know the input/output shape, pass structured data directly:

```python
from rulechef import RuleChef

chef = RuleChef(client=client, model="gpt-4o-mini")  # No task needed

# Works with any LLM — Anthropic, Groq, local models, etc.
response = anthropic_client.messages.create(
    model="claude-3-haiku-20240307",
    messages=[{"role": "user", "content": f"Classify: {query}"}],
)
chef.add_observation(
    {"text": query},
    {"label": response.content[0].text.strip()},
)

# After collecting enough observations, learn rules
chef.learn_rules()
```

### Raw observations (`add_raw_observation`)

When you don't know the schema, pass raw messages and let RuleChef discover it:

```python
chef = RuleChef(client=client, model="gpt-4o-mini")

# Capture the raw interaction — RuleChef figures out the schema later
for query in queries:
    response = any_llm_call(query)
    chef.add_raw_observation(
        messages=[{"role": "user", "content": query}],
        response=response,
    )

# Discovers task schema + maps observations + learns rules
chef.learn_rules()
print(chef.task.to_dict())  # See what was discovered
```

### Auto-capture for OpenAI clients (`start_observing`)

For OpenAI-compatible clients, monkey-patch to capture calls automatically:

```python
wrapped = chef.start_observing(openai_client, auto_learn=False)

# Use wrapped as normal — every call is captured
response = wrapped.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": query}],
)

chef.learn_rules()   # Discovers + maps + learns
chef.stop_observing()
```

When `auto_learn=True`, learning triggers automatically based on the coordinator's decision. Streaming calls (`stream=True`) are also observed — RuleChef wraps the stream to capture content after it completes.

## Training Data Logger (Distillation)

RuleChef can capture every LLM call made during rule synthesis as structured training data, suitable for fine-tuning a smaller model to replace the LLM. The logger is fully optional — pass a `TrainingDataLogger` instance and all calls (synthesis, patching, coordination, auditing) are written to a JSONL file.

```python
from rulechef import RuleChef, TrainingDataLogger

logger = TrainingDataLogger(
    "training_data/run_001.jsonl",
    run_metadata={"model": "kimi-k2", "dataset": "banking77"},
)
chef = RuleChef(task, client, training_logger=logger)

chef.add_example(...)
chef.learn_rules()

print(logger.stats)   # {"rule_synthesis": 5, "rule_patch": 3, "guide_refinement": 10, ...}
print(logger.count)    # 18 total entries
```

### Output format

Each line in the JSONL file is a self-contained training example:

```json
{
  "messages": [
    {"role": "user", "content": "...prompt..."},
    {"role": "assistant", "content": "...response..."}
  ],
  "call_type": "rule_synthesis",
  "metadata": {
    "model": "kimi-k2",
    "dataset": "banking77",
    "task_name": "Intent Classification",
    "dataset_size": 25,
    "num_rules_in_response": 8,
    "response_valid": true
  },
  "timestamp": "2026-02-19T14:30:00+00:00"
}
```

For fine-tuning, use only the `messages` field. The `metadata` and `call_type` are for filtering — e.g. keep only entries where `response_valid` is true, or only runs where the final F1 exceeded a threshold.

### Call types

| Call type | Source | Description |
|-----------|--------|-------------|
| `rule_synthesis` | Learner | Bulk rule generation from examples |
| `rule_synthesis_per_class` | Learner | Per-class rule generation |
| `rule_patch` | Learner | Patch rules targeted at failures |
| `synthetic_generation` | Learner | Synthetic example generation |
| `guide_refinement` | Coordinator | Per-iteration refinement guidance |
| `audit_rules` | Coordinator | Rule pruning/merging audit |
| `trigger_decision` | Coordinator | Should-learn decision |

### Generating training data at scale

To generate a diverse training corpus, run RuleChef across multiple datasets with varied configurations:

```python
import itertools
from rulechef import RuleChef, TrainingDataLogger, AgenticCoordinator

datasets = ["banking77", "clinc150", "snips", ...]
shots = [3, 5, 10]

for ds_name, n_shots in itertools.product(datasets, shots):
    logger = TrainingDataLogger(
        f"training_data/{ds_name}_{n_shots}shot.jsonl",
        run_metadata={"dataset": ds_name, "shots": n_shots},
    )
    coordinator = AgenticCoordinator(client, training_logger=logger)
    chef = RuleChef(task, client, coordinator=coordinator, training_logger=logger)
    # ... add examples, learn rules ...
```

!!! tip
    The logger appends to the file, so multiple runs can safely write to the same path. Each entry carries its own `run_metadata` and timestamp.

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

[grex](https://github.com/pemistahl/grex) is a library that infers regex patterns from example strings. RuleChef uses it to give the LLM concrete pattern suggestions during rule synthesis.

### Install

grex is an optional dependency:

```bash
pip install rulechef[grex]
```

It's enabled by default when installed. To disable:

```python
chef = RuleChef(task, client, use_grex=False)
```

### What grex does

You give grex a list of strings, it gives you a regex that matches all of them:

```python
from grex import RegExpBuilder

dates = ["2024-01-15", "2024-02-28", "2023-12-01"]

# Exact: alternation of all inputs
RegExpBuilder.from_test_cases(dates).without_anchors().build()
# → '(2023\-12\-01|2024\-01\-15|2024\-02\-28)'

# Generalized: replaces digits/repetitions with character classes
RegExpBuilder.from_test_cases(dates).without_anchors() \
    .with_conversion_of_digits().with_conversion_of_repetitions().build()
# → '\d{4}\-\d{2}\-\d{2}'
```

The generalized pattern `\d{4}\-\d{2}\-\d{2}` matches any date in that format, not just the three examples. This is what makes grex valuable — it finds structure.

### How RuleChef uses it

During rule synthesis, RuleChef builds a "data evidence" section in the prompt. Without grex, the LLM only sees raw example strings:

```
DATA EVIDENCE FROM TRAINING:
- exchange_rate (3 examples): "what is the exchange rate for USD to EUR?", "how much is a dollar in euros?", "I want to know the current rates"
- card_arrival (3 examples): "my new card still hasn't arrived", "when will my new card be delivered?", "my card hasn't come in the mail yet"
```

With grex enabled, each group gets regex pattern suggestions appended:

```
DATA EVIDENCE FROM TRAINING:
- exchange_rate (3 examples): "what is the exchange rate for USD to EUR?", ...
  Exact pattern: (I want to know the current rates|how much is a dollar in euros\?|what is the exchange rate for USD to EUR\?)
- card_arrival (3 examples): "my new card still hasn't arrived", ...
  Exact pattern: (my card hasn't come in the mail yet|my new card still hasn't arrived|when will my new card be delivered\?)
```

grex generates two types of patterns:

- **Exact pattern** — alternation of all seen strings (always included)
- **Structural pattern** — generalized version with digit/repetition conversion (included when it's meaningfully shorter than the exact pattern)

For NER and transformation tasks with structured values (dates, IDs, codes), the structural pattern is especially valuable:

```
- DATE (5 unique): "2024-01-15", "2024-02-28", "2023-12-01", ...
  Exact pattern: (2023\-12\-01|2024\-01\-15|2024\-02\-28|...)
  Structural pattern: \d{4}\-\d{2}\-\d{2}
```

The structural pattern `\d{4}\-\d{2}\-\d{2}` tells the LLM to write a general date regex rather than hardcoding the specific dates.

### When grex helps most

- **Structured extraction** — dates, phone numbers, IDs, codes, amounts
- **NER** — entity strings with consistent patterns (drug names, gene symbols)
- **Classification with keyword clusters** — groups of similar input phrases

### When grex doesn't help

- Very long input strings (>80 chars are skipped)
- Fewer than 2 unique strings per group
- Highly diverse strings with no shared structure (exact pattern becomes a giant alternation that the LLM ignores)

### Debugging

Set the environment variable to see when grex is used:

```bash
RULECHEF_GREX_LOG=1 python your_script.py
```

This prints lines like `[rulechef][grex] used CLASSIFICATION:exchange_rate` whenever a pattern is generated.

## Code Rule Security

Code rules (`RuleFormat.CODE`) are executed via Python's `exec()` in a **restricted namespace**. The default `__builtins__` are replaced with a curated safe subset, so code rules cannot import modules, access the filesystem, or execute arbitrary code.

This means code rules can use:

- `re` — the standard library regex module
- `Span` — RuleChef's span dataclass for returning results
- Safe builtins — `len`, `str`, `int`, `float`, `bool`, `list`, `dict`, `set`, `tuple`, `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `reversed`, `min`, `max`, `sum`, `any`, `all`, `abs`, `round`, `isinstance`, `type`, `print`
- Basic Python syntax (loops, conditionals, string methods, list comprehensions)

Code rules **cannot**:

- Import modules (`import os`, `__import__('subprocess')`)
- Access files or environment variables
- Make network calls
- Call `open()`, `eval()`, `exec()`, `getattr()`, or `compile()`

If you need capabilities beyond this, use regex or spaCy rules instead.

## CLI

Interactive CLI for quick experimentation:

```bash
export OPENAI_API_KEY=your_key
rulechef
```
